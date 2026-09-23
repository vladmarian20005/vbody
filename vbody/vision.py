"""ChArUco board, camera intrinsics, and the wrist-camera reading.

The board is printed once and taped flat to the table. The wrist camera
observes it; from the board pose in the camera frame we get the camera's
optical centre in the board frame, which is the point vBody measures. No
marker is attached to the arm.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

# Printed board geometry. Change these only if you print a different board,
# and then pass the same numbers to every command.
SQUARES_X = 7
SQUARES_Y = 5
SQUARE_M = 0.030
MARKER_M = 0.022


def make_board(
    squares_x: int = SQUARES_X,
    squares_y: int = SQUARES_Y,
    square_m: float = SQUARE_M,
    marker_m: float = MARKER_M,
) -> cv2.aruco.CharucoBoard:
    """The ChArUco board, described the same way for printing and detection."""
    return cv2.aruco.CharucoBoard((squares_x, squares_y), square_m, marker_m, DICT)


def render_board_image(board: cv2.aruco.CharucoBoard, px_per_m: int = 10000) -> np.ndarray:
    """Rasterize the board for printing.

    The canvas is an exact integer pixel count per square, so every square
    comes out at identical, exact scale. Sizing the whole canvas and truncating
    stretches the board a fraction of a percent undersized and anisotropically,
    which is enough to bias every later measurement.
    """
    sx, sy = board.getChessboardSize()
    px_per_square = round(board.getSquareLength() * px_per_m)
    return board.generateImage((sx * px_per_square, sy * px_per_square), marginSize=0, borderBits=1)


@dataclass
class BoardPose:
    """Board pose in the camera frame."""

    rvec: np.ndarray  # (3,) Rodrigues
    tvec: np.ndarray  # (3,) m, board origin in the camera frame
    n_corners: int


@dataclass
class Intrinsics:
    """Pinhole intrinsics and lens distortion for one camera."""

    camera_matrix: np.ndarray  # (3, 3)
    dist_coeffs: np.ndarray  # (5,)
    rms_reproj_px: float
    n_views: int
    image_size: tuple[int, int] | None = None

    def save(self, path: str | Path) -> None:
        blob = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "rms_reproj_px": self.rms_reproj_px,
            "n_views": self.n_views,
            "image_size": list(self.image_size) if self.image_size else None,
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        Path(path).write_text(json.dumps(blob, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Intrinsics:
        blob = json.loads(Path(path).read_text())
        size = blob.get("image_size")
        return cls(
            camera_matrix=np.array(blob["camera_matrix"], dtype=float),
            dist_coeffs=np.array(blob["dist_coeffs"], dtype=float),
            rms_reproj_px=float(blob.get("rms_reproj_px", float("nan"))),
            n_views=int(blob.get("n_views", 0)),
            image_size=tuple(size) if size else None,
        )


def calibrate_intrinsics(
    images: list[np.ndarray], board: cv2.aruco.CharucoBoard, min_corners: int = 6
) -> Intrinsics:
    """Fit intrinsics from board views. Wants roughly thirty varied views.

    Views should span the frame and tilt the board through a range of oblique
    angles; near-identical views make the fit degenerate. k3 is held at zero
    because views of a planar target barely constrain it, so it fits noise.
    """
    detector = cv2.aruco.CharucoDetector(board)
    obj_all: list[np.ndarray] = []
    img_all: list[np.ndarray] = []
    image_size = None
    for image in images:
        corners, ids, _, _ = detector.detectBoard(image)
        if corners is None or len(corners) < min_corners:
            continue
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        if obj_pts is None or len(obj_pts) < min_corners:
            continue
        obj_all.append(obj_pts.astype(np.float32))
        img_all.append(img_pts.astype(np.float32))
        image_size = (image.shape[1], image.shape[0])
    if len(obj_all) < 3:
        raise ValueError(
            f"only {len(obj_all)} usable board views (need at least 3, want about 30); "
            "check focus, lighting, and that the whole board is in frame"
        )
    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        obj_all, img_all, image_size, None, None, flags=cv2.CALIB_FIX_K3
    )
    return Intrinsics(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs.ravel(),
        rms_reproj_px=float(rms),
        n_views=len(obj_all),
        image_size=image_size,
    )


def estimate_board_pose(
    image: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    min_corners: int = 6,
) -> BoardPose | None:
    """Detect the board and solve for its pose. None if it is not seen well."""
    detector = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = detector.detectBoard(image)
    if corners is None or len(corners) < min_corners:
        return None
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerSubPix(
        gray,
        corners.astype(np.float32),
        winSize=(5, 5),
        zeroZone=(-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4),
    )
    obj_pts, img_pts = board.matchImagePoints(corners, ids)
    if obj_pts is None or len(obj_pts) < min_corners:
        return None
    # IPPE is the planar-target solver; LM then polishes reprojection error.
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE
    )
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, camera_matrix, dist_coeffs, rvec, tvec)
    return BoardPose(rvec=rvec.ravel(), tvec=tvec.ravel(), n_corners=len(corners))


@dataclass
class Reading:
    """One averaged wrist-camera measurement."""

    position_m: np.ndarray  # (3,) camera optical centre in the board frame
    n_corners: int  # fewest corners used in any averaged frame
    spread_mm: float  # largest per-axis standard deviation across frames


class WristCamera:
    """The wrist camera, read while the arm is standing still."""

    def __init__(
        self,
        camera_index: int,
        intrinsics: Intrinsics,
        board: cv2.aruco.CharucoBoard | None = None,
        min_corners: int = 10,
    ):
        self.intrinsics = intrinsics
        self.min_corners = min_corners
        self.board = board if board is not None else make_board()
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {camera_index}")
        # Intrinsics are only valid at the resolution they were fitted at, and
        # a camera may well open at another one.
        if intrinsics.image_size:
            width, height = intrinsics.image_size
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        for _ in range(10):  # let auto-exposure settle
            ok, frame = self.cap.read()
        if intrinsics.image_size and ok and (frame.shape[1], frame.shape[0]) != tuple(intrinsics.image_size):
            self.cap.release()
            raise RuntimeError(
                f"camera {camera_index} delivers {frame.shape[1]}x{frame.shape[0]}, but the "
                f"intrinsics were fitted at {intrinsics.image_size[0]}x{intrinsics.image_size[1]}"
            )

    def measure(self, n_frames: int = 5, max_tries: int = 40) -> Reading | None:
        """Average n_frames good views into one position in the board frame.

        Returns None when the board is not reliably visible. Callers must
        treat that as unmeasured, never as a zero.
        """
        positions: list[np.ndarray] = []
        corners: list[int] = []
        k = self.intrinsics.camera_matrix
        tries = 0
        while len(positions) < n_frames and tries < max_tries:
            ok, frame = self.cap.read()
            tries += 1
            if not ok:
                continue
            undistorted = cv2.undistort(frame, k, self.intrinsics.dist_coeffs)
            pose = estimate_board_pose(undistorted, self.board, k, np.zeros(5))
            if pose is None or pose.n_corners < self.min_corners:
                continue
            rmat, _ = cv2.Rodrigues(pose.rvec)
            positions.append(-rmat.T @ pose.tvec)  # camera centre, board frame
            corners.append(pose.n_corners)
        if len(positions) < n_frames:
            return None
        stack = np.array(positions)
        return Reading(
            position_m=stack.mean(axis=0),
            n_corners=int(min(corners)),
            spread_mm=float(stack.std(axis=0).max() * 1000),
        )

    def close(self) -> None:
        self.cap.release()


def capture_views(
    camera_index: int,
    board: cv2.aruco.CharucoBoard,
    n_views: int = 35,
    min_corners: int = 15,
    novelty_rot_rad: float = 0.12,
    novelty_pos_m: float = 0.05,
) -> list[np.ndarray]:
    """Collect calibration views by hand, one per sufficiently new viewpoint.

    Put the board flat on the table and move the camera around it. Coverage is
    what matters: strong tilts, all four sides, near and far, board near the
    frame corners. Ctrl-C stops early and keeps what it has.
    """
    detector = cv2.aruco.CharucoDetector(board)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {camera_index}")
    for _ in range(10):
        cap.read()
    ok, probe = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("camera read failed")
    h, w = probe.shape[:2]
    # A rough guess at the intrinsics, used only to judge whether a view is
    # new. It is never stored and never reaches the calibration.
    k_rough = np.array([[w * 0.8, 0, w / 2], [0, w * 0.8, h / 2], [0, 0, 1.0]])

    print(f"camera {camera_index} at {w}x{h}; want {n_views} views. Ctrl-C stops early.")
    views: list[np.ndarray] = []
    seen: list[tuple[np.ndarray, np.ndarray]] = []
    last_message = 0.0
    try:
        while len(views) < n_views:
            ok, frame = cap.read()
            if not ok:
                continue
            corners, ids, _, _ = detector.detectBoard(frame)
            now = time.time()
            if corners is None or len(corners) < min_corners:
                if now - last_message > 2:
                    print(f"  [{len(views)}/{n_views}] board not in view, or too shallow")
                    last_message = now
                continue
            obj_pts, img_pts = board.matchImagePoints(corners, ids)
            if obj_pts is None or len(obj_pts) < min_corners:
                continue
            pok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, k_rough, np.zeros(5), flags=cv2.SOLVEPNP_IPPE
            )
            if not pok:
                continue
            new = all(
                np.linalg.norm(rvec.ravel() - r) > novelty_rot_rad
                or np.linalg.norm(tvec.ravel() - t) > novelty_pos_m
                for r, t in seen
            )
            if not new:
                if now - last_message > 2:
                    print(f"  [{len(views)}/{n_views}] already have this angle, move or tilt more")
                    last_message = now
                continue
            seen.append((rvec.ravel().copy(), tvec.ravel().copy()))
            views.append(frame.copy())
            print(f"  [{len(views)}/{n_views}] captured, {len(corners)} corners")
    except KeyboardInterrupt:
        print(f"\nstopped early with {len(views)} views")
    finally:
        cap.release()
    return views
