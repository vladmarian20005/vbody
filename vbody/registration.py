"""Registration of the board frame to the arm base frame, and pose errors.

Forward kinematics predicts the camera in the arm's base frame; the camera
measures itself in the board frame. Comparing the two needs the rigid
transform between those frames, which is a property of how the board happens
to lie on the table and has to be fitted from the data.

The fit uses six parameters, a rotation and a translation, and holds the
camera offset at the value the user measured, so no part of the arm's error
can be absorbed into it. ``fit_offset=True`` adds the three offset components:
useful when the offset was estimated by eye, but it lets the fit soak up some
real error. The transform is always fitted on the reported joint angles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from vbody.kinematics import JOINT_NAMES, ArmKinematics

MIN_POSES_FOR_FIT = 4

# The sign check refits nine parameters per flip; below this many poses it
# would be fitting noise.
MIN_POSES_FOR_SIGN_CHECK = 8
# A flip that leaves less than this fraction of the residual is a sign to fix.
SIGN_CHECK_RATIO = 0.5


def kabsch(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best-fit rigid transform taking a to b: b is approximately R @ a + t."""
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    h = (a - ca).T @ (b - cb)
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, cb - r @ ca


@dataclass
class Registration:
    """Base frame to board frame, plus the camera offset it was fitted with."""

    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,) m
    camera_offset: np.ndarray  # (3,) m, wrist frame
    offset_was_fitted: bool
    rms_residual_mm: float
    n_poses: int
    converged: bool

    def to_board(self, points_base: np.ndarray) -> np.ndarray:
        """Map base-frame points into the board frame."""
        return np.atleast_2d(points_base) @ self.rotation.T + self.translation

    def to_dict(self) -> dict:
        return {
            "rotation": self.rotation.tolist(),
            "translation_m": self.translation.tolist(),
            "camera_offset_mm": (self.camera_offset * 1000).round(2).tolist(),
            "camera_offset_fitted": self.offset_was_fitted,
            "rms_residual_mm": self.rms_residual_mm,
            "n_poses": self.n_poses,
            "converged": self.converged,
        }


def fit_registration(
    kin: ArmKinematics,
    q_reported: np.ndarray,
    positions_measured: np.ndarray,
    camera_offset: np.ndarray,
    fit_offset: bool = False,
) -> Registration:
    """Fit the board-to-base transform on reported angles and measurements."""
    q_reported = np.atleast_2d(np.asarray(q_reported, dtype=float))
    positions_measured = np.atleast_2d(np.asarray(positions_measured, dtype=float))
    camera_offset = np.asarray(camera_offset, dtype=float)
    if len(q_reported) < MIN_POSES_FOR_FIT:
        raise ValueError(
            f"need at least {MIN_POSES_FOR_FIT} measured poses to fit the board "
            f"transform, have {len(q_reported)}"
        )

    predicted = kin.fk_batch(q_reported, camera_offset)
    r0, t0 = kabsch(predicted, positions_measured)
    x0 = np.concatenate([Rotation.from_matrix(r0).as_rotvec(), t0])
    if fit_offset:
        x0 = np.concatenate([x0, camera_offset])

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        offset = x[6:9] if fit_offset else camera_offset
        return Rotation.from_rotvec(x[:3]).as_matrix(), x[3:6], offset

    def residuals(x: np.ndarray) -> np.ndarray:
        r, t, offset = unpack(x)
        return (kin.fk_batch(q_reported, offset) @ r.T + t - positions_measured).ravel()

    result = least_squares(residuals, x0, method="lm", xtol=1e-12, ftol=1e-12)
    r, t, offset = unpack(result.x)
    per_pose = np.linalg.norm(
        kin.fk_batch(q_reported, offset) @ r.T + t - positions_measured, axis=1
    )
    return Registration(
        rotation=r,
        translation=t,
        camera_offset=offset,
        offset_was_fitted=fit_offset,
        rms_residual_mm=float(np.sqrt((per_pose**2).mean()) * 1000),
        n_poses=len(q_reported),
        converged=bool(result.success),
    )


def pose_errors(
    kin: ArmKinematics,
    q_commanded: np.ndarray,
    q_reported: np.ndarray,
    positions_measured: np.ndarray,
    reg: Registration,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pose distance in mm from the camera measurement to forward
    kinematics, evaluated first at the commanded angles and then at the
    angles the servos reported."""
    offset = reg.camera_offset
    predicted_commanded = reg.to_board(kin.fk_batch(q_commanded, offset))
    predicted_reported = reg.to_board(kin.fk_batch(q_reported, offset))
    measured = np.atleast_2d(positions_measured)
    return (
        np.linalg.norm(predicted_commanded - measured, axis=1) * 1000,
        np.linalg.norm(predicted_reported - measured, axis=1) * 1000,
    )


def summarize(errors_mm: np.ndarray) -> dict:
    """Mean, RMS and worst case of one error column, in mm."""
    errors_mm = np.asarray(errors_mm, dtype=float)
    if errors_mm.size == 0:
        return {"mean_mm": float("nan"), "rms_mm": float("nan"), "max_mm": float("nan")}
    return {
        "mean_mm": float(errors_mm.mean()),
        "rms_mm": float(np.sqrt((errors_mm**2).mean())),
        "max_mm": float(errors_mm.max()),
    }


def joint_sign_check(
    kin: ArmKinematics,
    q_reported: np.ndarray,
    positions_measured: np.ndarray,
    camera_offset: np.ndarray,
) -> dict:
    """Flip the recorded sign of each joint in turn and refit the board
    transform and the camera offset.

    A joint whose sign in the calibration file runs against the model's axis
    produces reported angles that no placement of the board or the camera
    can explain. Flipped, they suddenly can. Such a flip is a sign to fix in
    the calibration file, not something to fit: left in, the body model
    absorbs it as a wrist folded back on itself and a link of negative
    length, and every parameter it prints is then a fiction.
    """
    q_reported = np.atleast_2d(np.asarray(q_reported, dtype=float))
    positions_measured = np.atleast_2d(np.asarray(positions_measured, dtype=float))
    if len(q_reported) < MIN_POSES_FOR_SIGN_CHECK:
        raise ValueError(
            f"need at least {MIN_POSES_FOR_SIGN_CHECK} measured poses to check the joint "
            f"signs, have {len(q_reported)}"
        )
    base = fit_registration(
        kin, q_reported, positions_measured, camera_offset, fit_offset=True
    ).rms_residual_mm
    flipped = {}
    for j, name in enumerate(JOINT_NAMES):
        q = q_reported.copy()
        q[:, j] = -q[:, j]
        flipped[name] = fit_registration(
            kin, q, positions_measured, camera_offset, fit_offset=True
        ).rms_residual_mm
    suspect = [name for name, value in flipped.items() if value < SIGN_CHECK_RATIO * base]
    return {"rms_mm": base, "flipped_rms_mm": flipped, "suspect": suspect}
