"""Forward kinematics for the SO-ARM101, and the pose sampler.

The chain constants (link offsets, joint anchors and axes) are read once from
the vendored MJCF in ``assets/``. Nothing here runs a simulation; MuJoCo is
only the parser for the model of record.

Joint angles are radians, base to wrist, in the order the servos are chained.
The base frame has +z up and its origin at the table surface, so a point's z
is its height above the table. The tool point is the wrist link
(``Fixed_Jaw``) frame origin displaced by ``tool_offset``, expressed in the
wrist frame: here, the wrist camera's optical centre.

``ArmKinematics`` takes a scale per link and a zero offset per joint. With
both left at their defaults it is the nominal model. The body model fitted by
``vbody fit`` is the same chain with those two vectors set to what one arm
measured as; the simulated arm uses them to carry a deliberate error.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
NUM_JOINTS = len(JOINT_NAMES)

# Parent-ordered chain from the base to the wrist link.
CHAIN = ["Base", "Rotation_Pitch", "Upper_Arm", "Lower_Arm", "Wrist_Pitch_Roll", "Fixed_Jaw"]
WRIST_BODY = "Fixed_Jaw"

# The links whose offset from their parent can be scaled. The base itself has
# no offset to scale.
LINK_NAMES = CHAIN[1:]
NUM_LINKS = len(LINK_NAMES)

# Midpoint between the jaw pads, in the wrist frame. Not a tool point here;
# it is the lowest-hanging part of the gripper and so the point the table
# clearance check is applied to.
JAW_POINT = np.array([0.0, -0.09, 0.01])

MODEL_PATH = Path(__file__).with_name("assets") / "so_arm101.xml"


def _rot_from_quat(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix from a MuJoCo (w, x, y, z) quaternion."""
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, quat)
    return m.reshape(3, 3)


@lru_cache(maxsize=1)
def _load_chain() -> tuple:
    """Parse the MJCF once. Each step is (offset, rotation, scale index, joint
    index, anchor, K, K@K) where K is the joint axis cross-product matrix;
    scale and joint indices are -1 where the body has none."""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    joint_of_body = {}
    for i, name in enumerate(JOINT_NAMES):
        joint = model.joint(name)
        joint_of_body[model.body(joint.bodyid[0]).name] = (i, joint.pos.copy(), joint.axis.copy())
    steps = []
    for name in CHAIN:
        body = model.body(name)
        scale_idx = LINK_NAMES.index(name) if name in LINK_NAMES else -1
        jidx, anchor, kmat, kmat2 = -1, None, None, None
        if name in joint_of_body:
            jidx, anchor, axis = joint_of_body[name]
            a = axis / np.linalg.norm(axis)
            kmat = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            kmat2 = kmat @ kmat
        steps.append(
            (body.pos.copy(), _rot_from_quat(body.quat.copy()), scale_idx, jidx, anchor, kmat, kmat2)
        )
    return tuple(steps)


class ArmKinematics:
    """Forward kinematics over the five-joint chain.

    ``link_scale`` has one entry per ``LINK_NAMES`` and multiplies that link's
    offset from its parent. ``joint_offset`` has one entry per joint, in
    radians, and is subtracted from the commanded angle: the servo's zero is
    not quite the model's zero. Both default to the nominal model.
    """

    def __init__(
        self,
        link_scale: np.ndarray | None = None,
        joint_offset: np.ndarray | None = None,
    ):
        self.link_scale = (
            np.ones(NUM_LINKS) if link_scale is None else np.asarray(link_scale, dtype=float)
        )
        self.joint_offset = (
            np.zeros(NUM_JOINTS) if joint_offset is None else np.asarray(joint_offset, dtype=float)
        )
        if self.link_scale.shape != (NUM_LINKS,):
            raise ValueError(f"expected {NUM_LINKS} link scales, got shape {self.link_scale.shape}")
        if self.joint_offset.shape != (NUM_JOINTS,):
            raise ValueError(
                f"expected {NUM_JOINTS} joint offsets, got shape {self.joint_offset.shape}"
            )
        self._steps = _load_chain()
        self._eye3 = np.eye(3)

    def _scaled(self, offset: np.ndarray, scale_idx: int) -> np.ndarray:
        return offset * self.link_scale[scale_idx] if scale_idx >= 0 else offset

    def fk_jac(
        self, q: np.ndarray, tool_offset: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tool position (m) and its (3, 5) Jacobian with respect to the joint
        angles, from one pass over the chain. For a revolute joint with world
        axis z through world anchor a, the column is z x (p - a)."""
        q = np.asarray(q, dtype=float)
        if q.shape != (NUM_JOINTS,):
            raise ValueError(f"expected {NUM_JOINTS} joint angles, got shape {q.shape}")
        rot = self._eye3
        pos = np.zeros(3)
        axes = np.zeros((NUM_JOINTS, 3))
        anchors = np.zeros((NUM_JOINTS, 3))
        for offset, body_rot, scale_idx, jidx, anchor, kmat, kmat2 in self._steps:
            pos = pos + rot @ self._scaled(offset, scale_idx)
            rot = rot @ body_rot
            if jidx >= 0:
                angle = q[jidx] - self.joint_offset[jidx]
                axes[jidx] = rot @ np.array([kmat[2, 1], kmat[0, 2], kmat[1, 0]])
                anchors[jidx] = pos + rot @ anchor
                joint_rot = self._eye3 + np.sin(angle) * kmat + (1.0 - np.cos(angle)) * kmat2
                pos = pos + rot @ (anchor - joint_rot @ anchor)
                rot = rot @ joint_rot
        tool = np.zeros(3) if tool_offset is None else np.asarray(tool_offset, dtype=float)
        point = pos + rot @ tool
        return point, np.cross(axes, point - anchors).T

    def fk(self, q: np.ndarray, tool_offset: np.ndarray | None = None) -> np.ndarray:
        """Position of the tool point in the base frame (m) at joint angles q."""
        return self.fk_jac(q, tool_offset)[0]

    def fk_batch(self, qs: np.ndarray, tool_offset: np.ndarray | None = None) -> np.ndarray:
        """Tool positions for a stack of joint vectors, shape (n, 3)."""
        qs = np.atleast_2d(np.asarray(qs, dtype=float))
        if qs.shape[1] != NUM_JOINTS:
            raise ValueError(f"expected {NUM_JOINTS} joint angles per pose, got {qs.shape}")
        n = len(qs)
        rot = np.broadcast_to(self._eye3, (n, 3, 3))
        pos = np.zeros((n, 3))
        for offset, body_rot, scale_idx, jidx, anchor, kmat, kmat2 in self._steps:
            pos = pos + rot @ self._scaled(offset, scale_idx)
            rot = rot @ body_rot
            if jidx >= 0:
                angle = qs[:, jidx] - self.joint_offset[jidx]
                joint_rot = (
                    self._eye3
                    + np.sin(angle)[:, None, None] * kmat
                    + (1.0 - np.cos(angle))[:, None, None] * kmat2
                )
                swing = anchor - joint_rot @ anchor
                pos = pos + (rot @ swing[:, :, None])[:, :, 0]
                rot = rot @ joint_rot
        tool = np.zeros(3) if tool_offset is None else np.asarray(tool_offset, dtype=float)
        return pos + rot @ tool

    def ik(
        self,
        target: np.ndarray,
        q_seed: np.ndarray,
        tool_offset: np.ndarray | None = None,
        iterations: int = 30,
        damping: float = 1e-3,
        tol_m: float = 1e-5,
        joint_limits: np.ndarray | None = None,
    ) -> np.ndarray:
        """Damped least-squares inverse kinematics from a seed pose.

        Five joints against a three-dimensional target leave slack, and the
        seed decides how it is used: the answer is the small move from the
        seed that puts the tool point on the target. Nothing checks that the
        target was reached; the caller measures the miss and decides.
        """
        target = np.asarray(target, dtype=float)
        q = np.array(q_seed, dtype=float)
        damp = damping * self._eye3
        for _ in range(iterations):
            pos, jac = self.fk_jac(q, tool_offset)
            err = target - pos
            if err @ err < tol_m * tol_m:
                break
            q = q + jac.T @ np.linalg.solve(jac @ jac.T + damp, err)
            if joint_limits is not None:
                q = np.clip(q, joint_limits[:, 0], joint_limits[:, 1])
        return q


# ---- pose sampling ------------------------------------------------------

# Clearance every commanded pose must leave between the gripper and the table.
# Generous on purpose: the model is nominal, and the real arm sags below what
# the nominal model believes.
TABLE_CLEARANCE_M = 0.10

# Conservative sampling box, inside the joint limits of a healthy arm and
# biased towards poses with the arm reaching forward over the table, where the
# wrist camera can see a board lying in front of the base.
POSE_LOW = np.array([-1.2, -1.6, 0.2, -1.0, -1.5])
POSE_HIGH = np.array([1.2, 0.0, 2.4, 1.0, 1.5])


def workspace_safe(
    kin: ArmKinematics,
    q: np.ndarray,
    camera_offset: np.ndarray,
    min_clearance_m: float = TABLE_CLEARANCE_M,
) -> bool:
    """True if both the gripper and the camera clear the table at this pose.

    Joint limits alone do not exclude table strikes: a large part of the limit
    box drives the gripper straight into the table.
    """
    q = np.asarray(q, dtype=float)
    jaw_z = float(kin.fk(q, JAW_POINT)[2])
    cam_z = float(kin.fk(q, camera_offset)[2])
    return min(jaw_z, cam_z) >= min_clearance_m


def sample_poses(
    kin: ArmKinematics,
    n: int,
    seed: int,
    camera_offset: np.ndarray,
    joint_limits: np.ndarray | None = None,
    min_clearance_m: float = TABLE_CLEARANCE_M,
) -> np.ndarray:
    """Draw n poses from the sampling box that pass limits and clearance."""
    rng = np.random.default_rng(seed)
    low, high = POSE_LOW.copy(), POSE_HIGH.copy()
    if joint_limits is not None:
        low = np.maximum(low, joint_limits[:, 0])
        high = np.minimum(high, joint_limits[:, 1])
    if (low >= high).any():
        raise ValueError("joint limits leave no room in the sampling box")
    out: list[np.ndarray] = []
    tries = 0
    while len(out) < n and tries < 200 * n:
        tries += 1
        q = rng.uniform(low, high)
        if workspace_safe(kin, q, camera_offset, min_clearance_m):
            out.append(q)
    if len(out) < n:
        raise RuntimeError(f"only {len(out)} of {n} sampled poses cleared the table")
    return np.array(out)


# The uniform box above knows nothing about where the camera points, and on a
# real arm almost none of its poses show the board. The scan grid below points
# the camera down at the table in front of the base; the poses of it where the
# board is actually seen become the seeds that measure and correct jitter
# around. Per-joint jitter, radians: wide enough to vary the approach and the
# load, narrow enough that the board mostly stays in view.
# The defaults suit a board whose centre lies about 300 mm in front of the base.
SCAN_REACH_M = (0.28, 0.40)  # horizontal distance of the camera from the base axis
SCAN_HEIGHT_M = (0.20, 0.34)  # camera height above the table
SEED_JITTER = np.array([0.12, 0.12, 0.12, 0.12, 0.20])


def scan_grid(
    kin: ArmKinematics,
    camera_offset: np.ndarray,
    joint_limits: np.ndarray | None = None,
    min_clearance_m: float = TABLE_CLEARANCE_M,
    n: int = 30,
    reach_m: tuple[float, float] = SCAN_REACH_M,
    height_m: tuple[float, float] = SCAN_HEIGHT_M,
) -> np.ndarray:
    """Poses with the gripper pitched 45 to 65 degrees below horizontal and
    the camera over the table in front of the base, spread out in joint space.

    Pitch, elbow and wrist pitch turn about parallel axes, so the gripper's
    angle below horizontal is the sum of the three; fixing that sum is what
    keeps the camera looking at the table."""
    candidates = []
    for rot in np.linspace(-0.4, 0.4, 5):
        for q1 in np.linspace(-1.7, -0.2, 12):
            for q2 in np.linspace(0.2, 2.6, 14):
                for depression in (0.8, 0.95, 1.1):
                    for roll in (-0.4, 0.0, 0.4):
                        q = np.array([rot, q1, q2, depression - q1 - q2, roll])
                        if joint_limits is not None and (
                            (q < joint_limits[:, 0]).any() or (q > joint_limits[:, 1]).any()
                        ):
                            continue
                        if not workspace_safe(kin, q, camera_offset, min_clearance_m):
                            continue
                        x, y, z = kin.fk(q, camera_offset)
                        reach = float(np.hypot(x, y))
                        if not (reach_m[0] <= reach <= reach_m[1]):
                            continue
                        if not (height_m[0] <= z <= height_m[1]):
                            continue
                        candidates.append(q)
    if not candidates:
        raise RuntimeError("no scan pose fits inside the joint limits and the clearance")
    candidates = np.array(candidates)
    # Greedy max-min spread, so a short scan still covers the grid.
    picked = [0]
    nearest = np.linalg.norm(candidates - candidates[0], axis=1)
    while len(picked) < min(n, len(candidates)):
        best = int(np.argmax(nearest))
        if nearest[best] < 1e-9:
            break
        picked.append(best)
        nearest = np.minimum(nearest, np.linalg.norm(candidates - candidates[best], axis=1))
    return candidates[picked]


def jitter_poses(
    kin: ArmKinematics,
    seeds: np.ndarray,
    n: int,
    seed: int,
    camera_offset: np.ndarray,
    joint_limits: np.ndarray | None = None,
    min_clearance_m: float = TABLE_CLEARANCE_M,
    jitter: np.ndarray = SEED_JITTER,
) -> np.ndarray:
    """Draw n poses, each a seed pose chosen at random and moved by a uniform
    jitter per joint, that pass limits and clearance."""
    seeds = np.atleast_2d(np.asarray(seeds, dtype=float))
    if seeds.shape[1] != NUM_JOINTS or len(seeds) == 0:
        raise ValueError(f"expected seed poses of {NUM_JOINTS} joint angles, got shape {seeds.shape}")
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    tries = 0
    while len(out) < n and tries < 200 * n:
        tries += 1
        q = seeds[int(rng.integers(len(seeds)))] + rng.uniform(-jitter, jitter)
        if joint_limits is not None and (
            (q < joint_limits[:, 0]).any() or (q > joint_limits[:, 1]).any()
        ):
            continue
        if workspace_safe(kin, q, camera_offset, min_clearance_m):
            out.append(q)
    if len(out) < n:
        raise RuntimeError(f"only {len(out)} of {n} jittered poses cleared the table and the limits")
    return np.array(out)
