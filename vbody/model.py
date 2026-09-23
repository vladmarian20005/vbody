"""The body model: the analytic fit for one arm, and the residual on top of it.

Two questions, two expressions. The state estimate says where the arm is,
from the angles the servos report:

    p_state(q) = fk(q; theta)

The command prediction says where a command will put it:

    p(c, c_prev) = fk(c; theta) + r(c, c_prev)

theta is the nominal chain with a scale per link, a zero offset per joint,
the camera offset, and the board transform. r is a ridge regression on
physical features of the command: the angles, their sines and cosines, the
cumulative pitch angles that set each link's attitude to gravity, and the
sign of the last motion in each joint, which is what gives backlash a
handle. It absorbs what a rigid chain cannot express.

theta is fitted on the reported angles, because that is the question it
answers. r is fitted on the commanded angles, because that is its.

Four of the nineteen parameters cannot be told apart from others by any
measurement of the camera position, so they are held: the base joint's zero
offset is a yaw of the board transform, the first link's length is a
translation of it, the wrist link's length is the camera offset along the
roll axis, and the wrist roll zero is a rotation of the camera offset about
that axis. The last one is freed when the camera offset is held instead.
The fit reports how well each of the rest is determined by the poses it was
given.

What comes out is an operating condition for the poses it was fitted on, not
the geometry of the arm. On the arm this was built for, parameters that fit
one family of poses to two millimetres left more than a centimetre on a
deeply flexed family. Fit where you mean to work.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from vbody.kinematics import JOINT_NAMES, LINK_NAMES, NUM_JOINTS, NUM_LINKS, ArmKinematics
from vbody.registration import MIN_POSES_FOR_FIT, fit_registration, kabsch, pose_errors

MODEL_FORMAT = 1
RIDGE_ALPHA = 1e-4
N_FEATURES = 27

# Parameters held for exact redundancy with others; see the module docstring.
HELD_LINKS = {
    "Rotation_Pitch": "a translation of the board transform",
    "Fixed_Jaw": "the camera offset along the wrist roll axis",
}
HELD_JOINTS = {"Rotation": "a yaw of the board transform"}
# Held only while the camera offset is fitted alongside it.
HELD_JOINTS_WITH_CAMERA = {"Wrist_Roll": "a rotation of the camera offset about the roll axis"}
FREE_LINKS = [name for name in LINK_NAMES if name not in HELD_LINKS]
_FREE_LINK_IDX = [LINK_NAMES.index(name) for name in FREE_LINKS]


def held_joints(fit_camera_offset: bool) -> dict[str, str]:
    held = dict(HELD_JOINTS)
    if fit_camera_offset:
        held.update(HELD_JOINTS_WITH_CAMERA)
    return held


def free_joints(fit_camera_offset: bool) -> list[str]:
    held = held_joints(fit_camera_offset)
    return [name for name in JOINT_NAMES if name not in held]


# Fifteen parameters, three residuals per pose. Ten poses is the floor at
# which the fit is determined with some margin; it is not a good fit.
MIN_POSES_FOR_MODEL = 10
THIN_FIT_POSES = 25

# Below this many millimetres of tool displacement per natural unit of a
# parameter (a milliradian, a millimetre, one percent) the data barely
# constrain it and its fitted value should not be read as a measurement.
WEAK_SENSITIVITY_MM = 0.05

_PARAM_UNITS = {"rotation": 1e-3, "translation": 1e-3, "scale": 1e-2, "zero": 1e-3, "camera": 1e-3}
_PARAM_UNIT_NAMES = {
    "rotation": "mrad", "translation": "mm", "scale": "percent", "zero": "mrad", "camera": "mm"
}


def features(q_cmd: np.ndarray, prev_cmd: np.ndarray | None) -> np.ndarray:
    """The 27 features of one command: 1, q, sin q, cos q, the sines and
    cosines of the three cumulative pitch angles, and the sign of the motion
    in each joint from the previous command (zero if there was none)."""
    q = np.asarray(q_cmd, dtype=float)
    out = np.empty(N_FEATURES)
    out[0] = 1.0
    out[1:6] = q
    out[6:11] = np.sin(q)
    out[11:16] = np.cos(q)
    c1 = q[1]
    c2 = c1 + q[2]
    c3 = c2 + q[3]
    out[16:19] = np.sin([c1, c2, c3])
    out[19:22] = np.cos([c1, c2, c3])
    if prev_cmd is None:
        out[22:27] = 0.0
    else:
        out[22:27] = np.sign(q - np.asarray(prev_cmd, dtype=float))
    return out


def feature_matrix(q_commands: np.ndarray, prev_commands: list) -> np.ndarray:
    q_commands = np.atleast_2d(np.asarray(q_commands, dtype=float))
    if len(prev_commands) != len(q_commands):
        raise ValueError("one previous command per command is required (None is allowed)")
    return np.array([features(q, prev) for q, prev in zip(q_commands, prev_commands)])


@dataclass
class BodyModel:
    """One arm: the fitted chain, the board transform, and the residual."""

    rotation: np.ndarray  # (3, 3), base frame into board frame
    translation: np.ndarray  # (3,) m
    link_scale: np.ndarray  # (NUM_LINKS,) per LINK_NAMES
    zero_offset: np.ndarray  # (NUM_JOINTS,) rad per JOINT_NAMES
    camera_offset: np.ndarray  # (3,) m, wrist frame
    residual_weights: np.ndarray | None = None  # (N_FEATURES, 3), base frame, m
    ridge_alpha: float = RIDGE_ALPHA
    info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rotation = np.asarray(self.rotation, dtype=float).reshape(3, 3)
        self.translation = np.asarray(self.translation, dtype=float).reshape(3)
        self.link_scale = np.asarray(self.link_scale, dtype=float).reshape(NUM_LINKS)
        self.zero_offset = np.asarray(self.zero_offset, dtype=float).reshape(NUM_JOINTS)
        self.camera_offset = np.asarray(self.camera_offset, dtype=float).reshape(3)
        if self.residual_weights is not None:
            self.residual_weights = np.asarray(self.residual_weights, dtype=float).reshape(
                N_FEATURES, 3
            )
        self._kin: ArmKinematics | None = None

    @property
    def kinematics(self) -> ArmKinematics:
        if self._kin is None:
            self._kin = ArmKinematics(self.link_scale, self.zero_offset)
        return self._kin

    @property
    def has_residual(self) -> bool:
        return self.residual_weights is not None

    # ---- frames -----------------------------------------------------------

    def to_board(self, points_base: np.ndarray) -> np.ndarray:
        return np.atleast_2d(points_base) @ self.rotation.T + self.translation

    def to_base(self, points_board: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(points_board) - self.translation) @ self.rotation

    @property
    def board_origin_base(self) -> np.ndarray:
        """Where the board's origin sits in the base frame, m."""
        return -self.rotation.T @ self.translation

    # ---- the two expressions ---------------------------------------------

    def state(self, qs: np.ndarray) -> np.ndarray:
        """Camera position in the board frame at joint angles qs, (n, 3) m.
        Evaluated at reported angles this is the state estimate."""
        return self.to_board(self.kinematics.fk_batch(qs, self.camera_offset))

    def residual(self, q_cmd: np.ndarray, prev_cmd: np.ndarray | None) -> np.ndarray:
        """The learned correction for one command, base frame, m."""
        if self.residual_weights is None:
            return np.zeros(3)
        return features(q_cmd, prev_cmd) @ self.residual_weights

    def residual_batch(self, q_commands: np.ndarray, prev_commands: list) -> np.ndarray:
        if self.residual_weights is None:
            return np.zeros((len(np.atleast_2d(q_commands)), 3))
        return feature_matrix(q_commands, prev_commands) @ self.residual_weights

    def predict(self, q_commands: np.ndarray, prev_commands: list) -> np.ndarray:
        """Where each command will put the camera, board frame, (n, 3) m."""
        analytic = self.kinematics.fk_batch(q_commands, self.camera_offset)
        return self.to_board(analytic + self.residual_batch(q_commands, prev_commands))

    # ---- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        residual = None
        if self.residual_weights is not None:
            residual = {
                "ridge_alpha": self.ridge_alpha,
                "n_features": N_FEATURES,
                "weights_m": self.residual_weights.tolist(),
            }
        return {
            "vbody_model": MODEL_FORMAT,
            "board": {
                "rotation": self.rotation.tolist(),
                "translation_m": self.translation.tolist(),
            },
            "link_scale": dict(zip(LINK_NAMES, self.link_scale.tolist())),
            "zero_offset_rad": dict(zip(JOINT_NAMES, self.zero_offset.tolist())),
            "camera_offset_mm": (self.camera_offset * 1000).tolist(),
            "residual": residual,
            "fit": self.info,
        }

    @classmethod
    def from_dict(cls, blob: dict) -> BodyModel:
        if blob.get("vbody_model") != MODEL_FORMAT:
            raise ValueError(
                f"not a vbody model file (vbody_model = {blob.get('vbody_model')!r})"
            )
        residual = blob.get("residual")
        return cls(
            rotation=np.array(blob["board"]["rotation"]),
            translation=np.array(blob["board"]["translation_m"]),
            link_scale=np.array([blob["link_scale"][name] for name in LINK_NAMES]),
            zero_offset=np.array([blob["zero_offset_rad"][name] for name in JOINT_NAMES]),
            camera_offset=np.array(blob["camera_offset_mm"]) / 1000.0,
            residual_weights=None if residual is None else np.array(residual["weights_m"]),
            ridge_alpha=RIDGE_ALPHA if residual is None else float(residual["ridge_alpha"]),
            info=dict(blob.get("fit", {})),
        )

    def save(self, path: str | Path, **extra) -> None:
        blob = self.to_dict()
        blob["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        blob.update(extra)
        Path(path).write_text(json.dumps(blob, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> BodyModel:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"model file not found: {path}")
        return cls.from_dict(json.loads(path.read_text()))


# ---- the nominal belief ----------------------------------------------------

# The fitted chain and the nominal one do not share a base frame. The fit can
# trade a link scale against a translation of the board transform, and on a
# real arm that moves its base frame by centimetres. A point the nominal model
# believes in is therefore only meaningful in the board frame, placed there by
# the nominal chain's own registration, which the model file carries.


def _registration_dict(rotation: np.ndarray, translation: np.ndarray) -> dict:
    return {"rotation": np.asarray(rotation).tolist(), "translation_m": np.asarray(translation).tolist()}


def nominal_belief(model: BodyModel, q: np.ndarray) -> np.ndarray:
    """Where the nominal chain, with the camera offset the user gave, puts the
    camera at each pose, in the board frame, (n, 3) m."""
    reg = model.info.get("nominal_registration")
    given = model.info.get("camera_offset_given_mm")
    if reg is None or given is None:
        raise ValueError(
            "this model file does not carry the nominal registration; fit it again with "
            "this version of vbody"
        )
    rotation = np.array(reg["rotation"], dtype=float)
    translation = np.array(reg["translation_m"], dtype=float)
    points = ArmKinematics().fk_batch(q, np.array(given, dtype=float) / 1000.0)
    return points @ rotation.T + translation


# ---- the analytic fit ------------------------------------------------------


def _param_names(fit_camera_offset: bool) -> list[tuple[str, str]]:
    """(label, kind) for each free parameter, in vector order."""
    names = [(f"board rotation {a}", "rotation") for a in "xyz"]
    names += [(f"board translation {a}", "translation") for a in "xyz"]
    names += [(f"{name} scale", "scale") for name in FREE_LINKS]
    names += [(f"{name} zero", "zero") for name in free_joints(fit_camera_offset)]
    if fit_camera_offset:
        names += [(f"camera offset {a}", "camera") for a in "xyz"]
    return names


def _unpack(
    x: np.ndarray, camera_offset: np.ndarray, fit_camera_offset: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotation = Rotation.from_rotvec(x[0:3]).as_matrix()
    translation = x[3:6]
    link_scale = np.ones(NUM_LINKS)
    link_scale[_FREE_LINK_IDX] = x[6 : 6 + len(FREE_LINKS)]
    zero_offset = np.zeros(NUM_JOINTS)
    joints = [JOINT_NAMES.index(name) for name in free_joints(fit_camera_offset)]
    start = 6 + len(FREE_LINKS)
    zero_offset[joints] = x[start : start + len(joints)]
    start += len(joints)
    camera = x[start : start + 3] if fit_camera_offset else camera_offset
    return rotation, translation, link_scale, zero_offset, camera


def fit_body(
    q_reported: np.ndarray,
    positions_measured: np.ndarray,
    camera_offset: np.ndarray,
    fit_camera_offset: bool = True,
) -> BodyModel:
    """Fit the chain parameters and the board transform on reported angles.

    Starts from the six-parameter registration of the nominal model, the
    same one ``vbody measure`` reports, and then frees the link scales, the
    zero offsets and, unless told otherwise, the camera offset.
    """
    q_reported = np.atleast_2d(np.asarray(q_reported, dtype=float))
    positions_measured = np.atleast_2d(np.asarray(positions_measured, dtype=float))
    camera_offset = np.asarray(camera_offset, dtype=float)
    n = len(q_reported)
    if n < MIN_POSES_FOR_MODEL:
        raise ValueError(
            f"need at least {MIN_POSES_FOR_MODEL} measured poses to fit the body model, have {n}"
        )
    if len(positions_measured) != n:
        raise ValueError("one measurement per pose is required")

    reg = fit_registration(ArmKinematics(), q_reported, positions_measured, camera_offset)
    x0 = np.concatenate(
        [
            Rotation.from_matrix(reg.rotation).as_rotvec(),
            reg.translation,
            np.ones(len(FREE_LINKS)),
            np.zeros(len(free_joints(fit_camera_offset))),
            camera_offset if fit_camera_offset else [],
        ]
    )

    def residuals(x: np.ndarray) -> np.ndarray:
        rotation, translation, link_scale, zero_offset, camera = _unpack(
            x, camera_offset, fit_camera_offset
        )
        predicted = ArmKinematics(link_scale, zero_offset).fk_batch(q_reported, camera)
        return (predicted @ rotation.T + translation - positions_measured).ravel()

    result = least_squares(residuals, x0, method="lm", xtol=1e-12, ftol=1e-12)
    rotation, translation, link_scale, zero_offset, camera = _unpack(
        result.x, camera_offset, fit_camera_offset
    )
    per_pose = np.linalg.norm(result.fun.reshape(n, 3), axis=1) * 1000

    # How much the tool moves, RMS over the fitted poses, per natural unit of
    # each parameter. Near zero means the poses do not constrain it.
    sensitivity = {}
    weak = []
    for column, (label, kind) in zip(result.jac.T, _param_names(fit_camera_offset)):
        mm_per_unit = float(np.sqrt((column**2).mean()) * 1000 * _PARAM_UNITS[kind])
        sensitivity[label] = {"mm_per": _PARAM_UNIT_NAMES[kind], "value": mm_per_unit}
        if mm_per_unit < WEAK_SENSITIVITY_MM:
            weak.append(label)

    info = {
        "n_poses": int(n),
        "rms_mm": float(np.sqrt((per_pose**2).mean())),
        "max_mm": float(per_pose.max()),
        "converged": bool(result.success),
        "n_parameters": int(len(result.x)),
        "camera_offset_fitted": bool(fit_camera_offset),
        "camera_offset_given_mm": (camera_offset * 1000).tolist(),
        "nominal_registration_rms_mm": reg.rms_residual_mm,
        "nominal_registration": _registration_dict(reg.rotation, reg.translation),
        "held": {**{f"{k} scale": v for k, v in HELD_LINKS.items()},
                 **{f"{k} zero": v for k, v in held_joints(fit_camera_offset).items()}},
        "sensitivity": sensitivity,
        "weakly_determined": weak,
        "thin": bool(n < THIN_FIT_POSES),
    }
    return BodyModel(rotation, translation, link_scale, zero_offset, camera, info=info)


# ---- the residual ------------------------------------------------------------


def fit_residual(
    model: BodyModel,
    q_commanded: np.ndarray,
    positions_measured: np.ndarray,
    prev_commands: list,
    alpha: float = RIDGE_ALPHA,
) -> BodyModel:
    """Ridge-fit the residual on (command, measurement) pairs and return the
    model carrying it. The target is what the camera saw minus what the
    fitted chain predicts for the command, in the base frame."""
    q_commanded = np.atleast_2d(np.asarray(q_commanded, dtype=float))
    positions_measured = np.atleast_2d(np.asarray(positions_measured, dtype=float))
    phi = feature_matrix(q_commanded, prev_commands)
    analytic = model.kinematics.fk_batch(q_commanded, model.camera_offset)
    target = model.to_base(positions_measured) - analytic
    weights = np.linalg.solve(phi.T @ phi + alpha * np.eye(N_FEATURES), phi.T @ target)

    # A direction column that never changes sign is a constant, and the
    # intercept already has one: that joint's backlash cannot be learned.
    signs = phi[:, 22:27]
    uninformative = [
        JOINT_NAMES[j] for j in range(NUM_JOINTS)
        if not ((signs[:, j] > 0).any() and (signs[:, j] < 0).any())
    ]
    info = dict(model.info)
    info["residual"] = {
        "n_commands": int(len(q_commanded)),
        "ridge_alpha": float(alpha),
        "direction_uninformative": uninformative,
        "no_previous_commands": all(prev is None for prev in prev_commands),
    }
    return replace(model, residual_weights=weights, ridge_alpha=alpha, info=info)


def fit_model(
    q_commanded: np.ndarray,
    q_reported: np.ndarray,
    positions_measured: np.ndarray,
    camera_offset: np.ndarray,
    prev_commands: list,
    fit_camera_offset: bool = True,
    alpha: float = RIDGE_ALPHA,
) -> BodyModel:
    """The whole fit: chain and board on reported angles, residual on commands."""
    model = fit_body(q_reported, positions_measured, camera_offset, fit_camera_offset)
    return fit_residual(model, q_commanded, positions_measured, prev_commands, alpha)


# ---- evaluation ------------------------------------------------------------------


def evaluate(
    model: BodyModel,
    q_commanded: np.ndarray,
    q_reported: np.ndarray,
    positions_measured: np.ndarray,
    prev_commands: list,
    camera_offset_nominal: np.ndarray | None = None,
) -> dict:
    """Per-pose errors in mm for the five ways of predicting the camera.

    The nominal columns register the nominal chain on these poses with the
    board transform and the camera offset free, nine parameters, so that the
    comparison with the fitted model isolates the chain parameters and the
    residual rather than the user's tape measure. ``vbody measure`` holds the
    offset by default, so its reported column can be larger than this one.
    The nominal columns are NaN when there are too few poses.
    """
    q_commanded = np.atleast_2d(np.asarray(q_commanded, dtype=float))
    q_reported = np.atleast_2d(np.asarray(q_reported, dtype=float))
    measured = np.atleast_2d(np.asarray(positions_measured, dtype=float))
    n = len(q_commanded)
    nan = np.full(n, np.nan)
    nominal_cmd, nominal_rep = nan, nan
    if camera_offset_nominal is not None and n >= MIN_POSES_FOR_FIT:
        nominal = ArmKinematics()
        reg = fit_registration(
            nominal, q_reported, measured, camera_offset_nominal, fit_offset=True
        )
        nominal_cmd, nominal_rep = pose_errors(nominal, q_commanded, q_reported, measured, reg)
    return {
        "nominal_reported": nominal_rep,
        "state_reported": np.linalg.norm(model.state(q_reported) - measured, axis=1) * 1000,
        "nominal_commanded": nominal_cmd,
        "analytic_commanded": np.linalg.norm(model.state(q_commanded) - measured, axis=1) * 1000,
        "full_commanded": (
            np.linalg.norm(model.predict(q_commanded, prev_commands) - measured, axis=1) * 1000
        ),
    }


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt((values**2).mean())) if values.size else float("nan")


def cross_validate(
    q_commanded: np.ndarray,
    q_reported: np.ndarray,
    positions_measured: np.ndarray,
    camera_offset: np.ndarray,
    prev_commands: list,
    folds: int = 5,
    shuffles: int = 10,
    seed: int = 0,
    fit_camera_offset: bool = True,
    alpha: float = RIDGE_ALPHA,
) -> dict:
    """k-fold cross-validation of the whole fit, repeated over shuffles.

    Every fold refits the chain, the board and the residual on the other
    folds and scores the held-out poses, so nothing the fit saw is scored.
    Reports the RMS over held-out poses, averaged over shuffles, with the
    range across shuffles.
    """
    q_commanded = np.atleast_2d(np.asarray(q_commanded, dtype=float))
    q_reported = np.atleast_2d(np.asarray(q_reported, dtype=float))
    measured = np.atleast_2d(np.asarray(positions_measured, dtype=float))
    n = len(q_commanded)
    if folds < 2:
        raise ValueError("cross-validation needs at least two folds")
    if n - (n + folds - 1) // folds < MIN_POSES_FOR_MODEL:
        raise ValueError(
            f"{n} poses in {folds} folds leave fewer than {MIN_POSES_FOR_MODEL} to fit on"
        )
    rng = np.random.default_rng(seed)
    per_shuffle = {"state_reported": [], "analytic_commanded": [], "full_commanded": []}
    for _ in range(shuffles):
        held = {key: [] for key in per_shuffle}
        for part in np.array_split(rng.permutation(n), folds):
            train = np.setdiff1d(np.arange(n), part)
            model = fit_model(
                q_commanded[train], q_reported[train], measured[train], camera_offset,
                [prev_commands[i] for i in train], fit_camera_offset, alpha,
            )
            errors = evaluate(
                model, q_commanded[part], q_reported[part], measured[part],
                [prev_commands[i] for i in part],
            )
            for key in held:
                held[key].extend(errors[key])
        for key in per_shuffle:
            per_shuffle[key].append(rms(np.array(held[key])))
    out = {"folds": int(folds), "shuffles": int(shuffles), "seed": int(seed), "n_poses": int(n)}
    for key, values in per_shuffle.items():
        out[key] = {
            "rms_mm": float(np.mean(values)),
            "range_mm": [float(np.min(values)), float(np.max(values))],
        }
    return out


def hold_and_register(
    model: BodyModel, q_reported: np.ndarray, positions_measured: np.ndarray
) -> BodyModel:
    """Keep the arm's parameters and refit only where the board is.

    This is how a model is evaluated on poses measured in another session:
    the board may have been taped down again, and the arm has not changed
    in any way the model can express. The residual is carried over.
    """
    q_reported = np.atleast_2d(np.asarray(q_reported, dtype=float))
    measured = np.atleast_2d(np.asarray(positions_measured, dtype=float))
    n = len(q_reported)
    if n < MIN_POSES_FOR_FIT:
        raise ValueError(
            f"need at least {MIN_POSES_FOR_FIT} measured poses to place the board, have {n}"
        )
    predicted = model.kinematics.fk_batch(q_reported, model.camera_offset)
    r0, t0 = kabsch(predicted, measured)
    x0 = np.concatenate([Rotation.from_matrix(r0).as_rotvec(), t0])

    def residuals(x: np.ndarray) -> np.ndarray:
        rotation = Rotation.from_rotvec(x[:3]).as_matrix()
        return (predicted @ rotation.T + x[3:6] - measured).ravel()

    result = least_squares(residuals, x0, method="lm", xtol=1e-12, ftol=1e-12)
    rotation = Rotation.from_rotvec(result.x[:3]).as_matrix()
    per_pose = np.linalg.norm(result.fun.reshape(n, 3), axis=1) * 1000
    info = dict(model.info)
    given = info.get("camera_offset_given_mm")
    if given is not None:
        nominal = fit_registration(
            ArmKinematics(), q_reported, measured, np.array(given, dtype=float) / 1000.0
        )
        info["nominal_registration"] = _registration_dict(nominal.rotation, nominal.translation)
    info["board_refit"] = {
        "n_poses": int(n),
        "rms_mm": rms(per_pose),
        "converged": bool(result.success),
    }
    return replace(model, rotation=rotation, translation=result.x[3:6].copy(), info=info)


# ---- correction ------------------------------------------------------------------

MAX_RESIDUAL_M = 0.05


def correct_command(
    model: BodyModel,
    desired_base: np.ndarray,
    q_seed: np.ndarray,
    prev_command: np.ndarray | None,
    joint_limits: np.ndarray | None = None,
    sweeps: int = 2,
    ik_iterations: int = 6,
    max_residual_m: float = MAX_RESIDUAL_M,
    tol_m: float = 3e-4,
) -> tuple[np.ndarray, dict]:
    """The command whose predicted position lands on a desired base-frame point.

    A short fixed-point solve: each sweep predicts the residual at the
    current command, subtracts it from the desired point, and solves the
    fitted chain for that shifted point from the current command. The
    tolerance is 0.3 mm, about the residual's own accuracy. A predicted
    residual beyond ``max_residual_m`` is clamped to that length as a guard
    against a corrupt model, and the info says so. Nothing here checks the
    joint limits or the table clearance of the answer; the caller does.
    """
    desired = np.asarray(desired_base, dtype=float)
    command = np.array(q_seed, dtype=float)
    target = desired
    clamped = False
    residual = np.zeros(3)
    for _ in range(sweeps):
        residual = model.residual(command, prev_command)
        norm = float(np.linalg.norm(residual))
        if norm > max_residual_m:
            residual = residual * (max_residual_m / norm)
            clamped = True
        target = desired - residual
        command = model.kinematics.ik(
            target, command, model.camera_offset, ik_iterations, tol_m=tol_m,
            joint_limits=joint_limits,
        )
    miss = float(np.linalg.norm(target - model.kinematics.fk(command, model.camera_offset)))
    return command, {
        "residual_mm": float(np.linalg.norm(residual) * 1000),
        "clamped": clamped,
        "ik_miss_mm": miss * 1000,
        "correction_deg": np.rad2deg(command - np.asarray(q_seed, dtype=float)).tolist(),
    }
