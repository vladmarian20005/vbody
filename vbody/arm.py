"""Arm backends: the real SO-ARM101 over the Feetech bus, and a simulated arm.

The real driver speaks to Feetech STS3215 serial-bus servos through the
``scservo_sdk`` package, imported only when a real arm is actually opened.
Four rules hold the hardware path together. Units: the bus speaks ticks, 4096
per revolution, and this interface speaks radians, base to wrist, zeroed and
signed by a calibration file. Speed: acceleration and goal speed are capped on
every servo at connect time, before torque can exist. Stop: the stop works on a
half-built instance, and a failed connect disables whatever answered. Limits:
every command is checked before it is sent.

Constructing a ``RealArm`` never energizes the arm. Motion needs an explicit
``engage()``.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from vbody.kinematics import JOINT_NAMES, NUM_JOINTS, NUM_LINKS, ArmKinematics
from vbody.vision import Reading

# STS3215 control table (two-byte values, little-endian).
_ADDR_TORQUE_ENABLE = 40
_ADDR_ACCELERATION = 41  # unit: 100 ticks/s^2
_ADDR_GOAL_POSITION = 42
_ADDR_GOAL_SPEED = 46  # unit: ticks/s
_ADDR_PRESENT_POSITION = 56
_ADDR_PRESENT_SPEED = 58  # signed, bit 15

_TICKS_PER_REV = 4096
_RAD_PER_TICK = 2.0 * np.pi / _TICKS_PER_REV

STANDSTILL_RAD_S = 0.005


class Arm(ABC):
    """What the measurement loop needs from an arm."""

    joint_names: list[str] = JOINT_NAMES
    joint_limits: np.ndarray
    home_pose: np.ndarray

    def check_limits(self, q: np.ndarray) -> None:
        """Raise before anything moves if a command is out of range."""
        q = np.asarray(q, dtype=float)
        if q.shape != (NUM_JOINTS,):
            raise ValueError(f"expected {NUM_JOINTS} joint angles, got shape {q.shape}")
        lo, hi = self.joint_limits[:, 0], self.joint_limits[:, 1]
        # A NaN passes both comparisons below and must never reach a servo.
        bad = ~np.isfinite(q) | (q < lo) | (q > hi)
        if bad.any():
            names = [self.joint_names[i] for i in np.where(bad)[0]]
            raise ValueError(f"command outside the joint limits for {names}: {q[bad]}")

    @abstractmethod
    def engage(self) -> None:
        """Enable torque. The arm holds where it is."""

    @abstractmethod
    def command_joints(self, q: np.ndarray) -> None:
        """Command joint angles and block until the arm has settled."""

    @abstractmethod
    def read_positions(self) -> np.ndarray:
        """The angles the servos report, in radians."""

    @abstractmethod
    def emergency_stop(self) -> None:
        """Cut torque immediately. The arm will sag."""

    def home(self) -> None:
        """Move to the folded rest pose."""
        self.command_joints(self.home_pose)

    def close(self) -> None:
        """Release the hardware."""


@dataclass
class ArmConfig:
    """Servo wiring, calibration and software limits."""

    port: str = "/dev/ttyUSB0"
    baudrate: int = 1_000_000
    servo_ids: tuple[int, ...] = (1, 2, 3, 4, 5)  # base to wrist roll
    jaw_id: int = 6  # not driven, but torque-disabled on a stop
    zero_ticks: tuple[int, ...] = (2048, 2048, 2048, 2048, 2048)
    joint_signs: tuple[int, ...] = (1, 1, 1, 1, 1)
    joint_limits: np.ndarray = field(
        default_factory=lambda: np.array(
            [[-1.8, 1.8], [-3.2, 0.1], [-0.1, 3.1], [-1.6, 1.6], [-2.7, 2.7]]
        )
    )
    # Folded low-gravity rest pose. Not the model zero: at zero the arm is
    # extended horizontally, the worst pose to hold or park in.
    home_pose: tuple[float, ...] = (0.0, -2.95, 2.75, 1.30, 0.0)
    settle_timeout_s: float = 3.0
    # Feetech servos drive to a distant goal at full speed unless the
    # goal-speed register says otherwise.
    max_joint_speed_rad_s: float = 0.5
    max_joint_accel_rad_s2: float = 2.0

    @classmethod
    def from_calibration(cls, path: str | Path, **overrides) -> ArmConfig:
        """Read zero ticks, joint signs, limits and home pose from a file."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"calibration file not found: {path}. vBody will not move an arm "
                "without one: without zero ticks and joint signs, a commanded "
                "angle means nothing."
            )
        blob = json.loads(path.read_text())
        missing = [k for k in ("zero_ticks", "joint_signs") if k not in blob]
        if missing:
            raise ValueError(f"calibration file {path} is missing {missing}")
        if "joint_limits" in blob and "joint_limits" not in overrides:
            overrides["joint_limits"] = np.array(blob["joint_limits"], dtype=float)
        if "home_pose" in blob and "home_pose" not in overrides:
            overrides["home_pose"] = tuple(float(v) for v in blob["home_pose"])
        return cls(
            zero_ticks=tuple(int(t) for t in blob["zero_ticks"]),
            joint_signs=tuple(int(s) for s in blob["joint_signs"]),
            **overrides,
        )


class RealArm(Arm):
    """SO-ARM101 on the Feetech serial bus."""

    _COMM_RETRIES = 3

    def __init__(self, config: ArmConfig):
        self.config = config
        self.joint_limits = np.asarray(config.joint_limits, dtype=float)
        self.home_pose = np.array(config.home_pose, dtype=float)
        self._port = None
        self._packet = None
        self._engaged = False
        self._connect()

    # ---- bus primitives -------------------------------------------------
    # The half-duplex bus drops the occasional status packet. A lost reply is
    # retried; a servo error flag is a real answer and is never retried.

    def _txrx(self, op, what: str):
        from scservo_sdk import COMM_SUCCESS

        last = None
        for attempt in range(self._COMM_RETRIES):
            *val, res, err = op()
            if res == COMM_SUCCESS:
                if err != 0:
                    raise RuntimeError(f"{what}: servo error ({self._packet.getRxPacketError(err)})")
                return val[0] if val else None
            last = res
            if attempt < self._COMM_RETRIES - 1:
                time.sleep(0.01)
        raise RuntimeError(
            f"{what}: comm failure after {self._COMM_RETRIES} tries "
            f"({self._packet.getTxRxResult(last)})"
        )

    def _read2(self, sid: int, addr: int, what: str) -> int:
        from scservo_sdk import SCS_TOHOST

        val = self._txrx(lambda: self._packet.read2ByteTxRx(self._port, sid, addr), what)
        return SCS_TOHOST(val, 15)

    def _write1(self, sid: int, addr: int, value: int, what: str) -> None:
        self._txrx(lambda: self._packet.write1ByteTxRx(self._port, sid, addr, value), what)

    def _write2(self, sid: int, addr: int, value: int, what: str) -> None:
        self._txrx(lambda: self._packet.write2ByteTxRx(self._port, sid, addr, value), what)

    # ---- connection -----------------------------------------------------

    def _connect(self) -> None:
        try:
            from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler
        except ImportError as e:
            raise ImportError(
                "the real arm needs the Feetech SDK: pip install 'vbody[real]'. "
                "Use --dry-run to exercise the tool without hardware."
            ) from e

        self._port = PortHandler(self.config.port)
        self._packet = PacketHandler(0)  # STS series: protocol end 0
        if not self._port.openPort():
            self._port = None
            raise RuntimeError(f"cannot open {self.config.port}")
        try:
            self._port.setBaudRate(self.config.baudrate)
            for sid in self.config.servo_ids:
                _model, res, _err = self._packet.ping(self._port, sid)
                if res != COMM_SUCCESS:
                    raise RuntimeError(
                        f"servo id {sid} did not answer "
                        f"({self._packet.getTxRxResult(res)}); check power and the chain"
                    )
            speed_ticks = max(1, int(self.config.max_joint_speed_rad_s / _RAD_PER_TICK))
            accel_100 = max(1, int(self.config.max_joint_accel_rad_s2 / _RAD_PER_TICK / 100))
            for sid in self.config.servo_ids:
                self._write1(sid, _ADDR_TORQUE_ENABLE, 0, f"torque off id {sid}")
                self._write2(sid, _ADDR_GOAL_SPEED, speed_ticks, f"speed cap id {sid}")
                self._write1(sid, _ADDR_ACCELERATION, accel_100, f"accel cap id {sid}")
        except Exception:
            self.emergency_stop()  # leave nothing energized behind a failed setup
            self._port.closePort()
            self._port = None
            raise

    # ---- units ----------------------------------------------------------

    def _rad_to_ticks(self, q: np.ndarray) -> list[int]:
        ticks = np.rint(
            np.array(self.config.zero_ticks) + np.array(self.config.joint_signs) * q / _RAD_PER_TICK
        ).astype(int)
        if (ticks < 0).any() or (ticks >= _TICKS_PER_REV).any():
            raise ValueError(f"command maps outside the tick range 0..4095: {ticks}")
        return [int(t) for t in ticks]

    def _ticks_to_rad(self, ticks: np.ndarray) -> np.ndarray:
        offset = np.asarray(ticks, dtype=float) - np.array(self.config.zero_ticks)
        return np.array(self.config.joint_signs) * offset * _RAD_PER_TICK

    # ---- interface ------------------------------------------------------

    def engage(self) -> None:
        """Torque on. The goal register is synced to the present position
        first, so a stale goal left in memory cannot cause a jump."""
        for sid in self.config.servo_ids:
            present = self._read2(sid, _ADDR_PRESENT_POSITION, f"position id {sid}")
            self._write2(sid, _ADDR_GOAL_POSITION, present, f"goal sync id {sid}")
            self._write1(sid, _ADDR_TORQUE_ENABLE, 1, f"torque on id {sid}")
        self._engaged = True

    def command_joints(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float)
        self.check_limits(q)
        if not self._engaged:
            raise RuntimeError("the arm is not engaged: call engage() first, torque is off")
        before = self.read_positions()
        for sid, tick in zip(self.config.servo_ids, self._rad_to_ticks(q)):
            self._write2(sid, _ADDR_GOAL_POSITION, tick, f"goal id {sid}")
        self._wait_standstill(float(np.abs(q - before).max()))

    def _wait_standstill(self, travel_rad: float) -> None:
        """Block until reported speed is under the standstill threshold for six
        polls in a row, and never before 0.3 s has passed: a servo reads zero
        speed in the gap between the goal write and the start of motion."""
        threshold_ticks = max(STANDSTILL_RAD_S / _RAD_PER_TICK, 3)
        t0 = time.monotonic()
        deadline = t0 + max(
            self.config.settle_timeout_s, travel_rad / self.config.max_joint_speed_rad_s + 2.0
        )
        quiet = 0
        while time.monotonic() < deadline:
            speeds = [
                abs(self._read2(s, _ADDR_PRESENT_SPEED, f"speed id {s}"))
                for s in self.config.servo_ids
            ]
            if time.monotonic() - t0 >= 0.3 and max(speeds) <= threshold_ticks:
                quiet += 1
                if quiet >= 6:
                    return
            else:
                quiet = 0
            time.sleep(0.02)

    def read_positions(self) -> np.ndarray:
        ticks = [
            self._read2(s, _ADDR_PRESENT_POSITION, f"position id {s}")
            for s in self.config.servo_ids
        ]
        return self._ticks_to_rad(np.array(ticks))

    def emergency_stop(self) -> None:
        """Torque off every servo, best effort, never raising."""
        self._engaged = False
        if self._port is None or self._packet is None:
            return
        for sid in tuple(self.config.servo_ids) + (self.config.jaw_id,):
            try:
                self._packet.write1ByteTxRx(self._port, sid, _ADDR_TORQUE_ENABLE, 0)
            except Exception:  # noqa: BLE001, S112 - a stop must not die half way
                continue

    def close(self) -> None:
        self.emergency_stop()
        if self._port is not None:
            self._port.closePort()
            self._port = None


# ---- simulated arm ------------------------------------------------------

# Where the board lies, in the simulated setup: flat on the table in front of
# the base, its centre 300 mm forward, turned a little and not quite level.
_BOARD_CENTRE_BASE = np.array([0.02, -0.30, 0.0])
_BOARD_YAW_RAD = 0.12
_BOARD_TILT_RAD = 0.02
_BOARD_SIZE_M = np.array([0.21, 0.15])

# The simulated camera sees the board only from above it and not too far away.
_VISIBLE_MIN_HEIGHT_M = 0.05
_VISIBLE_MAX_RADIUS_M = 0.34
_VISIBLE_DROPOUT = 0.05


class SimArm(Arm):
    """A stand-in arm for dry runs and tests.

    It carries a hidden geometry error, hidden joint zero offsets, a hidden
    error in the camera offset, and hidden backlash and sag in the servo loop.
    The first three are invisible to the encoders and the last two are not,
    which is the point: commanded angles, reported angles and the camera
    measurement all disagree, the way they disagree on a real arm.
    """

    def __init__(self, camera_offset: np.ndarray, seed: int = 0):
        config = ArmConfig()
        self.config = config
        self.joint_limits = np.asarray(config.joint_limits, dtype=float)
        self.home_pose = np.array(config.home_pose, dtype=float)
        self._engaged = False
        self._rng = np.random.default_rng(seed)

        # Hidden geometry: link lengths a couple of percent off nominal, and
        # servo zeros out by a degree or so.
        rng = np.random.default_rng(seed + 991)
        self._true_kin = ArmKinematics(
            link_scale=1.0 + rng.uniform(-0.015, 0.015, NUM_LINKS),
            joint_offset=np.deg2rad(rng.uniform(-1.2, 1.2, NUM_JOINTS)),
        )
        # Hidden error in the camera offset the user measured, a few mm.
        self._true_camera_offset = np.asarray(camera_offset, dtype=float) + rng.uniform(
            -0.003, 0.003, 3
        )
        # Hidden servo loop: backlash taken up on the way out of home, and
        # gravity sag on the three pitching joints.
        self._backlash = np.deg2rad(rng.uniform(0.3, 1.1, NUM_JOINTS))
        self._sag = np.zeros(NUM_JOINTS)
        self._sag[1:4] = np.deg2rad(rng.uniform(0.4, 1.4, 3))

        rot = Rotation.from_euler("ZX", [_BOARD_YAW_RAD, _BOARD_TILT_RAD]).as_matrix()
        origin = _BOARD_CENTRE_BASE - rot @ np.array([_BOARD_SIZE_M[0] / 2, _BOARD_SIZE_M[1] / 2, 0])
        self._board_rotation = rot.T  # base frame into board frame
        self._board_translation = -self._board_rotation @ origin
        self._true_q = self.home_pose.copy()

    def engage(self) -> None:
        self._engaged = True

    def command_joints(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float)
        self.check_limits(q)
        if not self._engaged:
            raise RuntimeError("the arm is not engaged: call engage() first, torque is off")
        # Backlash is taken up in the direction of travel; sag pulls the
        # pitching joints down by an amount that depends on the pose.
        direction = np.sign(q - self._true_q)
        self._true_q = q + direction * self._backlash - self._sag * np.cos(q)

    def read_positions(self) -> np.ndarray:
        """The encoders see the true shaft angle, give or take their noise."""
        return self._true_q + self._rng.normal(0.0, np.deg2rad(0.02), NUM_JOINTS)

    def measure_camera(self) -> np.ndarray | None:
        """Where the wrist camera would report itself, in the board frame."""
        position = self._true_kin.fk(self._true_q, self._true_camera_offset)
        position = self._board_rotation @ position + self._board_translation
        centre = np.array([_BOARD_SIZE_M[0] / 2, _BOARD_SIZE_M[1] / 2])
        if position[2] < _VISIBLE_MIN_HEIGHT_M:
            return None
        if np.linalg.norm(position[:2] - centre) > _VISIBLE_MAX_RADIUS_M:
            return None
        if self._rng.random() < _VISIBLE_DROPOUT:
            return None
        return position + self._rng.normal(0.0, 0.0004, 3)

    def emergency_stop(self) -> None:
        self._engaged = False


class SimCamera:
    """The wrist camera of a simulated arm, with the same surface as the real
    one so the measurement loop does not know the difference."""

    def __init__(self, arm: SimArm):
        self._arm = arm

    def measure(self) -> Reading | None:
        position = self._arm.measure_camera()
        if position is None:
            return None
        return Reading(position_m=position, n_corners=24, spread_mm=0.2)

    def close(self) -> None:
        pass
