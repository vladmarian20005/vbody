"""Tests for vBody. None of these need hardware."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from vbody.arm import ArmConfig, SimArm, SimCamera
from vbody.cli import main
from vbody.kinematics import (
    JOINT_NAMES,
    LINK_NAMES,
    SEED_JITTER,
    ArmKinematics,
    jitter_poses,
    sample_poses,
    scan_grid,
    workspace_safe,
)
from vbody.model import (
    FREE_LINKS,
    N_FEATURES,
    BodyModel,
    correct_command,
    cross_validate,
    evaluate,
    fit_body,
    fit_model,
    free_joints,
    hold_and_register,
    rms,
)
from vbody.registration import fit_registration, joint_sign_check, pose_errors

CAMERA_OFFSET = np.array([0.060, -0.053, -0.015])


# ---- kinematics -----------------------------------------------------------


def test_batch_forward_kinematics_matches_one_pose_at_a_time():
    kin = ArmKinematics(link_scale=1 + np.linspace(-0.02, 0.02, 5), joint_offset=np.deg2rad([0.5, -1, 0.7, 0.3, -0.4]))
    q = sample_poses(ArmKinematics(), 12, seed=1, camera_offset=CAMERA_OFFSET)
    single = np.array([kin.fk(qi, CAMERA_OFFSET) for qi in q])
    assert np.allclose(kin.fk_batch(q, CAMERA_OFFSET), single, atol=1e-12)


def test_jacobian_matches_finite_differences():
    kin = ArmKinematics(joint_offset=np.deg2rad([0.2, -0.3, 0.4, 0.1, -0.2]))
    for q in sample_poses(ArmKinematics(), 5, seed=2, camera_offset=CAMERA_OFFSET):
        p, jac = kin.fk_jac(q, CAMERA_OFFSET)
        eps = 1e-7
        numeric = np.array(
            [(kin.fk(q + eps * np.eye(5)[i], CAMERA_OFFSET) - p) / eps for i in range(5)]
        ).T
        assert np.allclose(jac, numeric, atol=1e-6)


def test_inverse_kinematics_reaches_a_reachable_target():
    kin = ArmKinematics()
    q = sample_poses(kin, 4, seed=3, camera_offset=CAMERA_OFFSET)
    target = kin.fk(q[0], CAMERA_OFFSET)
    solved = kin.ik(target, q[0] + 0.05, CAMERA_OFFSET)
    assert np.linalg.norm(kin.fk(solved, CAMERA_OFFSET) - target) < 1e-4


# ---- registration ---------------------------------------------------------


def test_registration_recovers_a_known_transform():
    """Synthetic measurements from a known board pose: the fit must find it."""
    kin = ArmKinematics()
    rotation = Rotation.from_rotvec([0.03, -0.02, 0.35]).as_matrix()
    translation = np.array([0.12, -0.31, 0.02])

    q = sample_poses(kin, 20, seed=3, camera_offset=CAMERA_OFFSET)
    measured = kin.fk_batch(q, CAMERA_OFFSET) @ rotation.T + translation

    reg = fit_registration(kin, q, measured, CAMERA_OFFSET)
    assert reg.converged
    assert reg.rms_residual_mm < 1e-3
    assert np.allclose(reg.rotation, rotation, atol=1e-6)
    assert np.allclose(reg.translation, translation, atol=1e-6)

    commanded, reported = pose_errors(kin, q, q, measured, reg)
    assert commanded.max() < 1e-3
    assert reported.max() < 1e-3


def test_fitting_the_offset_recovers_a_wrong_offset():
    """With --fit-offset, an offset the user got wrong comes back out."""
    kin = ArmKinematics()
    rotation = Rotation.from_rotvec([0.0, 0.0, 0.2]).as_matrix()
    translation = np.array([0.1, -0.3, 0.0])
    true_offset = CAMERA_OFFSET + np.array([0.004, -0.006, 0.002])

    q = sample_poses(kin, 20, seed=4, camera_offset=CAMERA_OFFSET)
    measured = kin.fk_batch(q, true_offset) @ rotation.T + translation

    reg = fit_registration(kin, q, measured, CAMERA_OFFSET, fit_offset=True)
    assert np.allclose(reg.camera_offset, true_offset, atol=1e-5)


# ---- the body model ---------------------------------------------------------


class SyntheticArm:
    """A known arm behind a known board, for fits with a right answer."""

    def __init__(self, roll_zero_deg: float = 0.0):
        self.link_scale = 1 + np.array([0.0, 0.012, -0.009, 0.006, 0.0])
        self.zero_offset = np.deg2rad([0.0, -0.8, 0.5, -0.3, roll_zero_deg])
        self.camera_offset = CAMERA_OFFSET + np.array([0.003, -0.004, 0.002])
        self.kin = ArmKinematics(self.link_scale, self.zero_offset)
        self.rotation = Rotation.from_rotvec([0.02, -0.01, 0.3]).as_matrix()
        self.translation = np.array([0.1, -0.3, 0.01])

    def measure(self, q: np.ndarray, noise_m: float = 0.0, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        points = self.kin.fk_batch(q, self.camera_offset) @ self.rotation.T + self.translation
        return points + rng.normal(0.0, noise_m, points.shape)


def test_fit_recovers_a_known_body():
    truth = SyntheticArm()
    q = sample_poses(ArmKinematics(), 40, seed=11, camera_offset=CAMERA_OFFSET)
    model = fit_body(q, truth.measure(q), CAMERA_OFFSET)
    assert model.info["converged"]
    assert model.info["rms_mm"] < 1e-3
    for name in FREE_LINKS:
        i = LINK_NAMES.index(name)
        assert abs(model.link_scale[i] - truth.link_scale[i]) < 1e-5
    for name in free_joints(fit_camera_offset=True):
        i = JOINT_NAMES.index(name)
        assert abs(np.rad2deg(model.zero_offset[i] - truth.zero_offset[i])) < 1e-3
    assert np.linalg.norm(model.camera_offset - truth.camera_offset) < 1e-6
    # The held parameters stay where they were put.
    assert model.zero_offset[JOINT_NAMES.index("Rotation")] == 0.0
    assert model.link_scale[LINK_NAMES.index("Rotation_Pitch")] == 1.0
    assert model.link_scale[LINK_NAMES.index("Fixed_Jaw")] == 1.0
    assert model.info["weakly_determined"] == []


def test_a_noisy_fit_still_predicts_poses_it_never_saw():
    """Camera noise scatters the least constrained parameters, the link
    scales most of all; what matters is that the predictions hold up."""
    truth = SyntheticArm()
    q = sample_poses(ArmKinematics(), 40, seed=11, camera_offset=CAMERA_OFFSET)
    model = fit_body(q, truth.measure(q, noise_m=1e-4), CAMERA_OFFSET)
    assert model.info["rms_mm"] < 0.3
    fresh = sample_poses(ArmKinematics(), 20, seed=12, camera_offset=CAMERA_OFFSET)
    assert rms(np.linalg.norm(model.state(fresh) - truth.measure(fresh), axis=1) * 1000) < 0.3


def test_holding_the_camera_offset_frees_the_roll_zero():
    truth = SyntheticArm(roll_zero_deg=1.5)
    q = sample_poses(ArmKinematics(), 40, seed=13, camera_offset=CAMERA_OFFSET)
    model = fit_body(q, truth.measure(q), truth.camera_offset, fit_camera_offset=False)
    assert "Wrist_Roll zero" not in model.info["held"]
    assert abs(np.rad2deg(model.zero_offset[4]) - 1.5) < 0.05
    assert np.array_equal(model.camera_offset, truth.camera_offset)


def test_model_round_trips_through_json(tmp_path):
    truth = SyntheticArm()
    q = sample_poses(ArmKinematics(), 20, seed=14, camera_offset=CAMERA_OFFSET)
    measured = truth.measure(q)
    model = fit_model(q, q, measured, CAMERA_OFFSET, [None] + list(q[:-1]))
    model.save(tmp_path / "model.json", source="x.json")
    loaded = BodyModel.load(tmp_path / "model.json")
    assert np.allclose(loaded.rotation, model.rotation)
    assert np.allclose(loaded.translation, model.translation)
    assert np.allclose(loaded.link_scale, model.link_scale)
    assert np.allclose(loaded.zero_offset, model.zero_offset)
    assert np.allclose(loaded.camera_offset, model.camera_offset)
    assert loaded.residual_weights.shape == (N_FEATURES, 3)
    assert np.allclose(loaded.residual_weights, model.residual_weights)
    assert loaded.info["residual"]["n_commands"] == 20
    with pytest.raises(ValueError, match="not a vbody model"):
        BodyModel.from_dict({"vbody_model": 99})


def test_hold_and_register_finds_a_moved_board():
    truth = SyntheticArm()
    q = sample_poses(ArmKinematics(), 40, seed=15, camera_offset=CAMERA_OFFSET)
    model = fit_body(q, truth.measure(q), CAMERA_OFFSET)
    # The board is taped down somewhere else; the arm is the same.
    truth.rotation = Rotation.from_rotvec([-0.01, 0.02, -0.4]).as_matrix()
    truth.translation = np.array([0.05, -0.25, 0.0])
    fresh = sample_poses(ArmKinematics(), 8, seed=16, camera_offset=CAMERA_OFFSET)
    held = hold_and_register(model, fresh, truth.measure(fresh))
    assert np.allclose(held.link_scale, model.link_scale)
    assert np.allclose(held.zero_offset, model.zero_offset)
    assert held.info["board_refit"]["rms_mm"] < 0.3
    assert np.allclose(held.translation, truth.translation, atol=1e-3)


def test_correct_command_lands_on_the_desired_point():
    truth = SyntheticArm()
    q = sample_poses(ArmKinematics(), 30, seed=17, camera_offset=CAMERA_OFFSET)
    model = fit_body(q, truth.measure(q), CAMERA_OFFSET)
    # A residual that pulls every command 6 mm off, the way sag would.
    weights = np.zeros((N_FEATURES, 3))
    weights[0] = [0.004, -0.003, 0.003]
    model = BodyModel(
        model.rotation, model.translation, model.link_scale, model.zero_offset,
        model.camera_offset, residual_weights=weights,
    )
    desired = model.kinematics.fk(q[0], model.camera_offset)
    command, info = correct_command(model, desired, q[0], None)
    predicted = model.kinematics.fk(command, model.camera_offset) + model.residual(command, None)
    assert np.linalg.norm(predicted - desired) < 5e-4
    assert info["ik_miss_mm"] < 0.5
    assert not info["clamped"]
    # A corrupt residual is clamped rather than chased.
    weights[0] = [0.5, 0.0, 0.0]
    _, info = correct_command(model, desired, q[0], None)
    assert info["clamped"]
    assert info["residual_mm"] <= 50.0 + 1e-9


def collect(arm: SimArm, camera: SimCamera, poses: np.ndarray, path: bool):
    """Drive the simulated arm the way measure does and return the arrays."""
    arm.engage()
    arm.home()
    previous = arm.home_pose.copy()
    q_cmd, q_rep, meas, prevs = [], [], [], []
    for i, q in enumerate(poses):
        if i > 0 and not path:
            arm.home()
            previous = arm.home_pose.copy()
        arm.command_joints(q)
        reported = arm.read_positions()
        reading = camera.measure()
        if reading is not None:
            q_cmd.append(q)
            q_rep.append(reported)
            meas.append(reading.position_m)
            prevs.append(previous.copy())
        previous = q.copy()
    return np.array(q_cmd), np.array(q_rep), np.array(meas), prevs


def test_the_residual_learns_the_simulated_servo_loop():
    """The simulated arm has backlash and sag the chain cannot express. The
    residual must pick up part of it, and the direction term must be live."""
    kin = ArmKinematics()
    arm = SimArm(CAMERA_OFFSET, seed=5)
    poses = sample_poses(kin, 60, seed=5, camera_offset=CAMERA_OFFSET, joint_limits=arm.joint_limits)
    q_cmd, q_rep, meas, prevs = collect(arm, SimCamera(arm), poses, path=True)
    model = fit_model(q_cmd, q_rep, meas, CAMERA_OFFSET, prevs)
    assert model.info["residual"]["direction_uninformative"] == []
    errors = evaluate(model, q_cmd, q_rep, meas, prevs, CAMERA_OFFSET)
    assert rms(errors["state_reported"]) < rms(errors["nominal_reported"])
    assert rms(errors["state_reported"]) < 1.5
    assert rms(errors["full_commanded"]) < 0.5 * rms(errors["analytic_commanded"])
    cv = cross_validate(q_cmd, q_rep, meas, CAMERA_OFFSET, prevs, folds=5, shuffles=1)
    assert cv["state_reported"]["rms_mm"] < 1.5
    assert cv["full_commanded"]["rms_mm"] < cv["analytic_commanded"]["rms_mm"]


# ---- the simulated arm and the command line -----------------------------------


def test_sampled_poses_clear_the_table_and_the_limits():
    kin = ArmKinematics()
    limits = ArmConfig().joint_limits
    poses = sample_poses(kin, 30, seed=5, camera_offset=CAMERA_OFFSET, joint_limits=limits)
    assert poses.shape == (30, 5)
    for q in poses:
        assert workspace_safe(kin, q, CAMERA_OFFSET)
        assert (q >= limits[:, 0]).all() and (q <= limits[:, 1]).all()


def test_commands_outside_the_limits_are_refused():
    arm = SimArm(CAMERA_OFFSET, seed=0)
    arm.engage()
    with pytest.raises(ValueError, match="joint limits"):
        arm.command_joints(np.array([0.0, 0.0, 0.0, 0.0, 9.0]))
    with pytest.raises(ValueError, match="joint limits"):
        arm.command_joints(np.array([0.0, np.nan, 0.0, 0.0, 0.0]))


def test_a_missing_calibration_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="calibration"):
        ArmConfig.from_calibration(tmp_path / "not_here.json")


def test_the_simulated_arm_reports_something_other_than_the_command():
    """Commanded, reported and camera positions must all disagree."""
    kin = ArmKinematics()
    arm = SimArm(CAMERA_OFFSET, seed=2)
    camera = SimCamera(arm)
    arm.engage()
    seen = 0
    for q in sample_poses(kin, 6, seed=8, camera_offset=CAMERA_OFFSET):
        arm.home()
        arm.command_joints(q)
        reported = arm.read_positions()
        assert np.abs(reported - q).max() > np.deg2rad(0.1)
        reading = camera.measure()
        if reading is None:
            continue
        seen += 1
        assert not np.allclose(reading.position_m, kin.fk(reported, CAMERA_OFFSET), atol=1e-4)
    assert seen > 0


def dry_run(out, poses: int, seed: int, *extra: str) -> dict:
    """Run the measure command against the simulated arm and read the results."""
    argv = ["measure", "--camera-offset", "60", "-53", "-15", "--dry-run"]
    argv += ["--poses", str(poses), "--seed", str(seed), "--out", str(out), *extra]
    assert main(argv) == 0
    return json.loads(out.read_text())


def test_dry_run_measure_writes_results(tmp_path, capsys):
    """The whole measure path runs and writes a usable results file."""
    results = dry_run(tmp_path / "results.json", 12, 1)
    assert set(results) >= {
        "created_utc", "dry_run", "seed", "path", "camera_offset_mm", "home_pose", "n_poses",
        "n_measured", "n_missing", "registration", "summary", "model_evaluation", "poses",
    }
    assert results["n_poses"] == 12
    assert results["path"] is False
    assert results["model_evaluation"] is None
    assert results["n_measured"] + results["n_missing"] == 12
    assert results["n_measured"] >= 4
    assert results["registration"]["converged"]
    assert results["registration"]["camera_offset_fitted"] is False
    assert results["registration"]["camera_offset_mm"] == [60.0, -53.0, -15.0]
    assert len(results["poses"]) == 12

    for record in results["poses"]:
        assert set(record) >= {
            "pose", "q_commanded", "q_previous", "q_reported", "measured",
            "commanded_error_mm", "reported_error_mm",
        }
        # Every pose was approached from home.
        assert record["q_previous"] == results["home_pose"]
        if record["measured"]:
            assert record["commanded_error_mm"] is not None
            assert record["position_mm"] is not None

    table = capsys.readouterr().out
    assert "commanded (mm)" in table
    assert "reported angles" in table


def test_path_mode_records_the_previous_pose(tmp_path):
    results = dry_run(tmp_path / "path.json", 6, 1, "--path")
    assert results["path"] is True
    records = results["poses"]
    assert records[0]["q_previous"] == results["home_pose"]
    for before, after in zip(records, records[1:]):
        assert after["q_previous"] == before["q_commanded"]


def test_reported_angles_beat_commanded_angles_on_the_simulated_arm(tmp_path):
    """The servos see part of the error. Using what they report must help."""
    summary = dry_run(tmp_path / "results.json", 15, 7)["summary"]
    assert summary["reported_angles"]["mean_mm"] < summary["commanded_angles"]["mean_mm"]
    assert summary["reported_angles"]["rms_mm"] < summary["commanded_angles"]["rms_mm"]


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """A path-mode dry run on the simulated arm, and the model fitted to it."""
    directory = tmp_path_factory.mktemp("fit")
    results = directory / "path.json"
    model = directory / "model.json"
    dry_run(results, 60, 5, "--path")
    assert main(["fit", str(results), "--out", str(model), "--shuffles", "1"]) == 0
    return results, model


def test_fit_writes_a_model_that_beats_the_nominal_one(fitted, capsys):
    results, model_path = fitted
    blob = json.loads(model_path.read_text())
    assert blob["vbody_model"] == 1
    assert blob["measure_seed"] == 5
    assert blob["measure_dry_run"] is True
    assert blob["source"] == str(results)
    assert set(blob["link_scale"]) == set(LINK_NAMES)
    assert set(blob["zero_offset_rad"]) == set(JOINT_NAMES)
    assert blob["residual"]["n_features"] == N_FEATURES
    fit = blob["fit"]
    assert fit["converged"]
    assert fit["n_parameters"] == 15
    assert fit["camera_offset_given_mm"] == [60.0, -53.0, -15.0]
    in_sample = fit["in_sample_rms_mm"]
    assert in_sample["state_reported"] < in_sample["nominal_reported"]
    assert in_sample["full_commanded"] < in_sample["analytic_commanded"]
    cv = fit["cross_validation"]
    assert cv["folds"] == 5 and cv["shuffles"] == 1
    assert cv["state_reported"]["rms_mm"] < in_sample["nominal_reported"]
    assert fit["residual"]["direction_uninformative"] == []


def test_fit_warns_when_every_pose_came_from_home(tmp_path, capsys):
    results = tmp_path / "home.json"
    dry_run(results, 30, 4)
    assert main(["fit", str(results), "--out", str(tmp_path / "m.json"), "--folds", "0"]) == 0
    out = capsys.readouterr().out
    assert "direction term has nothing to learn from" in out
    assert "Pitch" in out
    assert "cross-validation" not in out


def test_fit_refuses_too_few_poses(tmp_path, capsys):
    results = tmp_path / "few.json"
    dry_run(results, 6, 1)
    assert main(["fit", str(results), "--out", str(tmp_path / "m.json")]) == 1
    assert "at least" in capsys.readouterr().err


def test_measure_evaluates_a_held_model_on_the_same_arm(fitted, tmp_path):
    _, model_path = fitted
    results = dry_run(tmp_path / "fresh.json", 14, 21, "--model", str(model_path))
    ev = results["model_evaluation"]
    assert ev["model"] == str(model_path)
    assert ev["board_refit"]["n_poses"] == results["n_measured"]
    assert ev["rms_mm"]["state_reported"] < ev["rms_mm"]["nominal_reported"]
    assert ev["rms_mm"]["state_reported"] < 1.5


def test_fit_hold_places_the_board_and_writes_the_held_model(fitted, tmp_path, capsys):
    _, model_path = fitted
    fresh = tmp_path / "fresh.json"
    dry_run(fresh, 14, 22)
    held = tmp_path / "held.json"
    assert main(["fit", str(fresh), "--hold", str(model_path), "--out", str(held)]) == 0
    out = capsys.readouterr().out
    assert "arm parameters held from" in out
    blob = json.loads(held.read_text())
    assert blob["held_from"] == str(model_path)
    original = json.loads(model_path.read_text())
    assert blob["link_scale"] == original["link_scale"]
    assert blob["fit"]["board_refit"]["n_poses"] >= 4


def test_correct_improves_held_out_targets_on_the_simulated_arm(fitted, tmp_path, capsys):
    _, model_path = fitted
    out = tmp_path / "correction.json"
    argv = ["correct", "--model", str(model_path), "--dry-run", "--targets", "8", "--seed", "9", "--out", str(out)]
    assert main(argv) == 0
    results = json.loads(out.read_text())
    summary = results["summary"]
    assert summary["n_targets"] == 8
    assert summary["n_both_measured"] >= 5
    assert summary["corrected"]["mean_mm"] < summary["uncorrected"]["mean_mm"]
    assert summary["n_improved"] > summary["n_both_measured"] // 2
    for row in results["targets"]:
        assert set(row) >= {"target", "q_target", "desired_mm", "q_corrected", "correction", "uncorrected", "corrected"}
        if row["q_corrected"] is not None:
            assert row["correction"]["refused"] is None
    printed = capsys.readouterr().out
    assert "measured in both passes" in printed
    assert "improved" in printed


def test_correct_warns_when_targets_are_the_calibration_poses(fitted, tmp_path, capsys):
    _, model_path = fitted
    argv = ["correct", "--model", str(model_path), "--dry-run", "--targets", "3", "--seed", "5", "--out", str(tmp_path / "c.json")]
    assert main(argv) == 0
    assert "calibration poses" in capsys.readouterr().out


# ---- the joint-sign check -------------------------------------------------------


def test_sign_check_finds_a_flipped_joint():
    """A joint whose recorded sign runs against the model's axis cannot be
    registered away; flipping it back can. The check must name it, and only it."""
    kin = ArmKinematics()
    arm = SimArm(CAMERA_OFFSET, seed=6)
    poses = sample_poses(kin, 30, seed=6, camera_offset=CAMERA_OFFSET, joint_limits=arm.joint_limits)
    q_cmd, q_rep, meas, _ = collect(arm, SimCamera(arm), poses, path=False)
    clean = joint_sign_check(kin, q_rep, meas, CAMERA_OFFSET)
    assert clean["suspect"] == []
    wrong = q_rep.copy()
    wrong[:, 4] = -wrong[:, 4]
    flagged = joint_sign_check(kin, wrong, meas, CAMERA_OFFSET)
    assert flagged["suspect"] == ["Wrist_Roll"]
    assert flagged["flipped_rms_mm"]["Wrist_Roll"] < 0.2 * flagged["rms_mm"]
    with pytest.raises(ValueError, match="at least"):
        joint_sign_check(kin, q_rep[:5], meas[:5], CAMERA_OFFSET)


def test_measure_records_the_sign_check_and_fit_refuses_a_flipped_joint(tmp_path, capsys):
    results = tmp_path / "survey.json"
    blob = dry_run(results, 30, 6)
    assert blob["sign_check"]["suspect"] == []
    assert set(blob["sign_check"]["flipped_rms_mm"]) == set(JOINT_NAMES)
    assert "no flip explains the poses better" in capsys.readouterr().out
    # The same run with the roll sign inverted, as a bad calibration file would record it.
    for record in blob["poses"]:
        for key in ("q_commanded", "q_previous", "q_reported"):
            record[key][4] = -record[key][4]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(blob))
    assert main(["fit", str(bad), "--out", str(tmp_path / "m.json")]) == 1
    out, err = capsys.readouterr()
    assert "recorded sign of Wrist_Roll runs against the model" in out
    assert "refusing to fit" in err
    assert not (tmp_path / "m.json").exists()
    assert main(["fit", str(bad), "--out", str(tmp_path / "m.json"), "--ignore-sign-check", "--folds", "0"]) == 0
    assert (tmp_path / "m.json").exists()


# ---- board-visible sampling ---------------------------------------------------

# The offset of the real wrist camera, which puts the scan window where the
# board of the simulated setup lies.
SCAN_OFFSET = np.array([-0.080, -0.008, -0.013])


def test_scan_grid_points_the_camera_at_the_table_in_front_of_the_base():
    kin = ArmKinematics()
    limits = ArmConfig().joint_limits
    poses = scan_grid(kin, SCAN_OFFSET, limits, n=20)
    assert len(poses) == 20
    for q in poses:
        assert workspace_safe(kin, q, SCAN_OFFSET)
        assert (q >= limits[:, 0]).all() and (q <= limits[:, 1]).all()
        # Pitch, elbow and wrist pitch sum to the gripper's depression.
        assert 0.8 - 1e-9 <= q[1] + q[2] + q[3] <= 1.1 + 1e-9


def test_jittered_poses_stay_near_their_seeds_and_inside_the_limits():
    kin = ArmKinematics()
    limits = ArmConfig().joint_limits
    seeds = scan_grid(kin, SCAN_OFFSET, limits, n=5)
    poses = jitter_poses(kin, seeds, 30, seed=4, camera_offset=SCAN_OFFSET, joint_limits=limits)
    assert len(poses) == 30
    for q in poses:
        assert (np.abs(seeds - q) <= SEED_JITTER + 1e-12).all(axis=1).any()
        assert workspace_safe(kin, q, SCAN_OFFSET)


def test_scan_then_measure_around_it(tmp_path, capsys):
    scan = tmp_path / "scan.json"
    offset = [str(v) for v in SCAN_OFFSET * 1000]
    assert main(["scan", "--camera-offset", *offset, "--dry-run", "--poses", "12", "--out", str(scan)]) == 0
    blob = json.loads(scan.read_text())
    assert blob["kind"] == "scan" and blob["n_measured"] >= 6
    out = tmp_path / "m.json"
    assert main([
        "measure", "--camera-offset", *offset, "--dry-run", "--poses", "10",
        "--around", str(scan), "--out", str(out),
    ]) == 0
    assert "jittering around" in capsys.readouterr().out
    assert json.loads(out.read_text())["n_measured"] >= 8


def test_correct_aims_at_the_nominal_belief_in_the_board_frame(fitted, tmp_path):
    # The aim point is where the nominal chain, registered on the calibration
    # poses, believes the target puts the camera. It must not be read in the
    # fitted chain's base frame, which a real fit moves by centimetres.
    _, model_path = fitted
    out = tmp_path / "c.json"
    assert main([
        "correct", "--model", str(model_path), "--dry-run", "--targets", "6",
        "--seed", "31", "--out", str(out),
    ]) == 0
    rows = json.loads(out.read_text())["targets"]
    model = BodyModel.load(model_path)
    from vbody.model import nominal_belief

    for row in rows:
        belief = nominal_belief(model, np.array([row["q_target"]]))[0] * 1000
        assert np.allclose(row["desired_mm"], belief, atol=1e-6)
        if row["uncorrected"] and row["uncorrected"]["error_mm"] is not None:
            assert row["uncorrected"]["error_mm"] < 40
