"""Command line interface: vbody board, camera, scan, measure, fit, correct."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from vbody.arm import Arm, ArmConfig, RealArm, SimArm, SimCamera
from vbody.kinematics import (
    JOINT_NAMES,
    LINK_NAMES,
    TABLE_CLEARANCE_M,
    SCAN_HEIGHT_M,
    SCAN_REACH_M,
    SEED_JITTER,
    ArmKinematics,
    jitter_poses,
    sample_poses,
    scan_grid,
    workspace_safe,
)
from vbody.model import (
    FREE_LINKS,
    MIN_POSES_FOR_MODEL,
    RIDGE_ALPHA,
    THIN_FIT_POSES,
    BodyModel,
    correct_command,
    cross_validate,
    evaluate,
    fit_model,
    free_joints,
    hold_and_register,
    nominal_belief,
    rms,
)
from vbody.registration import (
    MIN_POSES_FOR_FIT,
    MIN_POSES_FOR_SIGN_CHECK,
    Registration,
    fit_registration,
    joint_sign_check,
    pose_errors,
    summarize,
)
from vbody.vision import (
    Intrinsics,
    WristCamera,
    calibrate_intrinsics,
    capture_views,
    make_board,
    render_board_image,
)

INTRINSICS_BAR_PX = 0.3

# A corrected command whose fitted chain still misses its shifted target by
# more than this is asking for a point the arm cannot reach; refuse it.
MAX_IK_MISS_MM = 20.0


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---- vbody board --------------------------------------------------------


def cmd_board(args: argparse.Namespace) -> int:
    """Write the board image to print."""
    import cv2

    board = make_board(args.squares_x, args.squares_y, args.square_mm / 1000, args.marker_mm / 1000)
    image = render_board_image(board)
    out = Path(args.out)
    cv2.imwrite(str(out), image)
    width_mm = args.squares_x * args.square_mm
    height_mm = args.squares_y * args.square_mm
    print(f"wrote {out} ({image.shape[1]}x{image.shape[0]} px)")
    print(f"print it at 100 percent scale, no fit to page: {width_mm:.0f} x {height_mm:.0f} mm")
    print(f"then measure one square with calipers. It must be {args.square_mm:.1f} mm.")
    print("tape the print flat to the table. A curled sheet is a bent ruler.")
    return 0


# ---- vbody camera -------------------------------------------------------


def cmd_camera(args: argparse.Namespace) -> int:
    """Fit camera intrinsics from board views."""
    import cv2

    board = make_board(args.squares_x, args.squares_y, args.square_mm / 1000, args.marker_mm / 1000)
    if args.images:
        suffixes = {".png", ".jpg", ".jpeg", ".bmp"}
        paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in suffixes)
        views = [v for v in (cv2.imread(str(p)) for p in paths) if v is not None]
        print(f"read {len(views)} images from {args.images}")
    else:
        views = capture_views(args.camera_index, board, n_views=args.views)
        if args.save_views:
            directory = Path(args.save_views)
            directory.mkdir(parents=True, exist_ok=True)
            for i, view in enumerate(views, 1):
                cv2.imwrite(str(directory / f"view_{i:02d}.png"), view)
            print(f"saved {len(views)} views to {directory}")

    if len(views) < 10:
        print(f"only {len(views)} views: too few for a trustworthy fit", file=sys.stderr)
        return 1

    print("fitting...")
    intrinsics = calibrate_intrinsics(views, board)
    intrinsics.save(args.out)
    k = intrinsics.camera_matrix
    print(f"views used   : {intrinsics.n_views}")
    print(f"fx fy cx cy  : {k[0, 0]:.1f} {k[1, 1]:.1f} {k[0, 2]:.1f} {k[1, 2]:.1f}")
    print(f"reprojection : {intrinsics.rms_reproj_px:.3f} px RMS")
    print(f"wrote {args.out}")
    if intrinsics.rms_reproj_px >= INTRINSICS_BAR_PX:
        print(
            f"that is above {INTRINSICS_BAR_PX} px. Check focus and lighting and capture "
            "again before measuring anything: pose accuracy follows from this fit."
        )
    return 0


# ---- shared by measure and correct ---------------------------------------


def _check_hardware_args(args: argparse.Namespace) -> str | None:
    """The message to refuse with, or None when the run may proceed."""
    if args.dry_run:
        return None
    if not args.calibration:
        return "--calibration is required unless --dry-run is given"
    if not args.intrinsics:
        return "--intrinsics is required unless --dry-run is given"
    if not Path(args.intrinsics).is_file():
        return f"intrinsics file not found: {args.intrinsics}"
    return None


def _build_arm(
    args: argparse.Namespace, camera_offset: np.ndarray, sim_seed: int
) -> tuple[Arm, WristCamera | SimCamera]:
    """Open the arm and its wrist camera, real or simulated."""
    if args.dry_run:
        arm = SimArm(camera_offset, seed=sim_seed)
        return arm, SimCamera(arm)
    config = ArmConfig.from_calibration(args.calibration, port=args.port)
    intrinsics = Intrinsics.load(args.intrinsics)
    board = make_board(args.squares_x, args.squares_y, args.square_mm / 1000, args.marker_mm / 1000)
    camera = WristCamera(args.camera_index, intrinsics, board)
    try:
        return RealArm(config), camera
    except Exception:
        camera.close()
        raise


def _sim_seed(args: argparse.Namespace, model_path: str | None) -> int:
    """A dry run's seed defines the simulated arm as well as the poses. When a
    model fitted to a dry run is in play, simulate the arm it was fitted to,
    whatever seed picks the poses."""
    if args.dry_run and model_path and Path(model_path).is_file():
        blob = json.loads(Path(model_path).read_text())
        if blob.get("measure_dry_run") and blob.get("measure_seed") is not None:
            seed = int(blob["measure_seed"])
            if seed != args.seed:
                print(f"dry run: simulating the arm the model was fitted to (seed {seed})")
            return seed
    return args.seed


def _load_seed_poses(path: str | Path) -> np.ndarray:
    """Board-visible poses to jitter around: the hits of a scan file, or the
    measured poses of a measure results file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"seed pose file not found: {path}")
    blob = json.loads(path.read_text())
    key = "q" if blob.get("kind") == "scan" else "q_commanded"
    seeds = [r[key] for r in blob.get("poses", []) if r.get("measured")]
    if not seeds:
        raise ValueError(f"{path} records no pose where the board was seen")
    return np.array(seeds, dtype=float)


def _draw_poses(
    args: argparse.Namespace,
    kin: ArmKinematics,
    n: int,
    camera_offset: np.ndarray,
    joint_limits: np.ndarray,
    clearance: float,
) -> np.ndarray:
    """Jittered around the poses given with --around, or from the uniform box."""
    if args.around:
        seeds = _load_seed_poses(args.around)
        print(f"jittering around {len(seeds)} board-visible poses from {args.around}")
        return jitter_poses(kin, seeds, n, args.seed, camera_offset, joint_limits, clearance)
    if not args.dry_run:
        print(
            "warning: poses from the uniform box do not aim the camera at the board, and on a "
            "real arm few of them see it. Run vbody scan and pass its file with --around."
        )
    return sample_poses(kin, n, args.seed, camera_offset, joint_limits, clearance)


def _visit(arm: Arm, camera, q: np.ndarray, label: str) -> dict:
    """Command one pose, read the servos, ask the camera. The record has the
    same shape whether or not the board was seen."""
    arm.command_joints(q)
    reported = arm.read_positions()
    reading = camera.measure()
    record = {
        "q_reported": [float(v) for v in reported],
        "measured": reading is not None,
        "position_mm": None,
        "n_corners": None,
        "spread_mm": None,
    }
    if reading is None:
        print(f"  {label}  missing   board not visible")
    else:
        record["position_mm"] = [float(v) for v in reading.position_m * 1000]
        record["n_corners"] = int(reading.n_corners)
        record["spread_mm"] = float(reading.spread_mm)
        print(
            f"  {label}  measured  {np.round(reading.position_m * 1000, 1).tolist()} mm, "
            f"{reading.n_corners} corners, spread {reading.spread_mm:.2f} mm"
        )
    return record


def _load_results(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"results file not found: {path}")
    results = json.loads(path.read_text())
    if "poses" not in results or "camera_offset_mm" not in results:
        raise ValueError(f"{path} is not a vbody measure results file")
    return results


def _arrays(records: list[dict], home_pose: list | None = None):
    """The measured poses of a results file as arrays: (records, commanded,
    reported, positions in m, previous commands)."""
    hits = [r for r in records if r["measured"] and r.get("position_mm") is not None]
    q_commanded = np.array([r["q_commanded"] for r in hits]).reshape(-1, len(JOINT_NAMES))
    q_reported = np.array([r["q_reported"] for r in hits]).reshape(-1, len(JOINT_NAMES))
    measured = np.array([r["position_mm"] for r in hits]).reshape(-1, 3) / 1000.0
    fallback = None if home_pose is None else np.array(home_pose, dtype=float)
    previous = [
        np.array(r["q_previous"], dtype=float) if r.get("q_previous") is not None else fallback
        for r in hits
    ]
    return hits, q_commanded, q_reported, measured, previous


def _sign_check(
    q_reported: np.ndarray, measured: np.ndarray, camera_offset: np.ndarray
) -> dict | None:
    """Run the joint-sign check when there are enough poses, print the
    verdict, and return it. None when there were too few poses."""
    if len(q_reported) < MIN_POSES_FOR_SIGN_CHECK:
        print(
            f"\njoint signs: not checked, fewer than {MIN_POSES_FOR_SIGN_CHECK} measured poses"
        )
        return None
    check = joint_sign_check(ArmKinematics(), q_reported, measured, camera_offset)
    print("\njoint signs: each joint's recorded sign flipped in turn, board and camera offset refitted")
    print(
        "  " + "  ".join(f"{name} {value:.1f}" for name, value in check["flipped_rms_mm"].items())
        + f"  mm RMS (as recorded {check['rms_mm']:.1f})"
    )
    if check["suspect"]:
        for name in check["suspect"]:
            ratio = check["rms_mm"] / max(check["flipped_rms_mm"][name], 1e-9)
            print(
                f"  the recorded sign of {name} runs against the model's axis: flipped, it "
                f"explains the poses {ratio:.0f} times better."
            )
        print(
            "  Fix joint_signs for that joint in the calibration file and measure again. "
            "Do not fit a model to this run: the fit would absorb the flip as a wrist folded "
            "back on itself and a link of negative length."
        )
    else:
        print("  no flip explains the poses better than the signs as recorded")
    return check


ERROR_COLUMNS = [
    ("nominal_reported", "nominal@rep"),
    ("state_reported", "fitted@rep"),
    ("nominal_commanded", "nominal@cmd"),
    ("analytic_commanded", "fitted@cmd"),
    ("full_commanded", "fitted+res@cmd"),
]


def _print_error_table(hits: list[dict], errors: dict) -> None:
    print()
    print(f"{'pose':>5}" + "".join(f"{label:>16}" for _, label in ERROR_COLUMNS))
    for i, rec in enumerate(hits):
        cells = []
        for key, _ in ERROR_COLUMNS:
            value = errors[key][i]
            cells.append(f"{'-':>16}" if np.isnan(value) else f"{value:>16.2f}")
        print(f"{rec['pose']:>5}" + "".join(cells))
    print(f"{'':>5}" + f"{'(mm)':>16}")


def _print_model_summary(errors: dict, cv: dict | None, column: str = "in sample") -> None:
    def cell(value: float) -> str:
        return f"{'-':>12}" if value is None or np.isnan(value) else f"{value:>12.2f}"

    def cv_cell(key: str) -> str:
        return cell(None if cv is None else cv[key]["rms_mm"])

    print()
    print(f"{'':38}{column:>12}{'cross-val':>12}")
    print("  state estimate, at reported angles")
    print(f"    {'nominal model':34}{cell(rms(errors['nominal_reported']))}{cell(None)}")
    print(f"    {'fitted model':34}{cell(rms(errors['state_reported']))}{cv_cell('state_reported')}  mm RMS")
    print("  command prediction, at commanded angles")
    print(f"    {'nominal model':34}{cell(rms(errors['nominal_commanded']))}{cell(None)}")
    print(f"    {'fitted model':34}{cell(rms(errors['analytic_commanded']))}{cv_cell('analytic_commanded')}")
    print(f"    {'fitted model + residual':34}{cell(rms(errors['full_commanded']))}{cv_cell('full_commanded')}  mm RMS")
    if cv is not None:
        s_lo, s_hi = cv["state_reported"]["range_mm"]
        f_lo, f_hi = cv["full_commanded"]["range_mm"]
        print(
            f"  cross-validation: {cv['folds']} folds, {cv['shuffles']} shuffles; held-out RMS "
            f"ranged {s_lo:.2f}-{s_hi:.2f} (state) and {f_lo:.2f}-{f_hi:.2f} (prediction) mm"
        )


def _print_parameters(model: BodyModel) -> None:
    info = model.info
    print()
    print("fitted parameters: an operating condition for these poses, not the geometry of the arm")
    origin = np.round(model.board_origin_base * 1000, 1).tolist()
    print(f"  {'board origin in the base frame':32}{origin} mm")
    scales = "   ".join(
        f"{name} {model.link_scale[LINK_NAMES.index(name)]:.4f}" for name in FREE_LINKS
    )
    print(f"  {'link scale':32}{scales}")
    zeros = "   ".join(
        f"{name} {np.rad2deg(model.zero_offset[JOINT_NAMES.index(name)]):+.2f} deg"
        for name in free_joints(info.get("camera_offset_fitted", True))
    )
    print(f"  {'zero offset':32}{zeros}")
    offset = np.round(model.camera_offset * 1000, 2).tolist()
    if info.get("camera_offset_fitted", True):
        given = info.get("camera_offset_given_mm")
        print(f"  {'camera offset':32}{offset} mm" + ("" if given is None else f" (you gave {given} mm)"))
    else:
        print(f"  {'camera offset':32}{offset} mm, held at the value you gave")
    held = "; ".join(f"{name}: {why}" for name, why in info.get("held", {}).items())
    print(f"  {'held':32}{held}")
    weak = info.get("weakly_determined", [])
    if weak:
        detail = ", ".join(
            f"{name} ({info['sensitivity'][name]['value']:.2f} mm per {info['sensitivity'][name]['mm_per']})"
            for name in weak
        )
        print(f"  {'weakly determined':32}{detail}")
        print("    the poses barely move the camera through these; their values are not measurements")
    if info.get("thin"):
        print(f"  a fit on fewer than {THIN_FIT_POSES} poses is thin; more poses, spread wider, make it worth more")
    residual = info.get("residual")
    if residual is not None:
        print(
            f"  {'residual':32}27 features, ridge {residual['ridge_alpha']:g}, "
            f"fitted on {residual['n_commands']} commands"
        )
        dead = residual.get("direction_uninformative", [])
        if residual.get("no_previous_commands"):
            print(
                "    the results file records no previous commands, so the direction term "
                "is switched off. Measure again with this version, and with --path."
            )
        elif dead:
            print(
                f"    the direction term has nothing to learn from for {', '.join(dead)}: "
                "every pose was approached from the same side. Measure with --path."
            )


def _evaluate_held(
    model: BodyModel,
    q_commanded: np.ndarray,
    q_reported: np.ndarray,
    measured: np.ndarray,
    previous: list,
    camera_offset: np.ndarray,
) -> tuple[BodyModel, dict]:
    """Place the board for a held model on these poses, then score it."""
    held = hold_and_register(model, q_reported, measured)
    return held, evaluate(held, q_commanded, q_reported, measured, previous, camera_offset)


# ---- vbody scan ---------------------------------------------------------


def cmd_scan(args: argparse.Namespace) -> int:
    """Visit a grid of poses looking down at the table and keep the ones
    from which the board is seen."""
    camera_offset = np.array(args.camera_offset, dtype=float) / 1000.0
    kin = ArmKinematics()
    refusal = _check_hardware_args(args)
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    arm, camera = _build_arm(args, camera_offset, args.seed)
    records: list[dict] = []
    try:
        poses = scan_grid(
            kin, camera_offset, arm.joint_limits, args.clearance / 1000, args.poses,
            tuple(r / 1000 for r in args.reach), tuple(h / 1000 for h in args.height),
        )
        for i, q in enumerate(poses, 1):
            arm.check_limits(q)
            if not workspace_safe(kin, q, camera_offset, args.clearance / 1000):
                raise ValueError(f"scan pose {i} leaves less than {args.clearance:.0f} mm of clearance")
        if not args.dry_run:
            input(
                f"Workspace clear? {len(poses)} scan poses, visited in sequence. "
                "Keep a hand near the power switch. Enter to start... "
            )
        arm.engage()
        arm.home()
        for i, q in enumerate(poses, 1):
            record = {"pose": i, "q": [float(v) for v in q]}
            record.update(_visit(arm, camera, q, f"scan {i:>3}/{len(poses)}"))
            records.append(record)
        arm.home()
        arm.emergency_stop()
    except KeyboardInterrupt:
        print("\n*** stop: torque off, partial results saved ***")
        arm.emergency_stop()
    finally:
        arm.close()
        camera.close()

    hits = sum(1 for r in records if r["measured"])
    print(f"\n{hits} of {len(records)} scan poses see the board")
    if hits < MIN_POSES_FOR_FIT:
        print(
            "too few to jitter around. Move the board so its centre sits about 300 mm in "
            "front of the base, or widen --reach and --height to where it lies."
        )
    results = {
        "kind": "scan",
        "created_utc": _utc_now(),
        "dry_run": bool(args.dry_run),
        "camera_offset_mm": [float(v) for v in args.camera_offset],
        "n_poses": len(records),
        "n_measured": hits,
        "poses": records,
    }
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")
    return 0 if hits >= MIN_POSES_FOR_FIT else 1


# ---- vbody measure ------------------------------------------------------


def _print_table(records: list[dict]) -> None:
    print()
    print(f"{'pose':>5}  {'commanded (mm)':>15}  {'reported (mm)':>14}  camera")
    for rec in records:
        if rec["measured"] and rec["commanded_error_mm"] is not None:
            print(
                f"{rec['pose']:>5}  {rec['commanded_error_mm']:>15.2f}  "
                f"{rec['reported_error_mm']:>14.2f}  measured"
            )
        else:
            state = "measured" if rec["measured"] else "missing"
            print(f"{rec['pose']:>5}  {'-':>15}  {'-':>14}  {state}")


def _print_summary(
    records: list[dict],
    commanded: dict,
    reported: dict,
    reg: Registration | None,
    camera_offset: np.ndarray,
) -> None:
    n_measured = sum(1 for r in records if r["measured"])
    print()
    print(
        f"{len(records)} poses commanded, {n_measured} measured, "
        f"{len(records) - n_measured} missing (board not visible)"
    )
    if reg is None:
        print("too few measurements to fit the board transform; no errors reported")
        return
    print()
    print(f"{'':22}{'mean':>8}{'RMS':>9}{'worst':>9}")
    for name, stats in (("commanded angles", commanded), ("reported angles", reported)):
        print(
            f"  {name:20}{stats['mean_mm']:>8.2f}"
            f"{stats['rms_mm']:>9.2f}{stats['max_mm']:>9.2f}  mm"
        )
    print(
        f"\nboard to base: {9 if reg.offset_was_fitted else 6} parameters fitted on "
        f"{reg.n_poses} poses, residual {reg.rms_residual_mm:.2f} mm RMS"
        + ("" if reg.converged else ", DID NOT CONVERGE")
    )
    origin = -reg.rotation.T @ reg.translation
    print(f"  board origin in the base frame: {np.round(origin * 1000, 1).tolist()} mm")
    offset_mm = np.round(reg.camera_offset * 1000, 2).tolist()
    if reg.offset_was_fitted:
        given = np.round(camera_offset * 1000, 2).tolist()
        print(f"  camera offset fitted: {offset_mm} mm (you gave {given} mm)")
    else:
        print(f"  camera offset held at the value you gave: {offset_mm} mm")


def cmd_measure(args: argparse.Namespace) -> int:
    """Drive the poses, measure, and report the two errors."""
    camera_offset = np.array(args.camera_offset, dtype=float) / 1000.0
    kin = ArmKinematics()

    refusal = _check_hardware_args(args)
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    model = BodyModel.load(args.model) if args.model else None

    arm, camera = _build_arm(args, camera_offset, _sim_seed(args, args.model))
    home = np.array(arm.home_pose, dtype=float)
    records: list[dict] = []
    try:
        poses = _draw_poses(
            args, kin, args.poses, camera_offset, arm.joint_limits, args.clearance / 1000
        )
        # Refuse to move at all unless every pose in the list is safe.
        for i, q in enumerate(poses, 1):
            arm.check_limits(q)
            if not workspace_safe(kin, q, camera_offset, args.clearance / 1000):
                raise ValueError(f"pose {i} leaves less than {args.clearance:.0f} mm of clearance")
        if not args.dry_run:
            route = (
                "visited in sequence, moving straight from each to the next"
                if args.path else "each approached from home"
            )
            input(
                f"Workspace clear? {len(poses)} poses, {route}. "
                "Keep a hand near the power switch. Enter to start... "
            )
        arm.engage()

        arm.home()
        previous = home.copy()
        for i, q in enumerate(poses, 1):
            if i > 1 and not args.path:
                arm.home()
                previous = home.copy()
            record = {
                "pose": i,
                "q_commanded": [float(v) for v in q],
                "q_previous": [float(v) for v in previous],
                **_visit(arm, camera, q, f"pose {i:>3}/{len(poses)}"),
                "commanded_error_mm": None,
                "reported_error_mm": None,
            }
            records.append(record)
            previous = np.array(q, dtype=float)
        arm.home()
        arm.emergency_stop()
    except KeyboardInterrupt:
        print("\n*** stop: torque off, partial results saved ***")
        arm.emergency_stop()
    finally:
        arm.close()
        camera.close()

    hits, q_commanded, q_reported, measured, previous_cmds = _arrays(records)

    reg: Registration | None = None
    commanded_stats = summarize(np.array([]))
    reported_stats = summarize(np.array([]))
    if len(hits) >= MIN_POSES_FOR_FIT:
        reg = fit_registration(
            kin, q_reported, measured, camera_offset, fit_offset=args.fit_offset
        )
        commanded_err, reported_err = pose_errors(kin, q_commanded, q_reported, measured, reg)
        commanded_stats = summarize(commanded_err)
        reported_stats = summarize(reported_err)
        for rec, ec, er in zip(hits, commanded_err, reported_err):
            rec["commanded_error_mm"] = float(ec)
            rec["reported_error_mm"] = float(er)

    _print_table(records)
    _print_summary(records, commanded_stats, reported_stats, reg, camera_offset)
    sign_check = _sign_check(q_reported, measured, camera_offset) if reg is not None else None

    model_evaluation = None
    if model is not None:
        if len(hits) < MIN_POSES_FOR_FIT:
            print(f"\ntoo few measurements to place the board for {args.model}; model not evaluated")
        else:
            held, errors = _evaluate_held(
                model, q_commanded, q_reported, measured, previous_cmds, camera_offset
            )
            refit = held.info["board_refit"]
            print(
                f"\nagainst {args.model}, arm parameters held, board placed on "
                f"{refit['n_poses']} poses ({refit['rms_mm']:.2f} mm RMS):"
            )
            _print_model_summary(errors, None, column="this run")
            model_evaluation = {
                "model": str(args.model),
                "board_refit": refit,
                "rms_mm": {key: rms(errors[key]) for key, _ in ERROR_COLUMNS},
                "per_pose_mm": {
                    key: [None if np.isnan(v) else float(v) for v in errors[key]]
                    for key, _ in ERROR_COLUMNS
                },
            }

    results = {
        "created_utc": _utc_now(),
        "dry_run": bool(args.dry_run),
        "seed": int(args.seed),
        "path": bool(args.path),
        "camera_offset_mm": [float(v) for v in args.camera_offset],
        "home_pose": [float(v) for v in home],
        "n_poses": len(records),
        "n_measured": sum(1 for r in records if r["measured"]),
        "n_missing": sum(1 for r in records if not r["measured"]),
        "registration": reg.to_dict() if reg else None,
        "summary": {"commanded_angles": commanded_stats, "reported_angles": reported_stats},
        "sign_check": sign_check,
        "model_evaluation": model_evaluation,
        "poses": records,
    }
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    return 0


# ---- vbody fit ----------------------------------------------------------


def cmd_fit(args: argparse.Namespace) -> int:
    """Fit the body model to a results file, or evaluate a held one on it."""
    results = _load_results(args.results)
    camera_offset = np.array(results["camera_offset_mm"], dtype=float) / 1000.0
    hits, q_commanded, q_reported, measured, previous = _arrays(
        results["poses"], results.get("home_pose")
    )
    route = "visited in sequence" if results.get("path") else "each approached from home"
    print(f"{len(hits)} measured poses in {args.results}, {route}")
    if not args.ignore_sign_check and len(hits) >= MIN_POSES_FOR_SIGN_CHECK:
        check = _sign_check(q_reported, measured, camera_offset)
        if check and check["suspect"]:
            print(
                "\nrefusing to fit. Pass --ignore-sign-check to fit anyway, if you are sure.",
                file=sys.stderr,
            )
            return 1

    if args.hold:
        model = BodyModel.load(args.hold)
        if len(hits) < MIN_POSES_FOR_FIT:
            raise ValueError(
                f"need at least {MIN_POSES_FOR_FIT} measured poses to place the board, have {len(hits)}"
            )
        held, errors = _evaluate_held(
            model, q_commanded, q_reported, measured, previous, camera_offset
        )
        refit = held.info["board_refit"]
        print(
            f"arm parameters held from {args.hold}; board placed on {refit['n_poses']} poses, "
            f"{refit['rms_mm']:.2f} mm RMS"
            + ("" if refit["converged"] else ", DID NOT CONVERGE")
        )
        _print_error_table(hits, errors)
        _print_model_summary(errors, None, column="this file")
        if args.out:
            held.info["evaluation"] = {
                "source": str(args.results),
                "rms_mm": {key: rms(errors[key]) for key, _ in ERROR_COLUMNS},
            }
            held.save(args.out, source=str(args.results), held_from=str(args.hold))
            print(f"\nwrote {args.out}")
        return 0

    if len(hits) < MIN_POSES_FOR_MODEL:
        raise ValueError(
            f"need at least {MIN_POSES_FOR_MODEL} measured poses to fit the body model, "
            f"have {len(hits)}"
        )
    model = fit_model(
        q_commanded, q_reported, measured, camera_offset, previous,
        fit_camera_offset=not args.hold_camera_offset, alpha=args.ridge,
    )
    errors = evaluate(model, q_commanded, q_reported, measured, previous, camera_offset)
    cv = None
    if args.folds > 0:
        try:
            cv = cross_validate(
                q_commanded, q_reported, measured, camera_offset, previous,
                folds=args.folds, shuffles=args.shuffles, seed=args.cv_seed,
                fit_camera_offset=not args.hold_camera_offset, alpha=args.ridge,
            )
        except ValueError as e:
            print(f"cross-validation skipped: {e}")

    _print_error_table(hits, errors)
    _print_model_summary(errors, cv)
    _print_parameters(model)
    if not model.info["converged"]:
        print("\nthe fit DID NOT CONVERGE; do not use this model")

    model.info["in_sample_rms_mm"] = {key: rms(errors[key]) for key, _ in ERROR_COLUMNS}
    model.info["cross_validation"] = cv
    model.save(
        args.out,
        source=str(args.results),
        measure_seed=results.get("seed"),
        measure_dry_run=results.get("dry_run"),
        measure_path=results.get("path"),
    )
    print(f"\nwrote {args.out}")
    return 0


# ---- vbody correct ------------------------------------------------------


def cmd_correct(args: argparse.Namespace) -> int:
    """Approach held-out targets with and without the correction."""
    model_blob = json.loads(Path(args.model).read_text()) if Path(args.model).is_file() else {}
    model = BodyModel.load(args.model)
    if not model.has_residual:
        print("the model carries no residual; the correction uses the fitted chain alone")
    given = model.info.get("camera_offset_given_mm")
    if given is None:
        raise ValueError(f"{args.model} does not record the camera offset it was measured with")
    camera_offset = np.array(given, dtype=float) / 1000.0
    kin = ArmKinematics()

    refusal = _check_hardware_args(args)
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    measure_seed = model_blob.get("measure_seed")
    if measure_seed is not None and int(measure_seed) == args.seed:
        print(
            f"warning: --seed {args.seed} is the seed the model was measured with; "
            "the targets will be the calibration poses, not held-out ones"
        )
    arm, camera = _build_arm(args, camera_offset, _sim_seed(args, args.model))
    home = np.array(arm.home_pose, dtype=float)
    clearance = args.clearance / 1000
    outcomes: dict[str, list] = {"uncorrected": [], "corrected": []}
    try:
        targets = _draw_poses(args, kin, args.targets, camera_offset, arm.joint_limits, clearance)
        for i, q in enumerate(targets, 1):
            arm.check_limits(q)
            if not workspace_safe(kin, q, camera_offset, clearance):
                raise ValueError(f"target {i} leaves less than {args.clearance:.0f} mm of clearance")
        # The nominal belief of each target, placed in the board frame by the
        # nominal chain's registration, then in the fitted chain's base frame.
        desired = model.to_base(nominal_belief(model, targets))

        # Corrected commands, in the order they will run: each one's
        # predecessor is the last command actually sent.
        print(f"\n{len(targets)} targets; solving the corrected commands")
        previous = home.copy()
        corrected: list[np.ndarray | None] = []
        corrections: list[dict] = []
        for i, (q, point) in enumerate(zip(targets, desired), 1):
            command, info = correct_command(model, point, q, previous, arm.joint_limits)
            reason = None
            if info["ik_miss_mm"] > MAX_IK_MISS_MM:
                reason = (
                    f"unreachable, the fitted chain misses the shifted target by "
                    f"{info['ik_miss_mm']:.1f} mm"
                )
            else:
                try:
                    arm.check_limits(command)
                except ValueError as e:
                    reason = str(e)
                if reason is None and not workspace_safe(kin, command, camera_offset, clearance):
                    reason = f"corrected command leaves less than {args.clearance:.0f} mm of clearance"
            info["refused"] = reason
            corrections.append(info)
            if reason:
                corrected.append(None)
                print(f"  target {i:>3}: refused, {reason}")
            else:
                corrected.append(command)
                previous = command
                print(
                    f"  target {i:>3}: residual {info['residual_mm']:.1f} mm, "
                    f"correction {np.round(info['correction_deg'], 2).tolist()} deg"
                    + (", CLAMPED" if info["clamped"] else "")
                )

        if not args.dry_run:
            input(
                f"Workspace clear? Two passes of {len(targets)} targets, uncorrected then "
                "corrected, each pass from home and then straight from target to target. "
                "Keep a hand near the power switch. Enter to start... "
            )
        arm.engage()
        for name, commands in (("uncorrected", list(targets)), ("corrected", corrected)):
            print(f"\n{name} pass")
            arm.home()
            for i, command in enumerate(commands, 1):
                if command is None:
                    outcomes[name].append(None)
                    print(f"  target {i:>3}/{len(commands)}  skipped   refused above")
                    continue
                outcomes[name].append(_visit(arm, camera, command, f"target {i:>3}/{len(commands)}"))
        arm.home()
        arm.emergency_stop()
    except KeyboardInterrupt:
        print("\n*** stop: torque off, partial results saved ***")
        arm.emergency_stop()
    finally:
        arm.close()
        camera.close()

    n_run = len(outcomes["uncorrected"])
    targets = targets[:n_run]
    desired = desired[:n_run]
    corrected = corrected[:n_run]
    corrections = corrections[:n_run]
    outcomes["corrected"] += [None] * (n_run - len(outcomes["corrected"]))

    # Score in the model's board frame, or in one placed on this run.
    scored = model
    reregistered = None
    if args.reregister:
        seen = [(t, o) for t, o in zip(targets, outcomes["uncorrected"]) if o and o["measured"]]
        if len(seen) >= MIN_POSES_FOR_FIT:
            scored = hold_and_register(
                model,
                np.array([o["q_reported"] for _, o in seen]),
                np.array([o["position_mm"] for _, o in seen]) / 1000.0,
            )
            reregistered = scored.info["board_refit"]
            print(
                f"\nboard placed on the uncorrected pass: {reregistered['n_poses']} poses, "
                f"{reregistered['rms_mm']:.2f} mm RMS"
            )
        else:
            print("\ntoo few uncorrected readings to place the board again; using the model's placement")
    desired_board = nominal_belief(scored, targets)

    rows = []
    for i in range(n_run):
        row = {
            "target": i + 1,
            "q_target": [float(v) for v in targets[i]],
            "desired_mm": [float(v) for v in desired_board[i] * 1000],
            "q_corrected": None if corrected[i] is None else [float(v) for v in corrected[i]],
            "correction": corrections[i],
        }
        for name in ("uncorrected", "corrected"):
            outcome = outcomes[name][i]
            error = None
            if outcome and outcome["measured"]:
                error = float(
                    np.linalg.norm(np.array(outcome["position_mm"]) - desired_board[i] * 1000)
                )
            row[name] = None if outcome is None else {**outcome, "error_mm": error}
        rows.append(row)

    print()
    print(f"{'target':>6}  {'uncorrected (mm)':>17}  {'corrected (mm)':>15}")
    both = []
    for row in rows:
        u = row["uncorrected"]["error_mm"] if row["uncorrected"] else None
        c = row["corrected"]["error_mm"] if row["corrected"] else None
        note = ""
        if row["corrected"] is None:
            note = "  refused"
        elif c is None:
            note = "  corrected pass: board not visible"
        if u is None:
            note = "  uncorrected pass: board not visible" + note
        print(
            f"{row['target']:>6}  {('-' if u is None else f'{u:.2f}'):>17}  "
            f"{('-' if c is None else f'{c:.2f}'):>15}{note}"
        )
        if u is not None and c is not None:
            both.append((u, c))

    summary = {
        "n_targets": n_run,
        "n_refused": sum(1 for r in rows if r["corrected"] is None),
        "n_both_measured": len(both),
        "uncorrected": summarize(np.array([u for u, _ in both])),
        "corrected": summarize(np.array([c for _, c in both])),
        "n_improved": sum(1 for u, c in both if c < u),
    }
    print()
    if both:
        print(f"{len(both)} targets measured in both passes")
        print(f"{'':16}{'mean':>8}{'RMS':>9}{'worst':>9}")
        for name in ("uncorrected", "corrected"):
            s = summary[name]
            print(f"  {name:14}{s['mean_mm']:>8.2f}{s['rms_mm']:>9.2f}{s['max_mm']:>9.2f}  mm")
        print(f"  {summary['n_improved']} of {len(both)} improved")
    else:
        print("no target was measured in both passes; nothing to compare")
    missing = n_run - len(both)
    if missing:
        print(
            f"{missing} targets have no comparison: {summary['n_refused']} refused, "
            f"{missing - summary['n_refused']} with the board out of view in one pass"
        )

    results = {
        "created_utc": _utc_now(),
        "dry_run": bool(args.dry_run),
        "model": str(args.model),
        "seed": int(args.seed),
        "camera_offset_mm": [float(v) for v in given],
        "clearance_mm": float(args.clearance),
        "board_reregistered": reregistered,
        "summary": summary,
        "targets": rows,
    }
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    return 0


# ---- argument parsing ---------------------------------------------------


def _add_board_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--squares-x", type=int, default=7, help="board squares across")
    parser.add_argument("--squares-y", type=int, default=5, help="board squares down")
    parser.add_argument("--square-mm", type=float, default=30.0, help="printed square size")
    parser.add_argument("--marker-mm", type=float, default=22.0, help="printed marker size")


def _add_hardware_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--port", default="/dev/ttyUSB0", help="serial port of the servo bus")
    parser.add_argument("--calibration", help="servo calibration JSON, required on hardware")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--intrinsics", help="camera intrinsics JSON, required on hardware")
    parser.add_argument(
        "--clearance", type=float, default=TABLE_CLEARANCE_M * 1000,
        help="clearance every pose must leave above the table, mm",
    )
    parser.add_argument("--dry-run", action="store_true", help="use the simulated arm")
    _add_board_options(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vbody",
        description="Measure how far an SO-ARM101's forward kinematics is from "
        "what its wrist camera sees, fit a model of that arm, and correct targets with it.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    board = subparsers.add_parser("board", help="write the ChArUco board image to print")
    board.add_argument("--out", default="board.png", help="output PNG path")
    _add_board_options(board)
    board.set_defaults(func=cmd_board)

    camera = subparsers.add_parser("camera", help="fit camera intrinsics from board views")
    camera.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    camera.add_argument("--images", help="fit from this directory of images instead of live views")
    camera.add_argument("--views", type=int, default=35, help="live views to capture")
    camera.add_argument("--save-views", help="directory to write the captured views to")
    camera.add_argument("--out", default="intrinsics.json", help="output JSON path")
    _add_board_options(camera)
    camera.set_defaults(func=cmd_camera)

    scan = subparsers.add_parser(
        "scan", help="find the poses from which the wrist camera sees the board"
    )
    scan.add_argument(
        "--camera-offset", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"),
        help="camera optical centre in the wrist frame, mm",
    )
    scan.add_argument("--poses", type=int, default=30, help="scan poses to visit")
    scan.add_argument(
        "--reach", type=float, nargs=2, default=[r * 1000 for r in SCAN_REACH_M],
        metavar=("MIN", "MAX"), help="camera distance from the base axis to scan over, mm",
    )
    scan.add_argument(
        "--height", type=float, nargs=2, default=[h * 1000 for h in SCAN_HEIGHT_M],
        metavar=("MIN", "MAX"), help="camera height above the table to scan over, mm",
    )
    scan.add_argument("--seed", type=int, default=1, help="simulated arm seed for --dry-run")
    scan.add_argument("--out", default="scan.json", help="output JSON path")
    _add_hardware_options(scan)
    scan.set_defaults(func=cmd_scan)

    measure = subparsers.add_parser("measure", help="run the pose survey and report the errors")
    measure.add_argument(
        "--camera-offset", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"),
        help="camera optical centre in the wrist frame, mm",
    )
    measure.add_argument("--poses", type=int, default=25, help="poses to visit")
    measure.add_argument("--out", default="results.json", help="output JSON path")
    measure.add_argument(
        "--fit-offset", action="store_true",
        help="also fit the camera offset instead of holding it at the value given",
    )
    measure.add_argument(
        "--path", action="store_true",
        help="move straight from each pose to the next instead of returning home between "
        "them; this is what lets the fitted residual learn about approach direction",
    )
    measure.add_argument(
        "--model", help="a model from vbody fit to evaluate on this run, with its board placed anew"
    )
    measure.add_argument("--seed", type=int, default=1, help="pose sampling seed")
    measure.add_argument(
        "--around", metavar="FILE",
        help="draw poses by jittering around the board-visible poses of a scan or measure "
        f"file (per-joint jitter {np.round(SEED_JITTER, 2).tolist()} rad) instead of the "
        "uniform box, which does not aim the camera at the board",
    )
    _add_hardware_options(measure)
    measure.set_defaults(func=cmd_measure)

    fit = subparsers.add_parser(
        "fit", help="fit the body model and residual to a measure results file"
    )
    fit.add_argument("results", help="results JSON written by vbody measure")
    fit.add_argument("--out", default="model.json", help="output model JSON path")
    fit.add_argument(
        "--hold", metavar="MODEL",
        help="do not fit: hold this model's arm parameters, place its board on these poses, "
        "and report how it does",
    )
    fit.add_argument(
        "--hold-camera-offset", action="store_true",
        help="hold the camera offset at the value measure was given instead of fitting it",
    )
    fit.add_argument(
        "--ignore-sign-check", action="store_true",
        help="fit even when a joint's recorded sign appears to run against the model",
    )
    fit.add_argument("--ridge", type=float, default=RIDGE_ALPHA, help="ridge penalty of the residual")
    fit.add_argument("--folds", type=int, default=5, help="cross-validation folds, 0 to skip")
    fit.add_argument("--shuffles", type=int, default=10, help="cross-validation shuffles")
    fit.add_argument("--cv-seed", type=int, default=0, help="cross-validation shuffle seed")
    fit.set_defaults(func=cmd_fit)

    correct = subparsers.add_parser(
        "correct", help="approach held-out targets with and without the model's correction"
    )
    correct.add_argument("--model", required=True, help="model JSON written by vbody fit")
    correct.add_argument("--targets", type=int, default=10, help="targets to approach")
    correct.add_argument(
        "--seed", type=int, default=2,
        help="target sampling seed; use one the model was not measured with",
    )
    correct.add_argument(
        "--around", metavar="FILE",
        help="draw poses by jittering around the board-visible poses of a scan or measure "
        f"file (per-joint jitter {np.round(SEED_JITTER, 2).tolist()} rad) instead of the "
        "uniform box, which does not aim the camera at the board",
    )
    correct.add_argument("--out", default="correction.json", help="output JSON path")
    correct.add_argument(
        "--reregister", action="store_true",
        help="place the board anew on the uncorrected pass before scoring, for a board "
        "taped down again since the model was fitted",
    )
    _add_hardware_options(correct)
    correct.set_defaults(func=cmd_correct)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
