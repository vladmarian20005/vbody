# vBody

vBody measures how far an SO-ARM101 really is from where its own model says it
is, using nothing but the arm, a camera on its wrist, and a printed board. It
then fits a model of that one arm from the same measurements, and uses the
model to put the arm where it was asked to go.

A ChArUco board is taped to the table. The arm goes to a pose, the wrist camera
finds the board, and from the board's pose in the camera frame we get the
camera's optical centre in the board frame. That is the measurement. Against it
we put forward kinematics, evaluated twice: once at the angles that were
commanded, and once at the angles the servos reported once they had stopped
moving.

The gap between those two numbers is the point. The commanded-angle error
contains everything: servo tracking, backlash, sag, and the geometry the model
gets wrong. The reported-angle error contains only what the servos cannot see,
because everything they can see has already been folded into the angles they
report. An arm whose two columns are far apart is one you can improve by
reading its encoders. An arm whose reported column is still large has an error
its encoders will never show you.

## What you need

- An SO-ARM101 with Feetech STS3215 servos, on a USB serial bus.
- A camera mounted on the wrist link, rigidly.
- A ChArUco board, 7 by 5 squares of 30 mm, printed and taped flat to the
  table in front of the arm. `vbody board` writes the file to print.
- The offset of the camera's optical centre from the wrist link, in
  millimetres, measured in the wrist frame. See below.
- A servo calibration file: the encoder ticks at the reference pose, the
  per-joint signs, the joint limits, and the folded home pose.

## Installing

    pip install -e .          # measurement, fitting, simulation and tests
    pip install -e '.[real]'  # adds the Feetech serial bus SDK

The `real` extra pulls in `feetech-servo-sdk`, which installs the `scservo_sdk`
module. Without it everything except talking to a real arm still works.

## The six commands

`vbody board` writes the board image. Print it at 100 percent scale with page
scaling off, then measure one square with calipers. If it is not 30.0 mm, every
later number is wrong by the same proportion.

`vbody camera` fits the wrist camera's intrinsics. Put the board on the table
and move the camera around it by hand; the command captures a view each time it
sees the board from a genuinely new angle, and stops at 35. It can also fit from
a directory of images you captured yourself. Aim for a reprojection error under
0.3 px. Pose accuracy follows from this fit, so it is not a step to rush.

`vbody scan` finds where the camera can see the board. It visits a grid of
poses with the gripper pitched down at the table in front of the base and
keeps the ones where the board is seen. The grid covers camera positions 280
to 400 mm out from the base axis and 200 to 340 mm above the table, which
suits a board centred about 300 mm in front of the base; `--reach` and
`--height` move the window if your board lies elsewhere. On the arm this was
written for, 13 of 30 scan poses saw the board.

`vbody measure` is the measurement. Given a scan file with `--around`, it
draws poses by jittering around the scan's hits, up to 0.12 rad on the base
and pitch joints and 0.2 rad on the wrist roll, keeps those inside the joint
limits and clear of the table, visits each one,
reads the reported angles, and asks the camera where it is. Poses where the
board is not visible are recorded as missing and excluded from the fit.
Without `--around` it samples a box of joint angles that knows nothing about
where the camera points. That is fine for a dry run, but on a real arm almost
none of those poses see the board (0 of 33 on the arm this was written for),
and the command warns you. By
default every pose is approached from the home pose. With `--path` the arm
moves straight from each pose to the next, which is what a fit needs to learn
about approach direction; see below.

`vbody fit` takes the results of a measure run and fits the body model to
them: the chain the arm actually has, where the board is, and a residual for
what a rigid chain cannot express. It cross-validates the fit, prints what it
found, and writes a model file.

`vbody correct` is the test of that model. It samples fresh targets the model
has never seen, approaches each one twice, once with the plain command and
once with the command the model says will land there, and reports both
errors side by side.

## The camera offset

`--camera-offset X Y Z` is the camera's optical centre expressed in the wrist
link frame, in millimetres. The wrist link is the last body in the chain, the
one the wrist roll servo turns; its frame origin sits on the wrist roll axis.
The point midway between the jaw pads lies at (0, -90, 10) mm in that frame,
which gives you a second landmark to measure from.

You do not need this to be perfect. An error here shows up as a constant offset
in the wrist frame. `vbody measure --fit-offset` will fit it along with the
board transform and print what it found; use that once to learn the offset,
then go back to giving it fixed, because fitting it for every run lets the fit
absorb some of the arm's real error, which flatters the result. `vbody fit`
fits the offset as part of the body model by default, since there it is one
parameter among fifteen and the model is judged on held-out poses;
`--hold-camera-offset` keeps it at the value you gave.

## Check the joint signs

The calibration file says which way each servo counts. If one joint's sign
runs against the model's axis, the reported angles of that joint cannot be
explained by any placement of the board or the camera, and the whole
measurement is quietly wrong. `vbody measure` checks for this at the end of
every run of eight or more poses: it flips each joint's recorded sign in
turn, refits the board transform and the camera offset, and reports the
residual each flip leaves.

    joint signs: each joint's recorded sign flipped in turn, board and camera offset refitted
      Rotation 46.6  Pitch 76.8  Elbow 62.6  Wrist_Pitch 30.7  Wrist_Roll 4.1  mm RMS (as recorded 24.4)
      the recorded sign of Wrist_Roll runs against the model's axis: flipped, it explains the poses 6 times better.

That output is from a real arm whose calibration had the wrist roll sign
inverted. As recorded, reported angles disagreed with the camera by 24 mm;
with the sign fixed, by 4 mm. Nothing else in the run looked wrong. Fix
`joint_signs` for that joint in the calibration file and measure again.

`vbody fit` runs the same check first and refuses to fit when a flip wins,
because the fit would otherwise absorb the wrong sign as a wrist folded back
on itself and a link of negative length, and reach a good residual with
parameters that mean nothing. `--ignore-sign-check` overrides that if you
are sure.

## An example session

This is the session the paper's correction experiment was run with. The
hardware options are the same for every command that moves the arm, so they
are written once as `$HW`.

    HW="--port /dev/tty.usbmodem5A7A0590781 --calibration arm_calib.json \
        --camera-index 1 --intrinsics intrinsics.json"

    vbody board --out board.png
    vbody camera --camera-index 1 --out intrinsics.json
    vbody scan --camera-offset -80 -8 -13 $HW --out scan.json
    vbody measure --camera-offset -80 -8 -13 $HW --around scan.json \
        --poses 100 --path --seed 555 --out calib.json
    vbody fit calib.json --out model.json
    vbody correct --model model.json --around scan.json --targets 10 \
        --seed 20260923 $HW --out correction.json

The camera offset above is the one this arm's fit returned, give or take
2 mm; a camera mounted differently needs its own.

To see the whole thing run without hardware, add `--dry-run` to `measure` and
`correct`, which swaps in a simulated arm carrying a deliberate geometry error,
a wrong camera offset, backlash and sag:

    vbody measure --camera-offset -80 -8 -13 --dry-run --poses 60 --path --out sim.json
    vbody fit sim.json --out sim_model.json
    vbody correct --model sim_model.json --dry-run --targets 10 --seed 9 --out sim_correction.json

In a dry run the measure seed defines the simulated arm as well as the poses.
`fit` records it, and `correct` and `measure --model` simulate that same arm
whatever seed picks their poses, so a model is always tested on the arm it
was fitted to.

## Reading the measurement

    pose   commanded (mm)   reported (mm)  camera
       1             3.49            2.20  measured
       2            12.16            2.10  measured
      10                -               -  missing

    10 poses commanded, 9 measured, 1 missing (board not visible)

                              mean      RMS    worst
      commanded angles        6.49     6.89    12.16  mm
      reported angles         1.81     1.84     2.20  mm

    board to base: 6 parameters fitted on 9 poses, residual 1.84 mm RMS

Each row is the distance, in millimetres, between where the camera says it was
and where forward kinematics says it should have been. The board transform is
fitted on the reported angles, so the reported column is what is left after the
best possible rigid placement of the board, and the fit residual and the
reported RMS are the same quantity seen twice.

Everything printed is also written to the results JSON, per pose, including the
joint angles, the pose the arm came from, and the measured positions, so the
analysis can be redone without touching the arm again. That file is what
`vbody fit` reads.

## Reading the fit

The model answers two different questions, and the fit reports them
separately.

The state estimate says where the arm is now, from the angles the servos
report: forward kinematics through the fitted chain. The command prediction
says where a command will put the arm: the same chain evaluated at the
command, plus the residual. The first is the easier question, because the
encoders have already absorbed everything they can see. The second is the one
a controller has to answer.

    pose      nominal@rep      fitted@rep     nominal@cmd      fitted@cmd  fitted+res@cmd
       1             4.30            0.38            6.72            5.74            5.46
       2             2.29            0.32           12.56           11.90            0.67
                     (mm)

                                             in sample   cross-val
      state estimate, at reported angles
        nominal model                             3.47           -
        fitted model                              0.67        0.75  mm RMS
      command prediction, at commanded angles
        nominal model                            10.61           -
        fitted model                             10.52       10.54
        fitted model + residual                   2.61        6.27  mm RMS
      cross-validation: 5 folds, 10 shuffles; held-out RMS ranged ...

The nominal rows register the nominal chain with the board transform and the
camera offset free, so they isolate what the fitted chain and the residual
add over your tape measure; `vbody measure` holds the offset you gave, so its
reported column can be larger. The in-sample column scores the poses the fit
was made on; the cross-validation
column refits everything on four fifths of the poses and scores the fifth it
held back, repeated over shuffles, so nothing the fit saw is scored. The
cross-validated number is the one to quote. A large gap between the two
columns in the residual row means the residual is memorising poses rather
than learning the arm; more poses, or a larger `--ridge`, closes it.

Below the table the fit prints its parameters:

    fitted parameters: an operating condition for these poses, not the geometry of the arm
      board origin in the base frame  [-76.7, -386.7, -2.1] mm
      link scale                      Upper_Arm 0.9938   Lower_Arm 0.9890   Wrist_Pitch_Roll 0.9950
      zero offset                     Pitch -0.33 deg   Elbow +0.46 deg   Wrist_Pitch -1.27 deg
      camera offset                   [62.55, -50.25, -14.78] mm (you gave [60.0, -53.0, -15.0] mm)
      held                            ...
      residual                        27 features, ridge 0.0001, fitted on 54 commands

Read the first line literally. On the arm this tool was built for, parameters
that fitted one family of poses to under two millimetres left more than a
centimetre on a deeply flexed family. The fit describes the arm where you
measured it. Fit where you mean to work, and test with `vbody correct` on
targets from the same region.

The `held` line lists four parameters the fit does not free, because no
camera measurement can tell them apart from others: the base joint's zero is
a yaw of the board, the first link's length is a translation of the board,
the wrist link's length is the camera offset along the roll axis, and the
wrist roll zero is a rotation of the camera offset about that axis. The last
is freed when the camera offset is held with `--hold-camera-offset`. A
`weakly determined` line, when it appears, names parameters the poses barely
moved the camera through; their fitted values are not measurements, and
poses spread wider would pin them down.

## Approach direction

The residual has a term for the sign of the last motion in each joint, which
is what lets it express backlash. That term can only learn from poses that
were approached from both sides. A survey that returns home before every pose
approaches every pose from the same side in the joints that carry the load,
and the fit says so:

    the direction term has nothing to learn from for Pitch, Elbow, Wrist_Pitch:
    every pose was approached from the same side. Measure with --path.

`vbody measure --path` moves straight from each pose to the next. The path
between two poses that are each clear of the table is not itself checked, so
watch the first few moves of a path run with a hand near the switch.

## Evaluating a model on another run

A model file carries the arm's parameters and where the board was. To see how
the model does on poses it never saw, run

    vbody measure --camera-offset -80 -8 -13 ... --model model.json --out later.json
    vbody fit later.json --hold model.json

Either form holds the arm's parameters, places the board anew on the new
run's reported angles, since it may have been taped down again, and prints
the same summary as the fit, with the first column labelled for what it is.
This is the test of whether the parameters carry: measure a different family
of poses and see what is left.

`vbody correct` scores in the board placement the model was fitted with. If
the board has moved since, add `--reregister` and it will place the board on
the uncorrected pass before scoring both.

## Reading the correction

    target   uncorrected (mm)   corrected (mm)
         1              16.84             4.08
         2              19.59             5.50
         3              20.53             8.07
         4              22.31                -  corrected pass: board not visible
         5                  -             3.24  uncorrected pass: board not visible
         ...

    7 targets measured in both passes
                        mean      RMS    worst
      uncorrected      20.33    20.76    26.43  mm
      corrected         5.20     5.48     8.07  mm
      7 of 7 improved

That is the second batch of the paper's experiment, on a real arm. Each
target is a pose the nominal model believes puts the camera at some point.
The point is placed in the board frame by the nominal chain's own
registration, which the model file carries. It cannot be read in the fitted
chain's base frame: the fit trades link scales against the board transform,
and on a real arm that moves its base frame by centimetres. The uncorrected pass sends that pose; the corrected pass sends the
command the model predicts will actually reach the same point. Both errors
are distances from that point. Targets are drawn with a seed of their own,
and the command warns if it is the seed the model was measured with, because
then the targets are the calibration poses and the test proves nothing.

A target is refused, and skipped in the corrected pass, when the corrected
command would leave the joint limits, come within the clearance of the table,
or when the fitted chain cannot reach the shifted point at all. A predicted
residual longer than 50 mm is clamped to that length, since nothing on this
arm is that large and a prediction that is means the model is corrupt.

## Safety

Constructing the arm never turns torque on. Speed and acceleration limits are
written to every servo before torque can be enabled, so a move to a far pose is
a crawl rather than a swing.

vBody refuses to move if the calibration file is missing, if a pose falls
outside the joint limits, or if a pose would bring the gripper or the camera
within 100 mm of the table. The whole pose list is checked before the first
command, so a run either starts safely or does not start. Corrected commands
are checked the same way, and refused rather than sent.

Ctrl-C cuts torque and writes the results collected so far. The arm sags when
torque is cut, so do not leave it holding anything heavy or standing over
anything fragile, and keep a hand near the power switch during a run.

## Layout

    vbody/kinematics.py    forward kinematics, its Jacobian and inverse, the scan grid and the pose samplers
    vbody/vision.py        board, intrinsics, board pose, wrist camera
    vbody/arm.py           Feetech driver, and the simulated arm
    vbody/registration.py  board to base transform and the per-pose errors
    vbody/model.py         the body model: analytic fit, residual, evaluation, correction
    vbody/cli.py           the six commands
    vbody/assets/          the arm model the kinematics is read from

The model in `vbody/assets/so_arm101.xml` is the kinematic chain of the
TheRobotStudio SO-ARM100 MuJoCo model with the meshes and dynamics stripped
out. Link offsets and joint axes are unchanged. See `LICENSE.model` beside it.

## The paper

`paper.pdf` is "When Reported Joint Angles Are Not Enough: Local Calibration
of a Low-Cost Printed Arm", which reports what vBody measured on one
SO-ARM101. Its survey and correction experiments were run with this tool.

## Licence

MIT, see `LICENSE`. The arm model in `vbody/assets/` keeps its own Apache 2.0
licence, `vbody/assets/LICENSE.model`.

## Tests

    python -m pytest -q
