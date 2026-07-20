# Bugs to solve

Found while reading the codebase to scope the agentic-development project
(see `final_project_requirement.md`). Not fixed yet — tracked here so they
aren't lost before the camera is actually used for a mission.

## OAK-D camera (`donkeycar/parts/oak_d.py`)

This car uses an OAK-D camera (per `CLAUDE.md`), so these block any
CV-based autonomous behavior (line following, lane following, etc.) until
fixed.

1. **RGB preview-size crash.** ~~`setup_rgb_camera()` calls
   `cam_rgb.setPreviewSize(self.image_w, self.image_h)`, but `self.image_w`
   / `self.image_h` are never set anywhere in the class~~ **FIXED**: now
   calls `cam_rgb.setPreviewSize(width, height)` using the method's own
   params (which `setup_rgb_camera` is invoked with, currently the module
   constants `WIDTH, HEIGHT` = 640x480 — see note on #4).

2. **Preview size has no effect even once #1 is fixed.** ~~The pipeline
   links `cam_rgb.video.link(xout_rgb.input)` — the `.video` output, not
   `.preview`~~ **FIXED**: now links `cam_rgb.preview.link(xout_rgb.input)`
   so `setPreviewSize()` actually takes effect. Git history shows commit
   `8d4a707f` ("Switch OAK-D RGB output from video to preview stream
   (#1235)") intended this fix but never changed the link target — done now.

3. **No BGR→RGB conversion.** ~~`depthai`'s `ImgFrame.getCvFrame()` for a
   `ColorCamera` typically yields BGR~~ **FIXED**: added
   `cam_rgb.setColorOrder(depthai.ColorCameraProperties.ColorOrder.RGB)` so
   the camera emits RGB natively (matching `LineFollower`'s
   `cv2.COLOR_RGB2HSV` assumption), instead of converting in Python
   downstream like the other `add_camera()` branches do via `BGR2RGB`.

4. **Resize logic mismatch.** `_poll()` resizes via `cv2.resize` whenever
   `self.resize` is true (`width != 640 or height != 480`, hardcoded
   defaults). This was already consistent with the fix above — since
   `setup_rgb_camera` is still called with the module constants
   `WIDTH, HEIGHT` (640x480), not `self.width`/`self.height`, the capture is
   always native 640x480 and the resize step scales it down/up to whatever
   the caller actually requested. No change needed; documented here so the
   intent (native capture + software resize) is clear if this is revisited.

**Status: code fix applied in `donkeycar/parts/oak_d.py`. Not yet verified
against real hardware** — `depthai` isn't installed on this dev machine (per
`CLAUDE.md`, it must be pip-installed separately and the real car app lives
on the Pi). Please re-test on the car: confirm `enable_rgb=True` no longer
crashes, the streamed frame is actually `IMAGE_W`x`IMAGE_H`, and colors look
correct (e.g. point at something red and confirm it isn't rendered as blue).

## Cross-check with another team's line-following experiment

A teammate (sungsan) independently built a personal single-line-following
experiment (`/home/pi/mycar/*_sungsan.py`, not touching shared files) and
documented three problems. Cross-checked against the above:

- **BGR/RGB channel mismatch** — same root cause as bug #3 above, confirmed
  independently: their tests showed the raw OAK-D frame really is BGR and
  the stock line-color mask picks up cyan/gray/wall pixels when treated as
  RGB. Their personal controller works around it per-controller
  (`CV_INPUT_COLOR_ORDER="BGR"` + `cv2.COLOR_BGR2HSV`); our fix instead makes
  `oak_d.py` emit RGB natively at the source. **Coordination note:** once the
  `oak_d.py` fix is deployed to the Pi, their personal script's BGR
  assumption will be wrong again (double-inverts the channels) — they'll
  need to flip it to `CV_INPUT_COLOR_ORDER="RGB"` (or drop the conversion)
  after pulling the update.
- **Resolution mismatch** (their fixed `SCAN_Y=70` scanning the wrong part of
  the frame) — consistent with bug #2 above (`.video` link ignoring
  `setPreviewSize`, so frames stayed at a fixed hardware resolution
  regardless of `IMAGE_W`/`IMAGE_H`). Should no longer occur for the shared
  `line_follower.py` once the `oak_d.py` fix is on the Pi.
- **Color threshold values applied to shared config**: since we're driving
  on the same physical track/tape, applied their measured HSV yellow range
  to `donkeycar/templates/cfg_cv_control.py`:
  `COLOR_THRESHOLD_LOW = (18, 18, 35)`, `COLOR_THRESHOLD_HIGH = (35, 255, 255)`
  (previously `(0, 50, 50)` / `(50, 255, 255)`, untuned defaults). These are
  true-color HSV values so they transfer correctly regardless of BGR vs RGB
  pipeline internals — re-tune with `scripts/hsv_picker.py` if lighting or
  the tape changes.
- **Real bug fixed in shared `line_follower.py`**: their "Problem 2" —
  `TARGET_PIXEL = None` used to latch onto whichever pixel had the strongest
  color match in the *first frame*, which could be background clutter
  (wall/plant/reflection) rather than the actual line. Fixed to default to
  the image center (`cam_img.shape[1] / 2`) instead, matching the documented
  alternative of manually setting `TARGET_PIXEL = IMAGE_W / 2`.
- Everything else in their writeup (connected-component filtering, temporal
  jump prevention, candidate scoring, safe no-line stop, debug overlay) is
  scoped to their personal experimental controller, same situation as the
  unmerged `robustLineFollower` branch — not pulled into shared code for now.

## Docs path mismatch

`CLAUDE.md` refers to a gitignored `docs/` directory as the local copy of
docs.donkeycar.com, but that directory doesn't exist at that path. The
actual local copy currently lives at `donkeydocs-master/docs/` (also
gitignored). Not a functional bug, but the relative links in `CLAUDE.md`
(e.g. `docs/guide/computer_vision/computer_vision.md`) currently 404 —
worth fixing before someone follows a dead link.

## Hardware TODO (unconfirmed, not a bug)

`CLAUDE.md` has an open TODO that drivetrain hardware (PCA9685/servo-ESC vs.
other) and autopilot backend aren't settled yet. The source supports
`PWM_STEERING_THROTTLE` (standard servo+ESC) as the default drivetrain path,
and IMU support now includes BNO08x (`Bno08xIMU`, added recently alongside a
GPS+IMU EKF fusion part) in addition to MPU6050/MPU9250. Neither has been
confirmed as what's actually on this car — needs a check with teammates
before any doc/code assumes a specific drivetrain or IMU model.
