# CLAUDE.md

Fork of [autorope/donkeycar](https://github.com/autorope/donkeycar) — a self-driving RC car built on this library.

## Architecture

- Everything runs through a `Vehicle` loop ([donkeycar/vehicle.py](donkeycar/vehicle.py)): a fixed-rate loop over a list of `parts`, reading/writing named channels in a shared `Memory` key-value store. Each part is a plain class with `run()`/`run_threaded()`/`update()`/`shutdown()`.
- `donkeycar/parts/` — one file per sensor/actuator/autopilot backend (camera, controller, actuator, keras, pytorch, etc.).
- `donkeycar/templates/` — the car "app" templates (e.g. `complete.py`). `donkey createcar` copies one of these into a separate car-project folder as `manage.py`, plus `cfg_*.py` → `config.py` and a blank `myconfig.py` for local overrides.
- `donkeycar/management/base.py` — the `donkey` CLI (`donkey createcar`, `donkey train`, `donkey calibrate`, etc.).
- Requires **Python 3.11** specifically (enforced both in `setup.cfg` and at import time in `donkeycar/__init__.py`).

## This car's setup

- Camera: **OAK-D** (`donkeycar/parts/oak_d.py`, via the `depthai` SDK) — not the Pi Camera Module. We don't need `picamera2`/`libcamera`.
- `depthai` is **not declared anywhere in `setup.cfg`** — it must be pip-installed separately regardless of which extras group you use.
- **CV resolution is strictly 426x240** (`IMAGE_W`/`IMAGE_H` in `myconfig.py`, plumbed into `OakD(width=..., height=...)`). `oak_d.py`'s `ColorCamera` node must link its `.preview` output (scaled in hardware to `width`/`height`) rather than `.video` (which always streams at the full sensor resolution regardless of `setPreviewSize` — the historical bug that had this fork's camera silently running at 1920x1080). Any pixel-based tuning constant (`SCAN_Y`, `SCAN_HEIGHT`, `TARGET_THRESHOLD`, steering `PID_P`/`PID_D`, shape-filter constants like `MIN_LINE_WIDTH`/`MAX_JUMP_PIXELS`) is calibrated against this resolution — rescale and re-verify against real camera footage before changing it.
- The actual car app (calibrated `myconfig.py`, `models/`, `data/`) lives at **`/home/pi/mycar`** on the Pi, as a sibling directory to wherever this repo is cloned — not inside this repo, and not baked into the Docker image.
- <!-- TODO: fill in drivetrain (PCA9685/servo-ESC vs. other), autopilot backend (Keras/TFLite/Torch), and any other hardware once settled -->

## Current focus: right-lane driving + obstacle avoidance

- The track has two solid lines (one on each outer edge) and one dashed yellow line down the middle separating two lanes. The goal is to drive in the **right lane** (between the right solid edge and the center dashed yellow line) and **swerve around obstacles** that appear on the track, rather than just centering on a single line.
- `donkeycar/parts/lane_follower.py` (`LaneFollower`) is the new autopilot being built for this — a successor to the single-line trackers below. As of now it's just a file skeleton (imports/class stub); the actual lane-detection + obstacle-avoidance design is still being planned.
- Prior single-line-tracking iterations, in rough chronological order, for reference when designing `LaneFollower`: `LineFollower` (baseline, see below) → `RobustLineFollower` (`donkeycar/parts/robust_line_follower.py`, on the `robustLineFollower`/`robustRyan` branches — adds shape filtering and position continuity/tracking on top of the baseline) → `BetterLineFollower` (`donkeycar/parts/better_line_follower.py`, on the `better-line-follower` branch — further shape/color/PID fixes plus sustained-line-loss handling). None of these are lane-aware; they all track exactly one line.

## Reference docs

- `docs/` is a local, gitignored, one-time copy of docs.donkeycar.com (from `autorope/donkeydocs`) — not tracked in this repo, won't update automatically, but useful to read locally.
- Our task is improving the car's autonomous behaviors. [docs/guide/computer_vision/computer_vision.md](docs/guide/computer_vision/computer_vision.md) documents the **baseline** behavior we're trying to improve on: the built-in `cv_control` template's `LineFollower` autopilot ([donkeycar/parts/line_follower.py](donkeycar/parts/line_follower.py)) — a traditional (non-learned) computer-vision approach that takes a horizontal HSV color-threshold slice of the camera image to find a line, then a PID controller steers toward it and throttles down on turns / up on straights. Any new autonomous-behavior work should be understood as a comparison against this baseline, not a from-scratch design.

- Consult the relevant page there before modifying a part (e.g. reference-docs/parts/camera.md before touching camera code). The docs are background context — if they conflict with what you find in the actual source, trust the source."

