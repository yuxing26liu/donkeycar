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
- The actual car app (calibrated `myconfig.py`, `models/`, `data/`) lives at **`/home/pi/mycar`** on the Pi, as a sibling directory to wherever this repo is cloned — not inside this repo, and not baked into the Docker image.
- <!-- TODO: fill in drivetrain (PCA9685/servo-ESC vs. other), autopilot backend (Keras/TFLite/Torch), and any other hardware once settled -->

## Reference docs

- `docs/` is a local, gitignored, one-time copy of docs.donkeycar.com (from `autorope/donkeydocs`) — not tracked in this repo, won't update automatically, but useful to read locally.
- Our task is improving the car's autonomous behaviors. [docs/guide/computer_vision/computer_vision.md](docs/guide/computer_vision/computer_vision.md) documents the **baseline** behavior we're trying to improve on: the built-in `cv_control` template's `LineFollower` autopilot ([donkeycar/parts/line_follower.py](donkeycar/parts/line_follower.py)) — a traditional (non-learned) computer-vision approach that takes a horizontal HSV color-threshold slice of the camera image to find a line, then a PID controller steers toward it and throttles down on turns / up on straights. Any new autonomous-behavior work should be understood as a comparison against this baseline, not a from-scratch design.

- Consult the relevant page there before modifying a part (e.g. reference-docs/parts/camera.md before touching camera code). The docs are background context — if they conflict with what you find in the actual source, trust the source."
