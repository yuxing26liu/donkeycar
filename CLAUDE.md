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

## Docker

- `setup.cfg`'s extras groups (`pi`, `pc`, `nano`, `macos`) are for the full donkeycar experience (driving + training + the Kivy GUI). Some of what they pull in is slow to build from source (`python-prctl`, `stringzilla`/`simsimd` via `albumentations`) but does eventually succeed given enough time and the right system libs — it's not actually broken, just slow.
- Needed apt packages for `pip install -e .[pi]` to build cleanly: `build-essential`, `python3-dev`, `libcap-dev` (headers `python-prctl`, a `picamera2` dependency, needs to compile), `libhdf5-dev` (Keras model save/load), `libatlas3-base`/`libopenblas-dev` (numpy math).
- **Build natively on the target device — don't cross-compile.** A `docker build` on the laptop produces an `amd64` image; the Pi is `arm64`. An amd64 image will not run on the Pi (`exec format error`). Build on the Pi itself (e.g. over the existing Remote SSH terminal session) for anything meant to actually run on the car.
- `RPi.GPIO`'s installer does hardware detection and is only reliably installable on real Raspberry Pi hardware — another reason to build on-device rather than cross-compiling or building on the laptop.
- At `docker run` time, mount the real car folder and pass through hardware devices — don't rely on anything in the image for calibration/data:
  ```bash
  docker run -it --rm --privileged -v /dev:/dev \
    -v /home/pi/mycar:/mycar -w /mycar \
    donkeycar-dev python3 manage.py drive
  ```

## Gotchas

- `opencv-contrib-python` (non-headless, in the `pi`/`pc` extras) needs GUI runtime libs (`libGL.so.1`, etc.) that a minimal base image won't have unless installed — prefer `opencv-contrib-python-headless` if the GUI/imshow features aren't needed, which they aren't for `manage.py drive`.
- `kivy` and `albumentations` are only used by the desktop GUI (`donkey ui`) and the training augmentation pipeline, respectively — neither is imported by `manage.py drive`, so they're not required for a drive-only container.
