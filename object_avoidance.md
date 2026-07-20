# Object Avoidance (Depth Sensor)

This page documents the `ObjectAvoider` part, which adds oncoming-object
avoidance on top of the Line Follower autopilot (see
`donkeydocs-master/docs/guide/computer_vision/computer_vision.md`) -- the
third stage of this project's mission progression (line following -> lane
following -> two-way road navigation). Read the Line Follower doc first;
this part is layered on top of it and does not change how line following
works.

The car is expected to drive in the gap between a yellow dashed centerline
and a white solid outer boundary, sharing the track with another car driving
the opposite direction on the other side of the centerline. `LineFollower`
already handles staying in-lane relative to the centerline (see `TARGET_PIXEL`
in the Line Follower doc). `ObjectAvoider` adds the other half: if the other
car (or any object) gets close, swerve away from it instead of colliding.

## Sensor: OAK-D depth camera, start to finish

This is the end-to-end integration path for the depth sensor, from hardware
to the control loop.

**Hardware.** The car uses a Luxonis OAK-D, an RGB + stereo depth camera
connected over USB. No separate depth sensor wiring is needed -- it's one
device exposing two streams (RGB preview and a stereo-computed depth map).

**Driver / SDK.** The `depthai` Python SDK talks to the device. It is **not**
declared in `setup.cfg` and must be `pip install`ed separately on the Pi (see
`CLAUDE.md`). On Linux you also need a udev rule so the USB device is
accessible without root -- see the comment at the top of
`donkeycar/parts/oak_d.py` for the exact command.

**Donkeycar part.** `donkeycar/parts/oak_d.py` wraps the SDK in the `OakD`
class. Its `depthai.Pipeline` links a `ColorCamera` node (RGB) and a
`StereoDepth` node (fed by two `MonoCamera`s, left/right) to two output
streams, `"rgb"` and `"depth"`. `run_threaded()` returns
`(color_image, depth_image)` -- `depth_image` is a `uint16` numpy array,
one depth reading per pixel, in **millimeters**, with `0` meaning "no valid
reading" (occluded, out of range, or low-texture surface the stereo matcher
couldn't resolve).

**Wiring into the vehicle loop.** `add_camera()` in
`donkeycar/templates/complete.py` instantiates `OakD` and adds it to the
`Vehicle` when `CAMERA_TYPE == "OAKD"`:

```python
cam = OakD(enable_rgb=cfg.OAKD_RGB, enable_depth=cfg.OAKD_DEPTH, device_id=cfg.OAKD_ID)
V.add(cam, inputs=[], outputs=['cam/image_array', 'cam/depth_array'], threaded=True)
```

`cv_control.py` calls this same `add_camera()`, so switching to the OAK-D for
the `cv_control` template is a **config-only** change (see below) -- no
template code changes were needed. Once added, `cam/depth_array` is just
another named value in the `Vehicle`'s shared memory store, available to any
part added afterward, exactly like `cam/image_array`.

## Configuration

All settings below live in `donkeycar/templates/cfg_cv_control.py` (the
shared template) and can be overridden per-car in `myconfig.py` on the Pi.

### Camera selection

```python
CAMERA_TYPE = "OAKD"   # set this in myconfig.py -- template default stays "PICAM"
OAKD_RGB = True         # stream RGB preview -> cam/image_array (used by LineFollower)
OAKD_DEPTH = True        # stream depth map -> cam/depth_array (used by ObjectAvoider)
OAKD_ID = None            # device serial number; None uses the only/default connected device
```

`CAMERA_TYPE` is left as `"PICAM"` in the shared template on purpose --
camera hardware is per-car, so each car's `myconfig.py` (which lives outside
this repo, at `/home/pi/mycar/myconfig.py`) is where it actually gets set to
`"OAKD"`.

### ObjectAvoider

```python
HAVE_OBJECT_AVOIDANCE = True   # set False to disable the part entirely

DEPTH_SCAN_Y = 100       # top of the horizontal depth scan slice, in pixels
DEPTH_SCAN_HEIGHT = 40   # height of the scan slice, in pixels

OBJECT_MIN_VALID_DEPTH_MM = 200    # ignore readings closer than this (lens-adjacent noise)
OBJECT_DANGER_DISTANCE_MM = 800    # swerve when the nearest object is closer than this
OBJECT_AVOID_STEERING = 0.8        # magnitude of the override steering value (-1..1)
OBJECT_AVOID_THROTTLE = THROTTLE_MIN  # throttle to use while avoiding
```

These defaults are **guesses, not yet tuned** -- the depth stream from
`oak_d.py` has not been verified on real hardware yet (see
`bugs_to_solve.md`). Tune them the same way `LINE_WIDTH_MIN`/`MAX` are tuned
for `LineFollower`: run with `LOGLEVEL=DEBUG` and watch the console.

## How it works

`ObjectAvoider.run(depth_array, steering, throttle)`
(`donkeycar/parts/object_avoider.py`):

1. Take a horizontal slice of the depth image at `DEPTH_SCAN_Y` /
   `DEPTH_SCAN_HEIGHT` -- the same slice-based approach `LineFollower` uses
   for its color scan, just applied to depth instead of color.
2. Mask out invalid pixels (`0`, or anything nearer than
   `OBJECT_MIN_VALID_DEPTH_MM`, which is lens-adjacent noise rather than a
   real object).
3. Reduce each column of the slice to its minimum depth, then find the
   column with the overall minimum -- the nearest point across the slice,
   and its horizontal position.
4. If that nearest distance is below `OBJECT_DANGER_DISTANCE_MM`: override
   steering away from that column (object left of image-center -> steer
   right, and vice versa) and clamp throttle to `OBJECT_AVOID_THROTTLE`.
5. Otherwise: pass `steering`/`throttle` through unchanged.

## Wiring into `cv_control.py`

The part is added to the `Vehicle` right after the CV autopilot
(`LineFollower`) and before `DriveMode` (the part that picks between
user and autopilot output):

```python
if getattr(cfg, 'HAVE_OBJECT_AVOIDANCE', False):
    from donkeycar.parts.object_avoider import ObjectAvoider
    V.add(ObjectAvoider(cfg),
          inputs=['cam/depth_array', 'pilot/steering', 'pilot/throttle'],
          outputs=['pilot/steering', 'pilot/throttle'],
          run_condition='run_pilot')
```

The `Vehicle` loop applies parts in the order they were added, and each
part's outputs overwrite the named memory keys it declares
(`donkeycar/vehicle.py`). Because `ObjectAvoider` is added after
`LineFollower` and writes the same `pilot/steering` / `pilot/throttle` keys,
it transparently overrides the line-follower's output whenever it detects a
close object, and is a no-op (passes the values through) the rest of the
time. This is the same pattern the built-in `StopSignDetector` part uses to
override `pilot/throttle` in `donkeycar/templates/complete.py`. Neither
`LineFollower` nor `DriveMode` needed any changes.

If `CAMERA_TYPE` isn't `"OAKD"` (or `OAKD_DEPTH` is `False`), `cam/depth_array`
is never populated (stays `None`), and `ObjectAvoider.run()` returns its
inputs unchanged on the first line -- so this part is safe to leave enabled
even on a car without a depth camera.

## Testing / tuning on the car

This can't be verified off the car (no OAK-D hardware or `depthai` install on
a dev machine). On the car:

1. Set `CAMERA_TYPE = "OAKD"` in `myconfig.py` and confirm the car still
   drives normally with the RGB stream (line following unaffected).
2. Confirm `cam/depth_array` is populated with sane values -- add a temporary
   log of its shape/min/max, or watch the `LOGLEVEL=DEBUG` output described
   above.
3. Slowly walk an object into the depth scan slice and confirm the console
   logs a plausible nearest-distance and column, and that the car swerves
   the correct direction. If it swerves the wrong way, negate
   `OBJECT_AVOID_STEERING` -- the sign convention for "left" vs. "right"
   depends on the drivetrain and can't be predicted without testing.
4. Tune `OBJECT_DANGER_DISTANCE_MM` and `DEPTH_SCAN_Y`/`DEPTH_SCAN_HEIGHT`
   against the real track and the other car's approach speed.
