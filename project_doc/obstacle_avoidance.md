# Obstacle Avoidance: Plan and Status

Third stage of this project's mission progression (see `CLAUDE.md`): line
following -> lane following (`LaneFollower`, `donkeycar/parts/lane_follower.py`,
done and working) -> **two-way road navigation**, this document. On the
right-lane track (one white outer edge + one dashed yellow centerline, see
the reference photos in `CLAUDE.md`), the car must additionally:

1. Swerve around a **traffic cone** placed in our own lane, and stay in
   whichever lane that leaves us in for the rest of the drive rather than
   swerving back (see Decision 1.5 - revised from the original "return"
   plan).
2. Watch for an **oncoming car** in the opposite lane: ignore it while it
   stays in its own lane, swerve away if it crosses into ours.
3. Do both while ignoring background clutter (leaves/debris on the track,
   shadows, glare, the concrete's own expansion-joint seams) and without
   losing lane-keeping.

This is being built incrementally, one detector at a time, verified before
the next piece is layered on. **Step 1's detector and the cone avoidance
maneuver (Phase 2, below) are implemented; the oncoming-car detector is
not.**

## Design decisions

Each of these was evaluated as a set of options before picking one. Two
physical facts about the track drove the final choices, discovered partway
through design:

- The traffic cone's spot on the track is marked with a **square of blue
  tape** on the ground (visible in the reference photos in `CLAUDE.md`).
- The oncoming car has **black wheels and a black front**.

### Decision 1 — How to detect the cone

| Option | Pros | Cons |
|---|---|---|
| A. Detect the blue tape square (originally chosen) | Large, flat, high-saturation ground marking — a much more reliable classical-CV target at camera resolution/distance than a small cone silhouette; blue doesn't collide with anything else in play (gray concrete, white line, yellow dashes); doesn't require knowing the cone's own paint color. | It's a proxy for the cone's *position*, not the cone itself — if a cone is ever placed off its tape mark, this misses it. |
| B. Detect the cone's own color | Detects the real obstacle, not a proxy. | Cone's actual color unconfirmed; a small object is a weaker blob than a tape square at the same distance. |
| **C. Both (union of masks) — current** | Most robust; each detector is a strict positive-color match, so having both doesn't add false-positive risk over having either alone. | More tuning surface; blue also appears on the (off-track, background) recycling bin and a kiosk sign in the reference photos/footage. |

Revised from **A** to **C** after checking real on-car footage
(`data/tub_33_26-07-24`, 6463 frames): the blue tape marker described in
option A **does not appear anywhere in that recording** — every blue match
in the whole tub, at any position, traced back to the recycling bin or
kiosk sign at the image edges, never the track surface. The obstacles
actually on the track are plain orange cones with no ground marker at all.
Rather than assume every future track uses tape, `ObstacleAvoider.detect_cone`
now runs both the blue-tape and orange-cone HSV detectors every frame and
picks whichever produces the larger valid blob (see
`donkeycar/parts/obstacle_avoider.py`) — this was already anticipated as
"a config change, not a redesign" and reduces to option A's original
behavior if a track's tape marker is what's actually placed.
`ORANGE_HSV_THRESHOLD_LOW/HIGH` (unlike `BLUE_HSV_THRESHOLD_LOW/HIGH`,
which is still an untuned guess) was calibrated against real cone pixels
sampled directly from two frames of `tub_33_26-07-24`.

### Decision 2 — How to detect the oncoming car (not yet implemented)

| Option | Pros | Cons |
|---|---|---|
| **A. Color-key black (wheels/front) — planned next** | Simple, consistent with the rest of this codebase's color-blob approach; naturally ignores leaves/debris (not black) without extra logic. | Two known false-positive risks on this track: shadows cast on the concrete, and the dark expansion-joint seams visible in the reference photos. Needs shape/size filtering + a short trigger-hold, same as the cone detector. |
| B. Foreign-object-by-elimination (mask out everything already known, largest leftover blob = car) | Doesn't need to know the car's color. | Dropped once the black-wheels/front fact was known — strictly worse than A: any unexpected pixel (leaf, gravel, glare, tape edge) is a false positive by construction, since it's a not-known-stuff detector rather than a positive match. |
| C. A, plus a frame-over-frame blob-growth check | Extra guard against a *static* dark false positive (permanent shadow, the expansion joint) — a real approaching car's blob should grow; a static dark patch shouldn't. | More state, slower to trigger, another blind parameter. Held in reserve — only worth building if A's shape/size filter + trigger-hold isn't enough in on-car testing (YAGNI otherwise). |

Plan: **A**, with **C** as a documented fallback if the seam/shadow proves to
be a recurring false trigger during testing.

### Decision 1.5 — Return to the original lane after clearing the cone? (revised, implemented)

| Option | Pros | Cons |
|---|---|---|
| A. Swerve to the other lane, then swerve back once the cone is behind us (original plan) | Spends the least time in the "wrong" lane (matters once the oncoming-car detector exists). | A second maneuver = a second place to get the trigger condition wrong (when is the cone safely "cleared"?); on a short test track with only one cone, the extra complexity wasn't buying much. |
| **B. Swerve once, stay in the new lane for the rest of the drive (current, implemented)** | One maneuver, not two - nothing to get wrong about *when* to swerve back. Lower risk of clipping the cone on a premature return. | Wrong-lane for the remainder of the drive - only acceptable because the oncoming-car detector (Decision 2) isn't built yet; revisit once it is. |

Changed from A to B per direct instruction: once `ObstacleAvoider.avoiding`
latches (a cone confirmed in our lane), it never un-latches -
`ObstacleAvoider._avoid_step` steers toward the other lane's center
(`_other_lane_center`) for the rest of the drive, regardless of whether the
cone is still visible. The priority is staying on the track, not restoring
the original lane.

### Decision 3 — Losing the lane / uncertain geometry mid-maneuver (design settled, applies once a maneuver exists)

| Option | Pros | Cons |
|---|---|---|
| **A. Fully passive fallback: no valid lane geometry -> never trigger; if already avoiding, hold last output rather than steer off a guess (chosen)** | Safest — can't steer confidently off stale/missing data. | Could abort a maneuver early if `lane/yellow_x` flickers mid-swerve. Judged unlikely: `LaneFollower`'s own continuity gating already smooths normal dashed-line gaps at the source, so `lane/yellow_x` rarely goes fully `None` in normal driving. |
| B. Dead-reckon a few frames on last-known geometry through a drop-out | Smoother through a brief flicker. | Risk of continuing to swerve on stale/wrong geometry if the drop-out is real (car actually left the track). Not obviously justified yet. |

### Decision 4 — Scan geometry (design settled, applies once the car detector exists)

| Option | Pros | Cons |
|---|---|---|
| A. One shared forward scan band for both detectors | Simpler. | Physically the two detectors want different lead distances. |
| **B. Separate configurable scan rows per obstacle type (chosen)** | The cone is static (only needs to be seen before we reach it); the oncoming car is closing at combined speed and benefits from an earlier/farther scan row. Minimal extra config (a second Y value). | — |

`CONE_SCAN_Y`/`CONE_SCAN_HEIGHT` exist now (see below); a `CAR_SCAN_Y`/
`CAR_SCAN_HEIGHT` pair is expected when the car detector is built.

### Decision 5 — Simultaneous triggers (design settled, applies once a maneuver exists)

Once a maneuver is active for one obstacle type, ignore triggers of the
other type until back to cruising — a single steering actuator can't react
to both at once, and first-detected wins.

## What's implemented now: Phase 1 (cone detection) + Phase 2 (avoidance maneuver)

`donkeycar/parts/obstacle_avoider.py`, class `ObstacleAvoider`.

**Phase 1 is detection**, exactly as below. **Phase 2 (new) is the actual
maneuver**: once a cone latches as detected (`self.cone_detected` held for
`CONE_TRIGGER_FRAMES` consecutive frames), `self.avoiding` latches `True`
*permanently* and `ObstacleAvoider._avoid_step` overrides `pilot/steering`/
`pilot/throttle` for the rest of the drive - steering toward the other
lane's center (`_other_lane_center`, a `_lane_bounds(..., other_lane=True)`-style
mirror off `lane/yellow_x`) using a dedicated PID (`self.avoid_pid`, kept
separate from `LaneFollower`'s own `pid_st` so the two loops' integral
state can't corrupt each other). It does **not** swerve back to the
original lane once the cone is behind us - see Decision 1.5 above.

If lane geometry (`lane/yellow_x`) is lost mid-maneuver, `_avoid_step` holds
the last steering value and decays it (`LOST_STEERING_DECAY`, reusing
`LaneFollower`'s own sustained-loss constants) rather than guessing off
stale/missing data - Decision 3's passive fallback, applied here too.

### How detection works

1. Take a horizontal slice of the raw camera frame (`cam/image_array`, not
   whatever `LaneFollower` drew on `cv/image_array`) at `CONE_SCAN_Y` /
   `CONE_SCAN_HEIGHT` — the same slice-based approach every CV part in this
   codebase uses (`LineFollower`, `LaneFollower`'s per-color trackers, the
   unmerged `ObjectAvoider` prototype on `origin/estella`).
2. Convert to HSV and threshold on both `BLUE_HSV_THRESHOLD_LOW/HIGH` (tape
   marker) and `ORANGE_HSV_THRESHOLD_LOW/HIGH` (the cone's own body) to find
   a blob of either color, reusing `_select_line_blob` from
   `lane_follower.py` (the same connected-component shape/size filter the
   line trackers use) rather than writing a second implementation. If both
   colors produce a valid blob in the same frame, the larger one wins.
3. Compute our lane's pixel bounds (`_lane_bounds`, a small helper in this
   file — see "Why lane_follower.py wasn't touched" below) from
   `lane/yellow_x`, `lane/white_x`, `lane/width_px`, which `LaneFollower`
   already publishes every frame.
4. A detection only "counts" if it falls inside our lane's bounds (plus
   `LANE_SHIFT_MARGIN_PX`) for `CONE_TRIGGER_FRAMES` consecutive frames —
   rejecting a single noisy frame (glare, a leaf's edge) the same way
   `lane_follower.py`'s continuity gating rejects a single dropped frame
   for the dashed yellow line.
5. `obstacle/cone_detected` (bool) is published every frame; if
   `OVERLAY_IMAGE` is set, the detection is also drawn on `cv/image_array`
   (orange box = in our lane, gray box = detected but in the other lane).

### Terminal diagnostics (what to watch when verifying on the car)

With `cv_control.py`'s default logging (`--log=INFO`, no flag needed),
three things print to the terminal, independent of each other and of
whether a maneuver would trigger:

- **Once, at startup** (when `V.add(ObstacleAvoider(cfg), ...)` constructs
  the part), a line confirming the part is actually alive and what it's
  scanning for, e.g.:
  ```
  [cone_tape] ObstacleAvoider active - scanning rows [60,90) for blue tape HSV=(95, 100, 60)-(130, 255, 255) or orange cone HSV=(0, 90, 60)-(18, 255, 255)
  ```
  This is the direct answer to "is this part even wired in" — its absence
  means either `HAVE_OBSTACLE_AVOIDANCE` is `False` or `ObstacleAvoider`
  was never added to `V` at all (e.g. a `manage.py` that predates this
  wiring — see "Does this actually run" below), as opposed to "wired in but
  nothing detected yet", which prints nothing else until a cone appears.
- **Whenever a blue-tape or orange-cone blob enters/leaves the scan band**,
  one line naming which color matched, with the actually-sampled color at
  the detection, e.g.:
  ```
  [cone_tape] orange cone candidate at x=219.5, scan_y=60 - sampled color HSV=(8,200,190) RGB=(230,95,15) - IN our lane
  ```
  This repeats every `CONE_LOG_INTERVAL_FRAMES` frames (default 10, ~0.5s at
  20Hz) while the cone stays in view, so it's usable for live tuning — move
  the car/cone and watch the HSV number track. Fires regardless of lane
  geometry, so it confirms the color detectors themselves work even before
  `lane/*` is wired correctly (see the gotcha below). The sampled patch is
  centered on the scan band's vertical midline, not averaged over the whole
  band height — if the marker doesn't fill `CONE_SCAN_HEIGHT`, a full-height
  average would blend in the gray track and under-report saturation/value;
  this was caught by `test_raw_detection_logs_sampled_color_even_without_lane_geometry`
  actually failing (hue landed near 80, not blue's ~120) before the fix.
- **`obstacle/cone_detected` transitions** (a separate, debounced line) once
  a cone has been in our lane for `CONE_TRIGGER_FRAMES` consecutive
  frames, and again when it clears.

### Known gotcha: `CV_CONTROLLER_OUTPUTS` must include the `lane/*` keys

`LaneFollower.run()` returns 6 values: `(steering, throttle, image,
yellow_x, white_x, lane_width_px)`. The `Vehicle`/`Memory` plumbing
(`donkeycar/vehicle.py` `update_parts()` -> `Memory.put()`,
`donkeycar/memory.py`) assigns those 6 return values to
`cfg.CV_CONTROLLER_OUTPUTS` **by position, and silently drops anything past
the end of that list** if it's shorter than the tuple — no error, no
warning. The template's own default `CV_CONTROLLER_OUTPUTS` in
`cfg_cv_control.py` is still the 3-element
`['pilot/steering', 'pilot/throttle', 'cv/image_array']` (left that way
because it's shared with `LineFollower`, which only returns 3 values).

Since steering and throttle are first in `LaneFollower`'s tuple, **the car
drives identically either way** — a 3-element `CV_CONTROLLER_OUTPUTS`
doesn't break lane-keeping, it just means `lane/yellow_x`, `lane/white_x`,
`lane/width_px` are never written to `Memory`, `ObstacleAvoider` always sees
`None` for them, `_lane_bounds` always returns `(None, None)`, and the cone
can never be judged "in our lane" — with no symptom visible from driving
behavior alone. **This must be checked directly in the car's
`myconfig.py`** (outside this repo, at `/home/pi/mycar/myconfig.py` — not
verifiable from this repo alone):

```python
CV_CONTROLLER_MODULE = "donkeycar.parts.lane_follower"
CV_CONTROLLER_CLASS = "LaneFollower"
CV_CONTROLLER_OUTPUTS = ['pilot/steering', 'pilot/throttle', 'cv/image_array',
                          'lane/yellow_x', 'lane/white_x', 'lane/width_px']
```

If this isn't set, `ObstacleAvoider` now logs a one-time warning
(`_warn_if_lane_geometry_missing`) naming exactly this, instead of silently
never triggering.

### Why a positive color match, not "detect anything unusual"

The alternative (mask out everything already known about the track, treat
whatever's left as an obstacle) was seriously considered for the car
detector (Decision 2, option B) and rejected specifically because it's
fragile against exactly the failure modes this project called out up
front — leaves/debris, shadows, glare, ground misdetection. A positive
match against a known, saturated color (blue tape here; planned black for
the car) simply doesn't fire for things that aren't that color, which is a
much stronger guarantee against clutter than "isn't something else."

### Why `lane_follower.py` wasn't touched

Zero lines changed there. `ObstacleAvoider` is purely downstream, reusing:

- `LaneFollower`'s already-published per-frame outputs
  (`lane/yellow_x`, `lane/white_x`, `lane/width_px`) instead of re-deriving
  lane geometry — `LaneFollower`'s own class docstring anticipated exactly
  this ("so that obstacle-avoidance parts added later ... can read the raw
  lane geometry without re-running any CV").
- `_select_line_blob` (imported directly) for its own color-blob detection.

The one piece of lane geometry math that *is* duplicated is `_lane_bounds`
in `obstacle_avoider.py`, which mirrors `LaneFollower._lane_center`'s logic
but returns a lane's full `[low, high]` pixel extent instead of just its
center (needed to test "is x inside this lane", not just "how far off
center is x"). This was a deliberate choice over extracting a shared helper
into `lane_follower.py`: it keeps that file's working, tuned code completely
untouched, at the cost of two copies of a ~10-line formula that must be kept
in sync if the lane-geometry model ever changes.

### Wiring

`donkeycar/templates/cv_control.py`, added after the CV controller
(`LaneFollower`) and before recording, opt-in via
`HAVE_OBSTACLE_AVOIDANCE` (default `True` as of this project's current
focus — was `False`; flipped because detection-only means it's safe to
leave on, and leaving it off was the reason no terminal output was ever
observed on the car despite the code being correct):

```python
if getattr(cfg, 'HAVE_OBSTACLE_AVOIDANCE', False):
    from donkeycar.parts.obstacle_avoider import ObstacleAvoider
    V.add(ObstacleAvoider(cfg),
          inputs=['cam/image_array', 'lane/yellow_x', 'lane/white_x', 'lane/width_px',
                  'pilot/steering', 'pilot/throttle', 'cv/image_array'],
          outputs=['pilot/steering', 'pilot/throttle', 'cv/image_array', 'obstacle/cone_detected'],
          run_condition='run_pilot')
```

Requires `CV_CONTROLLER_CLASS = "LaneFollower"` (set in `myconfig.py`, per
`LaneFollower`'s own docstring) — with `LineFollower`, `lane/yellow_x` etc.
are never populated, so `_lane_bounds` always returns `(None, None)` and no
cone detection can ever be "in our lane". Safe to leave enabled regardless:
it degrades to reporting `cone_detected = False` always, never touching
`pilot/steering`/`pilot/throttle`.

### Does this actually run in full-auto mode on the car?

Traced through, not assumed:

- `run_condition='run_pilot'` — `Vehicle.update_parts()`
  (`donkeycar/vehicle.py`) reads `mem.get(['run_pilot'])` each loop tick and
  skips the part's `run()` entirely (inputs untouched, outputs untouched)
  when it's `False`.
- `run_pilot` is set by `UserPilotCondition.run()` (`donkeycar/templates/complete.py`):
  `True` whenever `user/mode != 'user'` — i.e. in both `'local_angle'`
  (assisted-steering) and `'local_pilot'` (full auto) modes. Only manual
  `'user'` mode is skipped.
- So yes: in full-auto mode, `ObstacleAvoider.run()` executes every loop
  tick alongside `LaneFollower`, in the order they were added to `V`
  (`LaneFollower` first, so its `lane/*` and `pilot/*` outputs are already
  in `Memory` — same loop tick, no staleness — by the time `ObstacleAvoider`
  reads them).
- The one thing this trace can't confirm from inside this repo is whether
  `myconfig.py` on the Pi overrides `HAVE_OBSTACLE_AVOIDANCE` back to
  `False`, and whether it sets the full `CV_CONTROLLER_OUTPUTS` list from
  the gotcha above — both live outside this repo and need to be checked
  directly on the car.
- **A real gap found via `data/tub_33_26-07-24`'s `manifest.json`**: that
  tub's declared schema is only `["cam/image_array", "steering", "throttle"]`
  — not the 6-field list `cv_control.py`'s `TubWriter` always declares
  (adding `lane/yellow_x`, `lane/white_x`, `lane/width_px`), regardless of
  `CV_CONTROLLER_CLASS`. `Manifest._read_contents()`
  (`donkeycar/parts/datastore_v2.py`) hard-asserts an existing tub's
  declared schema matches the constructor's `inputs` exactly, and would
  crash on mismatch — since this tub recorded 6463 frames without crashing,
  whatever produced it declared exactly those 3 fields the whole time. That
  means it wasn't running this repo's current `cv_control.py` at all. Since
  `donkey createcar` *copies* a template into the car project's `manage.py`
  rather than importing it live (per `CLAUDE.md`), the likely explanation is
  that `/home/pi/mycar/manage.py` is a stale copy predating the
  `ObstacleAvoider` wiring block entirely — meaning no config change in this
  repo (including flipping `HAVE_OBSTACLE_AVOIDANCE`) will take effect until
  `manage.py` is regenerated/re-copied from the current `cv_control.py` on
  the car. **This needs to be checked directly on the Pi** — not verifiable
  or fixable from this repo.

### Configuration

All in `donkeycar/templates/cfg_cv_control.py`, overridable per-car in
`myconfig.py` (which lives outside this repo, at `/home/pi/mycar/myconfig.py`
— see `CLAUDE.md`):

```python
HAVE_OBSTACLE_AVOIDANCE = True   # detection-only regardless, so safe to leave on

CONE_SCAN_Y = 60         # top of the forward scan slice, in pixels
CONE_SCAN_HEIGHT = 30    # height of the scan slice, in pixels

ORANGE_HSV_THRESHOLD_LOW = (0, 90, 60)      # calibrated against real cone pixels
ORANGE_HSV_THRESHOLD_HIGH = (18, 255, 255)  # sampled from tub_33_26-07-24 - see Decision 1

BLUE_HSV_THRESHOLD_LOW = (95, 100, 60)     # guessed, not yet tuned on hardware
BLUE_HSV_THRESHOLD_HIGH = (130, 255, 255)  # -- see "Testing / tuning" below

CONE_MIN_AREA_PX = 80    # smallest pixel area (in the scan slice) counted as the cone/marker
CONE_MAX_WIDTH_PX = 400  # widest pixel width (in the scan slice) counted as the cone/marker -
                          # raised from 250 after tub_41_26-07-24 showed a close, centered
                          # cone's own blob measuring 278-295px wide and getting rejected
                          # outright, right when avoidance mattered most (see below)

LANE_SHIFT_MARGIN_PX = 10  # margin added to our lane's bounds when testing membership
CONE_TRIGGER_FRAMES = 2    # consecutive in-lane frames required before latching

CONE_LOG_INTERVAL_FRAMES = 10  # re-print the sampled color this often (frames)
                                # while the tape stays in view -- see
                                # "Terminal diagnostics" above
```

Like every other CV threshold in this codebase (`COLOR_THRESHOLD_LOW/HIGH`,
`YELLOW_HSV_THRESHOLD_LOW/HIGH`, ...), the HSV bounds above are **an
untuned starting guess**, not a measurement — they need to be checked
against real tape under the car's actual lighting.

## Testing performed

`donkeycar/tests/test_obstacle_avoider.py`, 28 tests, all passing (run with
`conda activate donkey && python -m pytest donkeycar/tests/test_obstacle_avoider.py -v`
— this repo's dev tooling lives in the `donkey` conda environment, not the
system Python). All synthetic-image tests (no camera/hardware needed):

- `_lane_bounds` geometry: both lines visible, single-line extrapolation in
  each direction, the mirrored "other lane" calculation, and the
  `white_right_of_yellow=False` (left-lane) sign flip.
- Cone detected in our lane latches `cone_detected` after
  `CONE_TRIGGER_FRAMES` consecutive frames, at approximately the right x —
  for both the blue-tape and orange-cone detectors independently.
- Cone detected in the *other* lane is found but correctly not counted as
  "in our lane" (never latches) — both colors.
- A blue or orange patch outside the scan band, or too small
  (`< CONE_MIN_AREA_PX`), is ignored.
- When both colors produce a candidate in the same frame, the one that
  actually passes the shape filter wins (`test_larger_blob_wins_when_both_colors_present`).
- **Robustness / "ignore the background" checks:** a green (leaf-colored)
  patch in our lane doesn't trigger; a white line + yellow line drawn
  through the scan band don't trigger; a single detected frame followed by
  a miss correctly resets the debounce counter instead of carrying over.
- `cam_img=None` and the overlay path both pass through cleanly without
  altering `steering`/`throttle`.
- **Diagnostics:** a raw detection logs its sampled HSV/RGB color even with
  no lane geometry available, tagged with which color matched, and the
  logged hue is asserted close to true blue's ~120 / orange's ~10 (this
  test caught the vertical-averaging bug described in "Terminal
  diagnostics" above — it originally logged ~80 before the fix); the log
  clears on loss; a one-time startup line names both configured color
  ranges; the missing-lane-geometry warning fires exactly once across
  repeated frames, and never fires when geometry is present.

**Also validated against real camera footage** (`data/tub_33_26-07-24`,
6463 frames) — not a substitute for on-car verification, but stronger than
synthetic images alone:

- Sampled real cone pixels from two separate frames (saturation-filtered to
  exclude the cone's white reflective stripe and cast shadow) to calibrate
  `ORANGE_HSV_THRESHOLD_LOW/HIGH`, rather than guessing.
- Ran the actual `detect_cone()` over every 5th frame of the tub: the
  orange detector fires at the frames independently confirmed (by eye) to
  contain a cone, and stays quiet on frames without one.
- Ran the old blue-only detector the same way first, *before* adding
  orange: confirmed the blue tape marker from Decision 1's original design
  does not appear anywhere in this recording — every blue match traced to
  the recycling bin or kiosk sign at the image edges. This is what drove
  the Decision 1 revision and the orange detector's addition.

**Still not verified:** `BLUE_HSV_THRESHOLD_LOW/HIGH` remains an untuned
guess (no tape marker was present in the only footage checked so far), and
neither threshold has been checked against footage from a different
lighting condition or camera than `tub_33_26-07-24`. Same caveat every
other CV threshold in this codebase carries.

**On-car run, `tub_41_26-07-24` (terminal log, no frames recorded — see
"Does this actually run" below for why tubs from this session are empty):**
surfaced two real problems, not one:

1. **The car still drove straight into the cone even though the log showed
   repeated `SWERVE` triggers.** The traceback at the end of the log names
   `/home/pi/mycar/manage2.py`, and every trigger in the log prints the
   *old*, pre-avoidance-maneuver `_decide_action` text
   (`"SWERVE - ... confirmed in our lane, steer toward other lane"`) —
   never the current code's `"beginning avoidance maneuver ... will NOT
   return to the original lane"` / `"AVOIDING"` text, and `cone_detected`
   visibly un-latches and re-latches repeatedly (impossible once `avoiding`
   is set, since it never un-latches in the current code). This means the
   Pi was still running a stale copy of `obstacle_avoider.py` from before
   the avoidance maneuver existed — detection-only, exactly like Phase 1.
   **Not fixable from this repo**; the updated `donkeycar/parts/obstacle_avoider.py`
   needs to actually be deployed to `/home/pi/mycar`/the Pi's installed
   `donkeycar` package before any steering-override change here takes
   effect on the car.
2. **`CONE_MAX_WIDTH_PX=250` rejected the cone at exactly the range that
   matters most.** Repeated log lines like `"orange cone: 8654px matched
   but rejected (largest blob width 294px > CONE_MAX_WIDTH_PX 250)"` show
   a large, unambiguous orange blob (~65% of the whole scan band) being
   thrown out purely for being wide — precisely when the cone was close
   and centered. The width cap makes sense for the line tracker it was
   borrowed from (rejecting a wide sunlit patch of pavement pretending to
   be a paint stripe) but not for a real 3D cone, which legitimately fills
   much of the frame up close; since orange is a strong, exclusive color
   match on this track, a wide orange blob is stronger evidence of a real
   cone, not weaker. Fixed by raising `CONE_MAX_WIDTH_PX` to 400 (see above).

## On-car feedback (2026-07-25): overshoot off the track + false trigger on noisy lane width

Testing surfaced two more real problems, both traced to the same root
cause: `LaneFollower`'s published `lane/width_px` swung far faster than a
real lane can change width - e.g. 150 -> 217 -> 243 -> 150 -> 133 -> 392 ->
284 across a handful of frames of overlay logs during a maneuver, most
visibly right as the car turned hard. Both `_lane_bounds` (in-our-lane test)
and `_other_lane_center` (the avoid maneuver's steering target) derive
directly from this value, so both consumed the noise:

1. **The avoid maneuver overshot hard enough to leave the track entirely** -
   steering held near +-0.8 for many consecutive frames while
   `_other_lane_center` (derived from `yellow_x` and the noisy width) jumped
   around underneath the PID, well past `AVOID_TARGET_THRESHOLD`, with
   nothing to stop a single frame's jump from driving the P term straight to
   full lock - unlike `LaneFollower`, which already has two defenses against
   exactly this (see `lane_follower.py`'s module docstring: soft-saturating
   `tanh` error and `position_rate_limit_px`).
2. **A cone that was not actually on the track (e.g. past the white edge, in
   the planter/breezeway) still counted as "in our lane"** whenever the
   width estimate happened to be spiking wide at that moment, extrapolating
   `_lane_bounds`' far edge well past the real lane.

Fixed in `ObstacleAvoider`, without touching `LaneFollower`:

- `_smoothed_lane_width` - an EMA (`CONE_LANE_WIDTH_SMOOTHING_ALPHA`, default
  0.1) over a clamped (`CONE_LANE_WIDTH_MIN_PX`/`CONE_LANE_WIDTH_MAX_PX`,
  default 80/300) copy of `lane/width_px`, kept entirely inside this file.
  Computed every `run()` call (not just while avoiding), so it's already
  settled by the time a maneuver starts, and used in place of the raw value
  everywhere `lane_width_px` is consumed (`_lane_bounds`, `_avoid_step`).
- `_avoid_step` now mirrors `LaneFollower.run()`'s own two defenses,
  reusing the same technique and (by default) the same constants: the
  target position (`other_center`) is rate-limited per frame
  (`AVOID_POSITION_RATE_LIMIT_PX`, falls back to `LANE_POSITION_RATE_LIMIT_PX`,
  then 25px), and the remaining error is soft-saturated via `k*tanh(error/k)`
  (`AVOID_ERROR_SATURATION_PX`, falls back to `LANE_ERROR_SATURATION_PX`,
  then 80px) before reaching `self.avoid_pid`. The rate-limiter's anchor
  resets after a sustained lane-geometry loss (>15 frames, same threshold
  `LaneFollower` uses), so reacquiring snaps fresh instead of slewing from a
  stale position.

Covered by `donkeycar/tests/test_obstacle_avoider.py`:
`test_avoid_target_position_is_rate_limited` and
`test_single_wide_reading_does_not_widen_lane_bounds_immediately`. Like
every other constant in this file, the smoothing alpha/clamp bounds and the
reused rate-limit/saturation defaults are starting estimates copied from
`lane_follower.py`'s already-tuned values, not yet independently verified
against real on-car footage of the avoid maneuver itself - next on-car test
should watch `lane/width_px` and `avoid_steering` directly to confirm the
overshoot is actually gone, not just less noisy in isolation.

## On-car feedback (2026-07-27): still false-triggering in the other lane + turning too sharply

Further on-car testing (now against `lane_follower3.py`'s `LaneFollower`,
which publishes the same `yellow_x`/`white_x`/`lane_width_px` contract as
`lane_follower.py`'s - see its own module docstring) showed two more
problems with the maneuver, both consistent with what was already flagged
as unverified in the previous entry:

1. **A cone in the OTHER lane still latched `cone_detected`.** The
   smoothing added in the previous entry (`_smoothed_lane_width`) narrows
   the window where `lane/width_px` noise fools `_lane_bounds`, but doesn't
   close it - a single bad `yellow_x`/`white_x`/`lane_width_px` combination
   at just the wrong moment can still put a far-lane cone inside the
   (noise-derived) "our lane" bounds.
2. **The avoid maneuver turned too sharply into the other lane**, leaving
   too few frames for the lane tracker to pick up the new lane's own white
   boundary before the turn was already mostly complete - the maneuver
   steers independently of `LaneFollower`'s own detection the whole time
   it's active (see `_avoid_step`), so a fast turn-in outruns the tracker's
   ability to reacquire on the new lane.

Fixed in `ObstacleAvoider`, without touching `LaneFollower`/`lane_follower3.py`:

- **Two new gates on the trigger itself, independent of lane geometry
  entirely** (`cone_ready` in `run()`, in addition to - not instead of -
  the existing `cone_in_our_lane` lane-bounds test): `_is_centered` requires
  the raw detection to sit within `CONE_CENTER_MARGIN_PX` of the *raw
  image's* horizontal center (not our lane's center - a forward-facing
  camera already reads our own lane as roughly centered in frame, while the
  other lane sits well off to one side, so this holds even when
  `_lane_bounds` itself is fooled by noisy geometry); `_is_close` requires
  the winning blob's pixel area to clear `CONE_CLOSE_MIN_AREA_PX` - a real
  3D cone's apparent size grows as it nears the camera (the same reasoning
  `CONE_MAX_WIDTH_PX`'s docstring already relies on), so area is a cheap
  proxy for distance without a second, nearer scan row. All three
  (in-lane, centered, close) must hold for a frame to count toward the
  `CONE_TRIGGER_FRAMES` debounce.
- **`AVOID_STEERING_RATE_LIMIT`** (default 0.04/frame) caps how much
  `self.avoid_steering` may change per frame in `_avoid_step`, independent
  of `avoid_position_rate_limit_px` (which only slews the PID's *target*,
  not its output) - a deliberately slow turn-in so the car spends more
  frames still looking roughly down the lane while the new lane's white
  boundary comes into view. While the limiter is actively capping the turn,
  `avoid_throttle` also stays in the "turning hard" slow-down branch even
  if the remaining pixel error is already under `AVOID_TARGET_THRESHOLD` -
  the correction is still underway, just spread over more frames.

Covered by `donkeycar/tests/test_obstacle_avoider.py`:
`TestObstacleAvoiderCenteredAndClose` (in-lane-but-off-center and
in-lane-but-far cases, reproducing the false-trigger directly) and
`TestObstacleAvoiderSteeringRateLimit`. Like every other constant in this
file, `CONE_CENTER_MARGIN_PX`/`CONE_CLOSE_MIN_AREA_PX`/
`AVOID_STEERING_RATE_LIMIT` are starting estimates, not yet independently
verified against real on-car footage of the revised maneuver - next on-car
test should confirm the other-lane false trigger is actually gone and watch
how many frames it now takes to reacquire the new lane's white line during
the slower turn-in.

## On-car run, `tub_7_26-07-27`: false single-line lock in `lane_follower3.py` + a lane-loss deadlock in `ObstacleAvoider`

First drive with the centered/close gates and steering rate limit above
deployed. `manage2.py`'s `TubWriter` at the time only recorded
`cam/image_array`/`steering`/`throttle` (see below), so `lane/yellow_x`,
`lane/white_x`, `lane/width_px`, and `obstacle/cone_detected` weren't in the
tub - the investigation below replayed the recorded frames through the real
`_adaptive_lab_mask`/`_select_line_blob` code directly (not a guess) to
reconstruct what happened, then cross-checked the reconstruction's steering
curve shape against the tub's real recorded `steering` column.

**What happened, from the frames:** steering pinned to -0.79 within 1.5s of
starting, held there for 150 straight frames; the camera ends up facing a
courtyard seating area, then spends ~26s pressed against a cone rather than
driving past it, and the recording ends jammed against a second cone at
full throttle with no steering correction in the final approach.

**Root cause 1 (the swerve itself):** replaying the real white-tracker code
against the recorded frames found a false single-line "white" lock -
`_adaptive_lab_mask` (correctly) found a bright, low-saturation blob and it
(correctly, per its own shape filter) passed `_select_line_blob` - but the
blob was a sunlit building wall next to the start-line cones, not the
track's painted line. With `yellow_x` never found in this same stretch,
there was no second line to bound the pick, and - critically - nothing
existed at the time to require corroboration before trusting a single-line
extrapolation. Every defense already in `LaneFollower` (`position_rate_limit_px`,
the boundary-identity guards, `LANE_CENTER_MARGIN_PX`) only checks a NEW
detection against an ALREADY-ESTABLISHED `self._last_position` - on the
very first lock of a drive there isn't one yet, so the false lock was
trusted with zero corroboration and immediately fed the steering PID.
Fixed in `lane_follower3.py` (module docstring changelog entry 9): a row's
center only counts toward the first-ever `self._last_position` if it came
from BOTH lines together; single-line extrapolation is only trusted once a
real two-line sighting has established an anchor to rate-limit/sanity-check
against. Costs nothing once driving is underway - the car just coasts
through the existing full-loss handling until a genuine two-line
confirmation arrives. Covered by `donkeycar/tests/test_lane_follower3.py`.

**Root cause 2 (the second cone, no swerve at all):** by the time the car
reached the second cone it had likely never recovered real lane geometry
after the first incident. `ObstacleAvoider.cone_ready` required
`cone_in_our_lane` (from `_lane_bounds`) AND centered AND close - but
`cone_in_our_lane` is structurally `False` whenever `yellow_x`/`white_x` are
both `None` (`_x_in_bounds` returns `False` against `(None, None)` bounds),
so a car that has already lost the lane could never trigger avoidance no
matter how obviously a cone sat dead ahead and centered. Fixed: when
`lane_geometry_available` is `False` this frame, `cone_ready` falls back to
centered AND close alone (see `ObstacleAvoider.run()`) - the in-lane test
was the one gate that depended on lane geometry in the first place; the
other two were already designed to hold even when that geometry is wrong,
so relying on them alone when there's no geometry at all to test against is
strictly better than never triggering. Covered by
`TestObstacleAvoiderNoLaneGeometryFallback` in `test_obstacle_avoider.py`.

**Also fixed:** `manage2.py` (this car's actual `drive()` script, kept
alongside this repo) had drifted from the canonical
`donkeycar/templates/cv_control.py` template - it was missing
`lane/yellow_x`/`lane/white_x`/`lane/width_px` from `TubWriter`'s recorded
fields (exactly what made this investigation have to replay raw frames
instead of just reading the tub) and `VERBOSE_CAR`'s per-part timing table
on `V.start()`. Both re-synced to match `cv_control.py`.

**Not yet verified:** neither fix has been checked against a real on-car
drive yet - `lane_follower3.py`'s cold-start guard and `ObstacleAvoider`'s
lane-loss fallback are both reasoned from a direct replay of `tub_7`'s
frames through the real detection code, not from a fresh drive. Next
on-car test should confirm the car no longer locks onto background objects
at the start of a drive, and separately, still needs the oncoming-car
detector (see "Next steps" below) since that's a second, distinct source
of the same "already lost the lane" problem.

## On-car feedback (2026-07-29): drove straight into a close, centered cone - CONE_MAX_WIDTH_PX again

User report: cone was detected fine at a distance, but "when it's in the
center and super close then it went straight to the orange cone and didn't
avoid it." Reproduced directly (not guessed): `detect_cone()` against a
synthetic frame with a centered orange blob returns a real `x`/`area` up to
400px wide, then `None`/`0` for every width from 401px up to a full
426px-wide (`IMAGE_W`) frame-filling blob - `CONE_MAX_WIDTH_PX`'s shape
filter (`_select_line_blob`) was rejecting the cone's own blob outright
once it grew past the cap.

This is the exact same root cause as the tub_41_26-07-24 incident already
documented above, which raised the cap from 250 to 400 - and that fix
wasn't wrong in kind, just not generous enough: a real cone's own blob
legitimately grows to fill most or all of the frame width at close range
(this camera's tub_7 footage showed genuinely edge-to-edge orange frames -
see the tub_7 investigation above), not just the ~295px observed on
tub_41. There is no width cap a real close cone can't eventually exceed,
so "raise it again" would just set up a third occurrence at some larger
distance-dependent width. Fixed by making `CONE_MAX_WIDTH_PX` effectively
unbounded (100000) by default in both `cfg_cv_control.py` and
`ObstacleAvoider.__init__`'s fallback - see the comments there for why a
width cap doesn't transfer from the line-tracker shape filter it was
ported from (a wide sunlit patch is false-positive risk for a *line*; a
wide *orange* blob on this track is stronger cone evidence, not weaker).
Kept as a config knob, not removed outright, in case some other
track/marker needs one.

Covered by `TestObstacleAvoiderCloseConeWidth` in
`test_obstacle_avoider.py`: sweeps blob widths 300px through a full
426px-wide frame-filling blob and asserts every one still detects and
still latches `cone_detected`/`avoiding` end-to-end, not just passes the
shape filter in isolation.

## Next steps

1. Detect the car's black wheels/front (Decision 2, option A) the same way
   the cone's tape is detected now: a second color-keyed detector, its own
   scan row (Decision 4), reporting `obstacle/car_in_our_lane` — still
   detection-only at first, verified the same way before wiring in control.
2. Decide how the car detector interacts with the cone maneuver once it
   exists: since the cone maneuver (Decision 1.5) no longer releases back to
   `LaneFollower`, a car crossing into our (post-swerve) lane needs its own
   trigger logic layered on top rather than the "one maneuver, first wins"
   model Decision 5 originally assumed for two *returning* maneuvers.
3. Verify the avoidance maneuver (Phase 2, implemented) and every guessed
   constant (`AVOID_PID_P/I/D`, still just copied from the main steering
   PID) against real on-car footage - so far only checked against synthetic
   images in `donkeycar/tests/test_obstacle_avoider.py`, not a real cone/lane.
   Update this document with what changed and why (same as `CLAUDE.md`'s
   standing instruction to treat every CV/PID constant here as provisional
   until checked against hardware).
