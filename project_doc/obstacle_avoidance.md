# Obstacle Avoidance: Plan and Status

Third stage of this project's mission progression (see `CLAUDE.md`): line
following -> lane following (`LaneFollower`, `donkeycar/parts/lane_follower.py`,
done and working) -> **two-way road navigation**, this document. On the
right-lane track (one white outer edge + one dashed yellow centerline, see
the reference photos in `CLAUDE.md`), the car must additionally:

1. Swerve around a **traffic cone** placed in our own lane, then return to
   our lane once past it.
2. Watch for an **oncoming car** in the opposite lane: ignore it while it
   stays in its own lane, swerve away if it crosses into ours.
3. Do both while ignoring background clutter (leaves/debris on the track,
   shadows, glare, the concrete's own expansion-joint seams) and without
   losing lane-keeping.

This is being built incrementally, one detector at a time, verified before
the next piece is layered on. **Only step 1's detector (below) is
implemented so far.**

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
| **A. Detect the blue tape square (chosen)** | Large, flat, high-saturation ground marking — a much more reliable classical-CV target at camera resolution/distance than a small cone silhouette; blue doesn't collide with anything else in play (gray concrete, white line, yellow dashes); doesn't require knowing the cone's own paint color. | It's a proxy for the cone's *position*, not the cone itself — if a cone is ever placed off its tape mark, this misses it. |
| B. Detect the cone's own color | Detects the real obstacle, not a proxy. | Cone's actual color unconfirmed; a small object is a weaker blob than a tape square at the same distance. |
| C. Both (union of masks) | Most robust. | More tuning surface for marginal benefit right now; blue also appears on the (off-track, background) recycling bin in the reference photos. |

Chose **A**. Kept swappable — B or C is a config change (a second HSV
threshold + mask union), not a redesign, if it turns out cones aren't
reliably placed on their tape marks.

### Decision 2 — How to detect the oncoming car (not yet implemented)

| Option | Pros | Cons |
|---|---|---|
| **A. Color-key black (wheels/front) — planned next** | Simple, consistent with the rest of this codebase's color-blob approach; naturally ignores leaves/debris (not black) without extra logic. | Two known false-positive risks on this track: shadows cast on the concrete, and the dark expansion-joint seams visible in the reference photos. Needs shape/size filtering + a short trigger-hold, same as the cone detector. |
| B. Foreign-object-by-elimination (mask out everything already known, largest leftover blob = car) | Doesn't need to know the car's color. | Dropped once the black-wheels/front fact was known — strictly worse than A: any unexpected pixel (leaf, gravel, glare, tape edge) is a false positive by construction, since it's a not-known-stuff detector rather than a positive match. |
| C. A, plus a frame-over-frame blob-growth check | Extra guard against a *static* dark false positive (permanent shadow, the expansion joint) — a real approaching car's blob should grow; a static dark patch shouldn't. | More state, slower to trigger, another blind parameter. Held in reserve — only worth building if A's shape/size filter + trigger-hold isn't enough in on-car testing (YAGNI otherwise). |

Plan: **A**, with **C** as a documented fallback if the seam/shadow proves to
be a recurring false trigger during testing.

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

## What's implemented now: Phase 1, the cone's blue-tape marker

`donkeycar/parts/obstacle_avoider.py`, class `ObstacleAvoider`.

**Detection-only.** `pilot/steering` and `pilot/throttle` pass straight
through unchanged — this part currently *reports* what it sees, it doesn't
drive yet. That's deliberate: it lets the detector be tuned and verified
against real camera footage/tub recordings on the car before any avoidance
maneuver (which depends on decisions 2-5 above) is built on top of it.

### How it works

1. Take a horizontal slice of the raw camera frame (`cam/image_array`, not
   whatever `LaneFollower` drew on `cv/image_array`) at `CONE_SCAN_Y` /
   `CONE_SCAN_HEIGHT` — the same slice-based approach every CV part in this
   codebase uses (`LineFollower`, `LaneFollower`'s per-color trackers, the
   unmerged `ObjectAvoider` prototype on `origin/estella`).
2. Convert to HSV and threshold on `BLUE_HSV_THRESHOLD_LOW/HIGH` to find the
   tape's blob, reusing `_select_line_blob` from `lane_follower.py` (the
   same connected-component shape/size filter the line trackers use) rather
   than writing a second implementation.
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
`HAVE_OBSTACLE_AVOIDANCE` (default `False`):

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

### Configuration

All in `donkeycar/templates/cfg_cv_control.py`, overridable per-car in
`myconfig.py` (which lives outside this repo, at `/home/pi/mycar/myconfig.py`
— see `CLAUDE.md`):

```python
HAVE_OBSTACLE_AVOIDANCE = False   # opt-in; detection-only regardless

CONE_SCAN_Y = 60         # top of the forward scan slice, in pixels
CONE_SCAN_HEIGHT = 30    # height of the scan slice, in pixels

BLUE_HSV_THRESHOLD_LOW = (95, 100, 60)     # guessed, not yet tuned on hardware
BLUE_HSV_THRESHOLD_HIGH = (130, 255, 255)  # -- see "Testing / tuning" below

CONE_MIN_AREA_PX = 80    # smallest pixel area (in the scan slice) counted as the marker
CONE_MAX_WIDTH_PX = 250  # widest pixel width (in the scan slice) counted as the marker

LANE_SHIFT_MARGIN_PX = 10  # margin added to our lane's bounds when testing membership
CONE_TRIGGER_FRAMES = 2    # consecutive in-lane frames required before latching
```

Like every other CV threshold in this codebase (`COLOR_THRESHOLD_LOW/HIGH`,
`YELLOW_HSV_THRESHOLD_LOW/HIGH`, ...), the HSV bounds above are **an
untuned starting guess**, not a measurement — they need to be checked
against real tape under the car's actual lighting.

## Testing performed

`donkeycar/tests/test_obstacle_avoider.py`, 17 tests, all passing (run with
`conda activate donkey && python -m pytest donkeycar/tests/test_obstacle_avoider.py -v`
— this repo's dev tooling lives in the `donkey` conda environment, not the
system Python). All synthetic-image tests (no camera/hardware needed):

- `_lane_bounds` geometry: both lines visible, single-line extrapolation in
  each direction, the mirrored "other lane" calculation, and the
  `white_right_of_yellow=False` (left-lane) sign flip.
- Cone detected in our lane latches `cone_detected` after
  `CONE_TRIGGER_FRAMES` consecutive frames, at approximately the right x.
- Cone detected in the *other* lane is found but correctly not counted as
  "in our lane" (never latches).
- A blue patch outside the scan band, or too small
  (`< CONE_MIN_AREA_PX`), is ignored.
- **Robustness / "ignore the background" checks:** a green (leaf-colored)
  patch in our lane doesn't trigger; a white line + yellow line drawn
  through the scan band don't trigger; a single detected frame followed by
  a miss correctly resets the debounce counter instead of carrying over.
- `cam_img=None` and the overlay path both pass through cleanly without
  altering `steering`/`throttle`.

**Not yet tested/verified:** real camera footage. The HSV thresholds and
scan-row placement are guesses (see above) and need to be checked against
an actual frame of the taped track before relying on this on the car — same
caveat every other CV threshold in this codebase carries.

## Next steps

1. Detect the car's black wheels/front (Decision 2, option A) the same way
   the cone's tape is detected now: a second color-keyed detector, its own
   scan row (Decision 4), reporting `obstacle/car_in_our_lane` — still
   detection-only at first, verified the same way before wiring in control.
2. Build the actual avoidance maneuver: while either "cone in our lane" or
   "car crossing into our lane" is latched, retarget steering toward the
   *other* lane's center (`_lane_bounds(..., other_lane=True)`, already
   implemented) using a dedicated PID (kept separate from `LaneFollower`'s
   own `pid_st` so the two can't corrupt each other's integral state), then
   release back to `LaneFollower`'s normal output once clear. Apply
   Decisions 3 and 5 (passive fallback on lost lane geometry; one
   maneuver at a time) here.
3. Verify all of the above on the car, tune every guessed constant against
   real footage, and update this document with what changed and why (same
   as `CLAUDE.md`'s standing instruction to treat every CV threshold here as
   provisional until checked against hardware).
