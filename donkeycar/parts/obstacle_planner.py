"""
obstacle_planner.py

Cone-avoidance state machine for the cone-dodger3000 branch. Decides
whether/when to switch LaneFollower to the neighboring lane and back --
it never drives directly. All durations below are FRAME COUNTS at the
car's fixed DRIVE_LOOP_HZ, not wall-clock seconds/timers: this project has
already been burned once by wall-clock-based test harnesses producing
non-reproducible results (see this project's own testing notes on PID
timing), and frame counts are exactly what LaneFollower itself already
uses everywhere (REACQUIRE_AFTER_FRAMES, MAX_LOST_FRAMES, CONE_TRIGGER_
FRAMES-style debounce) -- this file follows that existing convention
rather than introducing a second one.

Architecture recap (see this project's cone-avoidance architecture notes
for the full comparison against origin/blue-tape-detect and
origin/marcus-object-detection):

  - The maneuver itself is executed by literally retargeting
    LaneFollower at the other lane via its set_lane() method (added this
    branch) -- reusing its own hardened PID/detection/rate-limiting,
    rather than a second bespoke controller like both teammate branches
    built. This planner only decides WHEN to call set_lane(), never how
    to steer.
  - Every state transition that matters for safety is gated on a
    perception signal (cone relevance/distance, lane-geometry
    acquisition) confirmed over several consecutive frames, not a fixed
    steering duration -- this is the single biggest fix over
    origin/marcus-object-detection's scripted MOVE_AROUND_OBJECT/
    PASS_OBJECT/STEER_BACK phases, which held a fixed steering value for
    a fixed number of seconds regardless of the car's actual speed or
    the cone's actual position, and were explicitly the mechanism behind
    several of the "turns too wide"/"drifts"/"doesn't generalize across
    speed" failures already diagnosed for this project.
  - Minimum per-state dwell counts exist ONLY as hysteresis (to stop a
    single noisy frame from flapping the state), never as the actual
    trigger condition -- every transition still additionally requires
    its perception gate.

KNOWN LIMITATION, stated plainly rather than worked around: the
"in-path" corridor test below uses LaneFollower's PRIMARY scan row
values (lane/yellow_x, lane/white_x, lane/width_px), the same row-mismatch
limitation already diagnosed in this project's review of
origin/blue-tape-detect (a cone detected at a different image row than
the lane geometry was measured at is being compared against a
perspective-inconsistent width). LaneGeometry.corridor_row_scale_per_px
(obstacle_types.py) is a configurable correction hook, defaulted to 0.0
(no correction), because per-row LaneFollower width data was not
available to calibrate it honestly during this pass -- see this
project's implementation notes for what real data would be needed to
do this properly (recording LANE_SCAN_ROWS' individual widths, not just
the primary row).

ALSO STATED PLAINLY: the distance thresholds below default to
reasonable, documented starting points, not values validated against a
real driving approach with real depth data -- none of the five tubs
recorded for this branch include depth (see cone_detector.py's module
docstring). Where a real driving approach WAS available
(cone_approach_right, bbox-height only, no depth), the bbox-height
thresholds were set from that real data and are noted as such; the mm
thresholds are carried over from the architecture-planning discussion's
order-of-magnitude reasoning and are explicitly flagged as unverified.
"""
import logging
from collections import deque

from donkeycar.parts.obstacle_types import PlannerState, RolloutMode, PlannerDecision

logger = logging.getLogger(__name__)


class ObstaclePlanner:
    def __init__(self, cfg):
        mode = getattr(cfg, 'OBSTACLE_AVOIDANCE_MODE', 'disabled')
        self.mode = RolloutMode(mode)

        self.detect_confirm_frames = getattr(cfg, 'CONE_DETECT_CONFIRM_FRAMES', 3)
        self.clear_confirm_frames = getattr(cfg, 'CONE_CLEAR_CONFIRM_FRAMES', 5)
        self.lane_acquire_confirm_frames = getattr(cfg, 'CONE_LANE_ACQUIRE_CONFIRM_FRAMES', 6)

        # mm thresholds: architecture-level starting points, NOT verified
        # against real depth+approach data (see module docstring).
        self.watch_distance_mm = getattr(cfg, 'CONE_WATCH_DISTANCE_MM', 3000)
        self.commit_distance_mm = getattr(cfg, 'CONE_COMMIT_DISTANCE_MM', 1200)
        self.emergency_distance_mm = getattr(cfg, 'CONE_EMERGENCY_DISTANCE_MM', 400)
        # Emergency requires this many CONSECUTIVE qualifying frames before
        # triggering SAFE_STOP, UNLESS distance/size is at or past the
        # tighter "critical" threshold below, which still triggers
        # instantly - added after avoid_cone's frame-162 anomaly (an
        # unexplained single-frame emergency reading right after
        # completing a maneuver); not reproduced in cone_approach_right2/
        # cone_negative2, so this is a general protective measure rather
        # than a fix for a root-caused bug.
        self.emergency_confirm_frames = getattr(cfg, 'CONE_EMERGENCY_CONFIRM_FRAMES', 2)
        self.critical_distance_mm = getattr(cfg, 'CONE_CRITICAL_DISTANCE_MM', 200)
        self.critical_bbox_height_px = getattr(cfg, 'CONE_CRITICAL_BBOX_HEIGHT_PX', 210)

        # bbox-height (px) fallback thresholds. Checked directly against
        # cone_approach_right's real, monotonic bbox-height progression
        # (a genuine driving approach, no depth recorded): h rose ~29px
        # (idx0, far) -> ~225px (idx245, the point the human stopped
        # manually to avoid collision). watch=45 sits around idx120 (~half
        # the approach -- comfortable lead time to start monitoring);
        # commit=90 sits around idx180-185 (~45-65 frames / 2.25-3.25s of
        # runway left to execute a switch before the human's stop point);
        # emergency=170 sits around idx225-230 (close to where the human
        # actually stopped -- should rarely be reached if commit already
        # triggered with time to spare). This is real-data-grounded but
        # from ONE approach at ONE speed -- not yet confirmed to
        # generalize across speed/distance/cone placement, which is
        # exactly the failure mode this project's cone-avoidance
        # architecture review flagged in both teammate branches' scripted
        # timings. Re-check against additional approach recordings before
        # trusting these for an active-mode test.
        self.watch_bbox_height_px = getattr(cfg, 'CONE_WATCH_BBOX_HEIGHT_PX', 45)
        self.commit_bbox_height_px = getattr(cfg, 'CONE_COMMIT_BBOX_HEIGHT_PX', 90)
        self.emergency_bbox_height_px = getattr(cfg, 'CONE_EMERGENCY_BBOX_HEIGHT_PX', 170)

        # NEGATIVE narrows the corridor inward (the car's real path is
        # narrower than the full lane), POSITIVE widens it outward.
        # Modest default inward narrowing -- generically justified, but
        # explicitly NOT validated as sufficient: replaying cone_negative
        # (a real recording where the cone should never be judged
        # blocking) still produced a false in-path commit at this and
        # even much larger (-60, -90px) narrowing amounts. Root-caused to
        # LaneFollower's own yellow tracker false-locking onto a small
        # bright-sunlight background object in that recording (a known
        # category of LaneFollower limitation, not a new bug in this
        # file) -- see this project's replay findings. No margin value
        # fixes a corrupted upstream corridor; this is a real, open,
        # unresolved finding, not tuned away by this default.
        self.corridor_margin_px = getattr(cfg, 'CORRIDOR_SAFETY_MARGIN_PX', -20)
        self.center_fallback_margin_px = getattr(cfg, 'CONE_CENTER_FALLBACK_MARGIN_PX', 70)
        self.image_center_px = getattr(cfg, 'IMAGE_W', 426) / 2.0

        # Lateral-sweep-past veto, added 2026-07-30 after replaying real
        # on-car shadow-mode tubs (cone_approach_right2, cone_negative2):
        # a cone truly in our path stays laterally near-centered as
        # distance closes (cone_approach_right2: bbox center held in a
        # ~190-231px band, max 10-frame drift 18px once bbox_h>=45);
        # a cone in the neighboring lane that the car simply drives past
        # sweeps steadily toward one edge of the frame as it closes
        # (cone_negative2: bbox center drifted 143px->31px, a clean
        # monotonic sweep, >=25px net drift over any 10-frame window by
        # idx152 - 4 frames before that recording's false CONFIRMED_IN_
        # PATH commit at idx156). This is a real, physical distinction
        # (parallax of a passing object vs. an object dead ahead), not a
        # tuned-away symptom - gated by a minimum bbox height so it only
        # applies once the blob is large enough that its centroid isn't
        # dominated by detection noise (checked: false-triggered at up to
        # 83px drift on far/tiny <45px-tall blobs in cone_approach_right2
        # before this gate was added).
        self.lateral_history_frames = getattr(cfg, 'CONE_LATERAL_HISTORY_FRAMES', 10)
        self.lateral_sweep_reject_px = getattr(cfg, 'CONE_LATERAL_SWEEP_REJECT_PX', 25)
        self.lateral_min_height_px = getattr(cfg, 'CONE_LATERAL_MIN_HEIGHT_PX', 45)
        self._cx_history = deque(maxlen=self.lateral_history_frames)

        # Corridor plausibility, requested as defense-in-depth alongside
        # the lateral-sweep fix above - NOTE this did NOT catch either
        # confirmed real false-positive on its own (the first cone_
        # negative's corrupted yellow-lock was smooth/self-consistent,
        # not a jump; cone_negative2's width 277-289px was within this
        # band) - it's an additional net, not a replacement for the fix
        # that's actually proven against real data.
        self.corridor_width_min_px = getattr(cfg, 'CORRIDOR_WIDTH_MIN_PX', 90)
        self.corridor_width_max_px = getattr(cfg, 'CORRIDOR_WIDTH_MAX_PX', 340)

        # Require a real depth reading (not the uncalibrated bbox-height
        # fallback) before ever committing to a lane switch - one of
        # today's requested conservative defaults for the first active
        # test. Both real approach tubs (cone_approach_right2,
        # cone_negative2) had valid depth throughout their commit-
        # relevant windows, so this doesn't block the legitimate case.
        self.require_valid_depth_to_switch = getattr(cfg, 'CONE_REQUIRE_VALID_DEPTH_TO_SWITCH', True)

        self.prepare_min_frames = getattr(cfg, 'PLANNER_PREPARE_MIN_FRAMES', 4)
        self.hold_min_frames = getattr(cfg, 'PLANNER_HOLD_MIN_FRAMES', 10)
        self.maneuver_timeout_frames = getattr(cfg, 'PLANNER_MANEUVER_TIMEOUT_FRAMES', 300)
        self.safe_stop_recover_frames = getattr(cfg, 'PLANNER_SAFE_STOP_RECOVER_FRAMES', 40)
        self.safe_stop_auto_recover = getattr(cfg, 'PLANNER_SAFE_STOP_AUTO_RECOVER', True)

        self.prepare_throttle_scale = getattr(cfg, 'PLANNER_PREPARE_THROTTLE_SCALE', 0.6)
        self.switch_throttle_scale = getattr(cfg, 'PLANNER_SWITCH_THROTTLE_SCALE', 0.5)
        self.hold_throttle_scale = getattr(cfg, 'PLANNER_HOLD_THROTTLE_SCALE', 0.6)
        self.return_throttle_scale = getattr(cfg, 'PLANNER_RETURN_THROTTLE_SCALE', 0.8)

        self.state = PlannerState.FOLLOW_LANE
        self._frames_in_state = 0
        self._watch_count = 0
        self._clear_count = 0
        self._acquire_count = 0
        self._safe_recover_count = 0
        self._emergency_count = 0
        self._maneuver_side = None      # 'left' | 'right' -- the neighbor lane this maneuver targets
        self._original_side = None      # lane to return to
        self._closest_seen = None       # tracks whether the cone has started retreating in HOLD

    # ---- perception helpers ----------------------------------------

    def _update_lateral_history(self, detection):
        """Call exactly once per tick (from step(), before any
        _is_relevant() calls) - maintains the rolling bbox-center history
        the lateral-sweep veto below reads. Cleared on a miss so a
        different object appearing later doesn't inherit stale history."""
        if detection is None:
            self._cx_history.clear()
            return
        self._cx_history.append(detection.bbox.cx)

    def _is_sweeping_past(self, detection):
        """True if the cone's lateral position is drifting steadily
        toward one edge of the frame as it closes - the signature of an
        object in the NEIGHBOR lane being driven past, not one actually
        in our path (see __init__ comment for the real-data numbers this
        was calibrated against). Only evaluated once the blob is large
        enough (lateral_min_height_px) that its centroid isn't dominated
        by detection noise."""
        if detection.bbox.h < self.lateral_min_height_px:
            return False
        if len(self._cx_history) < self.lateral_history_frames:
            return False
        drift = self._cx_history[-1] - self._cx_history[0]
        return abs(drift) >= self.lateral_sweep_reject_px

    def _corridor_plausible(self, geometry):
        """Cheap defense-in-depth sanity check on the corridor itself
        (order + width band) - requested alongside the lateral-sweep fix.
        NOTE: neither confirmed real false-positive in this project was
        actually caught by this check (both had a self-consistent,
        in-band corridor) - this is an extra net, not the proven fix."""
        corridor = geometry.corridor()
        if corridor is None:
            return False
        lo, hi = corridor
        width = hi - lo
        return lo < hi and self.corridor_width_min_px <= width <= self.corridor_width_max_px

    def _is_relevant(self, detection, geometry):
        """Is this detection actually in our driving path, vs. merely
        visible? Corridor-overlap when lane geometry is available (and
        plausible, and the cone isn't just sweeping past - see the two
        helpers above); image-center fallback (same idea as
        origin/blue-tape-detect's lane_geometry_available branch, reused
        as a concept) when lane geometry isn't available at all."""
        if detection is None:
            return False
        if self._is_sweeping_past(detection):
            return False
        corridor = geometry.corridor(at_row_y=detection.bbox.cy) if geometry else None
        if corridor is not None:
            if not self._corridor_plausible(geometry):
                return False
            lo, hi = corridor
            return detection.bbox.overlaps_x(lo - self.corridor_margin_px, hi + self.corridor_margin_px)
        # no lane geometry this frame -- conservative image-center fallback
        return abs(detection.bbox.cx - self.image_center_px) <= self.center_fallback_margin_px

    def _distance_at_or_below(self, detection, mm_threshold, px_threshold):
        if detection.distance_valid:
            return detection.distance_mm <= mm_threshold
        return detection.bbox.h >= px_threshold

    def _distance_at_or_above(self, detection, mm_threshold, px_threshold):
        if detection.distance_valid:
            return detection.distance_mm >= mm_threshold
        return detection.bbox.h <= px_threshold

    def _lane_acquired(self, geometry):
        """Stricter than _is_relevant's corridor check on purpose: this
        gates 'is the (possibly just-switched-to) lane genuinely being
        tracked right now', not 'is there SOME lane estimate available'.
        LaneFollower's lane/yellow_x and lane/white_x Memory outputs are
        last-known-good telemetry that does NOT reset to None on a miss
        (by design, for continuity) - found via real active-mode test
        replay (tub_122_26-07-30): after set_lane('left'), white_x
        correctly went None (never re-acquired) but yellow_x stayed
        frozen at one stale pre-switch value for 45+ frames, and the
        single-line corridor fallback happily returned non-None from
        that stale value the whole time - the planner falsely believed
        the neighbor lane was acquired (HOLD_UNTIL_CLEAR) while
        LaneFollower was actually completely blind and decaying to its
        own MAX_LOST_FRAMES stop. Requiring BOTH lines closes this
        specific hole: white_x present or not is NOT sticky the same way
        (it only persists a value it actually had), so this is a real,
        verified fix for the exact failure observed, not a guess."""
        return (geometry is not None
                and geometry.yellow_x is not None
                and geometry.white_x is not None)

    # ---- main step ---------------------------------------------------

    def step(self, detection, geometry):
        """One frame's decision. Always runs the full state machine
        regardless of rollout mode -- callers (cv_control.py) decide
        whether to actually call LaneFollower.set_lane() based on
        self.mode; this keeps 'shadow' mode's log identical to what
        'active' would have done."""
        self._frames_in_state += 1
        self._update_lateral_history(detection)
        transition = self._step_fsm(detection, geometry)
        if transition is not None:
            new_state, reason = transition
            logger.info(f"ObstaclePlanner: {self.state.value} -> {new_state.value} ({reason})")
            self.state = new_state
            self._frames_in_state = 0
            reason_out = reason
        else:
            reason_out = f"holding {self.state.value}"

        cone_in_path = self._is_relevant(detection, geometry)
        requested_lane = None
        throttle_scale = 1.0

        if self.state == PlannerState.FOLLOW_LANE:
            throttle_scale = 1.0
        elif self.state == PlannerState.OBJECT_WATCH:
            throttle_scale = 1.0
        elif self.state == PlannerState.CONFIRMED_IN_PATH:
            throttle_scale = self.prepare_throttle_scale
        elif self.state == PlannerState.PREPARE_SLOW:
            throttle_scale = self.prepare_throttle_scale
        elif self.state == PlannerState.SWITCH_TO_NEIGHBOR_LANE:
            requested_lane = self._maneuver_side
            throttle_scale = self.switch_throttle_scale
        elif self.state == PlannerState.HOLD_UNTIL_CLEAR:
            requested_lane = self._maneuver_side
            throttle_scale = self.hold_throttle_scale
        elif self.state == PlannerState.RETURN_TO_ORIGINAL_LANE:
            requested_lane = self._original_side
            throttle_scale = self.return_throttle_scale
        elif self.state == PlannerState.SAFE_STOP:
            throttle_scale = 0.0

        return PlannerDecision(state=self.state, reason=reason_out, requested_lane=requested_lane,
                                throttle_scale=throttle_scale, cone_in_path=cone_in_path)

    def _step_fsm(self, detection, geometry):
        """Returns (new_state, reason) if a transition should happen this
        frame, else None (stay in self.state)."""
        relevant = self._is_relevant(detection, geometry)

        # Global safety net: an emergency-close relevant cone always wins,
        # regardless of current state, EXCEPT while already executing the
        # maneuver we'd otherwise be trying to (re-)trigger, or while
        # already in SAFE_STOP (a self-transition here would reset
        # _frames_in_state every tick, spam identical log lines, and is
        # meaningless -- SAFE_STOP is already the maximally-safe state;
        # found by this exact symptom replaying cone_static_right, where
        # the cone sits emergency-close for hundreds of consecutive
        # frames).
        emergency_candidate = (relevant and self.state not in (PlannerState.SWITCH_TO_NEIGHBOR_LANE,
                                                                 PlannerState.HOLD_UNTIL_CLEAR,
                                                                 PlannerState.RETURN_TO_ORIGINAL_LANE,
                                                                 PlannerState.SAFE_STOP)
                               and self._distance_at_or_below(detection, self.emergency_distance_mm,
                                                                self.emergency_bbox_height_px))
        if emergency_candidate:
            critical = self._distance_at_or_below(detection, self.critical_distance_mm,
                                                    self.critical_bbox_height_px)
            if critical:
                return PlannerState.SAFE_STOP, "critically-close relevant cone (immediate)"
            self._emergency_count += 1
            if self._emergency_count >= self.emergency_confirm_frames:
                return PlannerState.SAFE_STOP, f"emergency-close relevant cone ({self._emergency_count} frames)"
        else:
            self._emergency_count = 0

        if self.state == PlannerState.FOLLOW_LANE:
            self._watch_count = self._watch_count + 1 if relevant else 0
            if self._watch_count >= self.detect_confirm_frames:
                return PlannerState.OBJECT_WATCH, f"cone relevant for {self._watch_count} frames"
            return None

        if self.state == PlannerState.OBJECT_WATCH:
            if relevant:
                self._clear_count = 0
                within_commit = self._distance_at_or_below(detection, self.commit_distance_mm,
                                                             self.commit_bbox_height_px)
                if within_commit:
                    if self.require_valid_depth_to_switch and not detection.distance_valid:
                        return None  # close enough by bbox-height alone, but no real depth to trust yet
                    return PlannerState.CONFIRMED_IN_PATH, "cone within commit distance"
                return None
            self._clear_count += 1
            if self._clear_count >= self.clear_confirm_frames:
                return PlannerState.FOLLOW_LANE, "cone cleared before commit distance"
            return None

        if self.state == PlannerState.CONFIRMED_IN_PATH:
            return PlannerState.PREPARE_SLOW, "beginning slow-down before maneuver"

        if self.state == PlannerState.PREPARE_SLOW:
            if not relevant:
                self._clear_count += 1
                if self._clear_count >= self.clear_confirm_frames:
                    return PlannerState.FOLLOW_LANE, "cone cleared during prepare"
                # NOTE: deliberately does not fall through to the switch
                # check below on a tick where the cone wasn't relevant --
                # committing to a lane switch must only happen on a tick
                # where the cone is currently confirmed blocking, not
                # merely "prepare_min_frames have elapsed since entry"
                # (found by this file's own test suite: without this,
                # a cone that started clearing mid-PREPARE_SLOW could
                # still trigger SWITCH_TO_NEIGHBOR_LANE on a frame where
                # it had already gone not-relevant, racing the abort path).
                return None
            self._clear_count = 0
            if self._frames_in_state >= self.prepare_min_frames and self._lane_acquired(geometry):
                self._original_side = geometry.white_right_of_yellow and 'right' or 'left'
                self._maneuver_side = 'left' if self._original_side == 'right' else 'right'
                self._acquire_count = 0
                self._closest_seen = None
                return PlannerState.SWITCH_TO_NEIGHBOR_LANE, f"committing to switch to {self._maneuver_side}"
            return None

        if self.state == PlannerState.SWITCH_TO_NEIGHBOR_LANE:
            if self._frames_in_state >= self.maneuver_timeout_frames:
                return PlannerState.SAFE_STOP, "maneuver timeout awaiting neighbor-lane acquisition"
            if self._lane_acquired(geometry):
                self._acquire_count += 1
            else:
                self._acquire_count = 0
            if self._acquire_count >= self.lane_acquire_confirm_frames:
                return PlannerState.HOLD_UNTIL_CLEAR, "neighbor lane geometry acquired"
            return None

        if self.state == PlannerState.HOLD_UNTIL_CLEAR:
            if self._frames_in_state >= self.maneuver_timeout_frames:
                return PlannerState.SAFE_STOP, "maneuver timeout awaiting cone clearance"
            if detection is not None and detection.distance_valid:
                self._closest_seen = detection.distance_mm if self._closest_seen is None \
                    else min(self._closest_seen, detection.distance_mm)
            if self._frames_in_state < self.hold_min_frames:
                return None
            # "passed" evidence: cone no longer detected at all for
            # clear_confirm_frames, OR (if still visible) distance is
            # retreating past its closest recorded point by a real margin.
            if detection is None:
                self._clear_count += 1
            elif detection.distance_valid and self._closest_seen is not None \
                    and detection.distance_mm > self._closest_seen + 200:
                self._clear_count += 1
            elif not relevant:
                self._clear_count += 1
            else:
                self._clear_count = 0
            if self._clear_count >= self.clear_confirm_frames:
                return PlannerState.RETURN_TO_ORIGINAL_LANE, "cone confirmed passed"
            return None

        if self.state == PlannerState.RETURN_TO_ORIGINAL_LANE:
            if self._frames_in_state >= self.maneuver_timeout_frames:
                return PlannerState.SAFE_STOP, "maneuver timeout awaiting original-lane re-acquisition"
            if self._lane_acquired(geometry):
                self._acquire_count += 1
            else:
                self._acquire_count = 0
            if self._acquire_count >= self.lane_acquire_confirm_frames:
                return PlannerState.FOLLOW_LANE, "original lane re-acquired"
            return None

        if self.state == PlannerState.SAFE_STOP:
            if not self.safe_stop_auto_recover:
                return None
            clean = (detection is None or not relevant) and self._lane_acquired(geometry)
            self._safe_recover_count = self._safe_recover_count + 1 if clean else 0
            if self._safe_recover_count >= self.safe_stop_recover_frames:
                return PlannerState.FOLLOW_LANE, "safe-stop auto-recovered: clear + lane reacquired"
            return None

        return None
