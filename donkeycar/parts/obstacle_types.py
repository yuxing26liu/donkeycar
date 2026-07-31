"""
obstacle_types.py

Small, dependency-free data types shared between cone_detector.py and
obstacle_planner.py (cone-dodger3000 branch). Kept separate from both so
the vocabulary (what a "detection" or "lane corridor" even means) can be
read/reviewed in one place, and so cone_detector.py and obstacle_planner.py
don't need to import each other.

Structurally similar in spirit to origin/marcus-object-detection's
obstacle_types.py (typed dataclasses shared across a detector/planner/
arbiter split rather than one monolithic class like origin/blue-tape-detect's
ObstacleAvoider) -- but written fresh, not copied, per the project decision
to rebuild this component rather than adopt either teammate branch's code
wholesale. See CLAUDE.md and this project's own architecture-planning
notes for why.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


@dataclass(frozen=True)
class BBox:
    """Pixel-space bounding box, image coordinates (x right, y down)."""
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def aspect_ratio(self) -> float:
        """height / width -- a cone is taller than it is wide, even when
        cropped by the bottom of the frame at close range (confirmed
        against real cone_static_right/tub_30 frames: aspect ratio ranged
        ~1.2-2.3 across the whole observed distance range, never < 1)."""
        return self.h / float(self.w) if self.w else 0.0

    def overlaps_x(self, lo: float, hi: float) -> bool:
        """Does this box's horizontal extent overlap the closed interval
        [lo, hi] at all (partial overlap counts)?"""
        return self.x <= hi and self.x2 >= lo


@dataclass
class Detection:
    """One frame's cone detection, or None if nothing passed the filters."""
    bbox: BBox
    area_px: int
    distance_mm: Optional[float]
    distance_valid: bool
    # 'depth' if distance_mm came from the OAK-D depth ROI percentile,
    # 'bbox_height' if it fell back to the (uncalibrated -- see
    # cone_detector.py) bounding-box-height ordinal proxy, meaning
    # distance_mm is None and callers must use bbox.h directly as a
    # closer-is-bigger signal, not a real distance.
    distance_source: str


@dataclass
class LaneGeometry:
    """
    A single frame's lane geometry, as published by LaneFollower
    (lane/yellow_x, lane/white_x, lane/width_px) -- i.e. the PRIMARY scan
    row's values only, not the multi-row-corrected internal position.
    This is a known, carried-over limitation (see this project's own
    architecture-planning notes on the row-mismatch bug found in
    origin/blue-tape-detect): a cone detected higher up the frame (farther
    away, smaller y) than LaneFollower's primary scan row sits at a
    perspective distance where the true lane width differs from
    width_px. corridor_row_scale_per_px below is a configurable hook for
    correcting this, defaulted to 0.0 (no correction) because no
    per-row LaneFollower width data was available to calibrate it
    honestly in this pass -- see obstacle_planner.py's module docstring.
    """
    yellow_x: Optional[float]
    white_x: Optional[float]
    width_px: float
    white_right_of_yellow: bool
    primary_row_y: float
    corridor_row_scale_per_px: float = 0.0
    # Frames since the primary-row yellow tracker last had a genuine
    # (not carried-forward/stale) detection -- 0 means detected this
    # exact frame. None means unknown (caller didn't/couldn't supply it,
    # e.g. offline replay reconstructing geometry without direct tracker
    # access). Read directly from LaneFollower.yellow_trackers[0].
    # lost_frames by pilot_arbiter.py -- lane/yellow_x itself is
    # last-known-good telemetry that never resets to None on a miss, so
    # it can't be used on its own to tell fresh tracking from a stale
    # carried-forward value (see obstacle_planner.py's acquisition logic).
    yellow_lost_frames: Optional[int] = None

    @property
    def valid(self) -> bool:
        return self.yellow_x is not None or self.white_x is not None

    def corridor(self, at_row_y: Optional[float] = None) -> Optional[Tuple[float, float]]:
        """
        Return (lo, hi) pixel bounds of the CURRENT lane, or None if no
        lane geometry is available at all this frame.

        If both lines are visible, the corridor is directly the span
        between them (this is exactly the current lane -- no "other lane"
        math needed here, unlike a lane-CHANGE target computation).
        If only one line is visible, the far edge is estimated the same
        way LaneFollower._lane_center falls back internally: the visible
        line +/- width_px/2, signed by which side white is expected on.

        at_row_y, if given, applies corridor_row_scale_per_px * (row_y -
        primary_row_y) as a width correction -- a no-op at the default
        scale of 0.0. See the class docstring for why this isn't
        calibrated yet.
        """
        if self.yellow_x is not None and self.white_x is not None:
            lo, hi = sorted((self.yellow_x, self.white_x))
        else:
            sign = 1.0 if self.white_right_of_yellow else -1.0
            half = self.width_px / 2.0
            if self.yellow_x is not None:
                other = self.yellow_x + sign * self.width_px
                lo, hi = sorted((self.yellow_x, other))
            elif self.white_x is not None:
                other = self.white_x - sign * self.width_px
                lo, hi = sorted((self.white_x, other))
            else:
                return None
        if at_row_y is not None and self.corridor_row_scale_per_px:
            row_delta = at_row_y - self.primary_row_y
            width = hi - lo
            correction = self.corridor_row_scale_per_px * row_delta * width
            lo -= correction / 2.0
            hi += correction / 2.0
        return (lo, hi)


class PlannerState(Enum):
    FOLLOW_LANE = 'FOLLOW_LANE'
    OBJECT_WATCH = 'OBJECT_WATCH'
    CONFIRMED_IN_PATH = 'CONFIRMED_IN_PATH'
    PREPARE_SLOW = 'PREPARE_SLOW'
    SWITCH_TO_NEIGHBOR_LANE = 'SWITCH_TO_NEIGHBOR_LANE'
    HOLD_UNTIL_CLEAR = 'HOLD_UNTIL_CLEAR'
    RETURN_TO_ORIGINAL_LANE = 'RETURN_TO_ORIGINAL_LANE'
    SAFE_STOP = 'SAFE_STOP'


class RolloutMode(Enum):
    DISABLED = 'disabled'   # planner/detector not run at all
    OBSERVE = 'observe'     # run + log/publish to Memory, never touches driving
    SHADOW = 'shadow'       # observe, plus compute+log what active mode WOULD do
    ACTIVE = 'active'       # actually calls LaneFollower.set_lane()


@dataclass
class PlannerDecision:
    """One frame's planner output -- what obstacle_planner.py decided, and
    (in ACTIVE mode) what pilot_arbiter-equivalent code should do about it.
    Always populated regardless of rollout mode, so 'shadow' mode can log
    it without it ever being applied."""
    state: PlannerState
    reason: str
    requested_lane: Optional[str]     # 'left' | 'right' | None (no change requested)
    throttle_scale: float             # multiply LaneFollower's throttle by this
    cone_in_path: bool
    # Added 2026-07-30 after cone_test5 (real active-mode lane change that
    # overshot the yellow boundary and ran off track): during
    # SWITCH_TO_NEIGHBOR_LANE/RETURN_TO_ORIGINAL_LANE, LaneFollower's own
    # raw steering (still correct in DIRECTION, since it's the same
    # hardened PID, just now targeting the flipped lane) pinned near +/-1.0
    # for 30+ frames because the tanh saturation treats "target is now on
    # the other side of a whole lane width away" the same as any other
    # large error. steering_cap bounds that raw output's MAGNITUDE (not
    # direction) during the crossing, tapering down as the car actually
    # approaches the yellow boundary (see ObstaclePlanner._crossing_
    # progress) - None means "don't cap, use LaneFollower's steering as
    # normal" (every other state, including once the crossing is
    # confirmed). Applied only in ACTIVE mode by pilot_arbiter.py.
    steering_cap: Optional[float] = None
