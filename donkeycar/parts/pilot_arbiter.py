"""
pilot_arbiter.py

The Vehicle-part glue between LaneFollower, ConeDetector, and
ObstaclePlanner for the cone-dodger3000 branch. Deliberately thin: all
real detection/decision logic lives in cone_detector.py/obstacle_planner.py
(both independently unit-tested and validated offline against real tubs
-- see scripts/replay_cone_planner.py); this class only adapts between
the Vehicle framework's flat Memory-key convention and those classes'
richer objects, and applies the planner's decision according to the
configured rollout mode.

Architecturally simpler than origin/marcus-object-detection's
pilot_arbiter.py by design: because this project's maneuver is executed
by retargeting LaneFollower's OWN steering via set_lane() (see
lane_follower3.py) rather than a second bespoke steering controller,
there is no separate steering value to arbitrate between -- LaneFollower's
pilot/steering output already IS the avoidance-adjusted output once
set_lane() has been called. The only thing this class actually adjusts is
throttle (via the planner's throttle_scale) and, in ACTIVE mode, whether
set_lane() gets called at all.

Named the same as origin/marcus-object-detection's file because it plays
the same conceptual role (the arbitration layer between lane-following
and obstacle avoidance) -- written fresh for this project's simpler
reuse-the-existing-PID architecture, not copied from that branch.
"""
import logging

from donkeycar.parts.obstacle_types import LaneGeometry, RolloutMode

logger = logging.getLogger(__name__)


class PilotArbiter:
    def __init__(self, cfg, lane_follower, cone_detector, obstacle_planner):
        self.lane_follower = lane_follower
        self.cone_detector = cone_detector
        self.planner = obstacle_planner
        self.mode = self.planner.mode
        self.primary_row_y = lane_follower.scan_rows[0]['scan_y']
        # Minimum absolute throttle during an active maneuver, added
        # 2026-07-30 after cone_test4: LaneFollower's OWN confidence-based
        # speed policy already drops to THROTTLE_MIN as soon as detection
        # confidence falls (which naturally happens right after a lane
        # switch, while re-acquiring) - the planner's throttle_scale
        # (e.g. 0.5 during SWITCH_TO_NEIGHBOR_LANE) then multiplies THAT
        # already-reduced value, not the nominal cruising throttle.
        # Confirmed directly: cone_test4 recorded throttle=0.075 for the
        # entire 300-frame maneuver (exactly THROTTLE_MIN=0.15 * 0.5) and
        # the cone's distance never changed the whole time - the car
        # wasn't slow, it was too close to stalled to make real progress,
        # so the maneuver just sat there until the 15s timeout gave up.
        # This floor stops the two independent slowdowns from compounding
        # below a speed that can actually complete a maneuver - it does
        # NOT apply to SAFE_STOP (throttle_scale=0.0), which must still
        # mean a real stop.
        self.min_maneuver_throttle = getattr(cfg, 'PLANNER_MIN_MANEUVER_THROTTLE',
                                              getattr(cfg, 'THROTTLE_MIN', 0.15))
        # Steering rate limit, added 2026-07-30 after cone_test5's return
        # leg: right after RETURN_TO_ORIGINAL_LANE began, commanded
        # steering oscillated wildly frame-to-frame (-0.13 -> +0.216 ->
        # -0.375 -> -0.6 -> ... -> -0.794 within ~10 frames) before white
        # reappeared and it settled. Bounds how much the CAPPED steering
        # (see decision.steering_cap below) can change per frame, applied
        # only while a cap is active (SWITCH_TO_NEIGHBOR_LANE/RETURN_TO_
        # ORIGINAL_LANE in ACTIVE mode) -- normal lane-following steering
        # is never rate-limited by this.
        self.lane_change_steer_rate_limit = getattr(cfg, 'LANE_CHANGE_STEER_RATE_LIMIT', 0.15)
        self._last_out_steering = None

    def run(self, cam_img, depth_img, pilot_steering, pilot_throttle,
            lane_yellow_x, lane_white_x, lane_width_px):
        geometry = LaneGeometry(
            yellow_x=lane_yellow_x,
            white_x=lane_white_x,
            width_px=lane_width_px,
            white_right_of_yellow=self.lane_follower.white_right_of_yellow,
            primary_row_y=self.primary_row_y,
            # read directly off the tracker object (not the Memory bus) --
            # lane/yellow_x itself never resets to None on a miss, so
            # lost_frames is the clean freshness signal getattr(cfg,...)
            # on the published value alone can't give us.
            yellow_lost_frames=self.lane_follower.yellow_trackers[0].lost_frames,
        )
        detection, _debug = self.cone_detector.detect(cam_img, depth_img)
        decision = self.planner.step(detection, geometry, raw_steering=pilot_steering)

        out_steering = pilot_steering  # overridden only by the capped/rate-limited value below
        out_throttle = pilot_throttle

        if self.mode == RolloutMode.ACTIVE:
            if decision.requested_lane is not None \
                    and decision.requested_lane != self.lane_follower.current_lane:
                self.lane_follower.set_lane(decision.requested_lane)
            if decision.steering_cap is not None:
                # Bound the MAGNITUDE (never the direction/sign) of
                # LaneFollower's own steering during the crossing -- see
                # ObstaclePlanner._steering_cap_for_progress and the
                # module docstring's reuse-the-existing-PID rationale.
                cap = abs(decision.steering_cap)
                capped = max(-cap, min(cap, pilot_steering))
                if self._last_out_steering is not None:
                    delta = capped - self._last_out_steering
                    limit = self.lane_change_steer_rate_limit
                    if delta > limit:
                        capped = self._last_out_steering + limit
                    elif delta < -limit:
                        capped = self._last_out_steering - limit
                out_steering = capped
            scaled_throttle = pilot_throttle * decision.throttle_scale
            if 0.0 < decision.throttle_scale < 1.0:
                # actively maneuvering (not FOLLOW_LANE/OBJECT_WATCH at
                # scale 1.0, not a genuine SAFE_STOP at scale 0.0) -- floor
                # it so this scale-down can't compound with LaneFollower's
                # own already-reduced low-confidence throttle into
                # something too small to actually move the car
                out_throttle = max(scaled_throttle, self.min_maneuver_throttle)
            else:
                out_throttle = scaled_throttle
            self._last_out_steering = out_steering
        else:
            self._last_out_steering = None
        if self.mode == RolloutMode.SHADOW:
            if decision.requested_lane is not None or decision.throttle_scale < 1.0:
                logger.info(f"[SHADOW] would apply: requested_lane={decision.requested_lane} "
                            f"throttle_scale={decision.throttle_scale:.2f} state={decision.state.value} "
                            f"reason={decision.reason}")
        # RolloutMode.OBSERVE: compute + publish only, no logging beyond
        # what the returned Memory keys already give a tub recording.

        bbox = detection.bbox if detection else None
        return (
            out_steering, out_throttle,
            decision.state.value, decision.reason, decision.requested_lane or '',
            decision.cone_in_path,
            bbox.x if bbox else None, bbox.y if bbox else None,
            bbox.w if bbox else None, bbox.h if bbox else None,
            detection.distance_mm if detection else None,
            detection.distance_valid if detection else False,
            detection.distance_source if detection else '',
        )
