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

    def run(self, cam_img, depth_img, pilot_steering, pilot_throttle,
            lane_yellow_x, lane_white_x, lane_width_px):
        geometry = LaneGeometry(
            yellow_x=lane_yellow_x,
            white_x=lane_white_x,
            width_px=lane_width_px,
            white_right_of_yellow=self.lane_follower.white_right_of_yellow,
            primary_row_y=self.primary_row_y,
        )
        detection, _debug = self.cone_detector.detect(cam_img, depth_img)
        decision = self.planner.step(detection, geometry)

        out_steering = pilot_steering  # never overridden -- see module docstring
        out_throttle = pilot_throttle

        if self.mode == RolloutMode.ACTIVE:
            if decision.requested_lane is not None \
                    and decision.requested_lane != self.lane_follower.current_lane:
                self.lane_follower.set_lane(decision.requested_lane)
            out_throttle = pilot_throttle * decision.throttle_scale
        elif self.mode == RolloutMode.SHADOW:
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
