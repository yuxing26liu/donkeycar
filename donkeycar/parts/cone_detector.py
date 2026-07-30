"""
cone_detector.py

HSV+contour cone detector for the cone-dodger3000 branch. Detection-only:
never touches steering/throttle, just reports where the cone is (if any)
and how far away it is.

HSV thresholds below were NOT guessed -- they were measured directly off
real recorded frames across five different tubs/lighting conditions
(cone_static_right, cone_approach_right, cone_negative, avoid_cone, plus
the older tub_30_26-07-28) via connectedComponentsWithStats sampling:
observed cone-pixel population sat at H 0-17, S 83-207+, V 88-250 across
all of them. Defaults below (H 0-18, S 70-255, V 70-255) add margin below
the observed floor rather than tightening around it, matching this
project's usual practice of erring toward a looser threshold plus a
shape/continuity/mask-fraction guard (see lane_follower3.py's own
YELLOW_HSV_THRESHOLD_LOW comment for the same reasoning applied there).

Confirmed background false-positive source (found by running this same
threshold over the lane_change tub, which has no cone at all): a
reddish-orange wooden lounge chair in the courtyard background produces a
persistent ~400px blob at a fixed screen position. This detector's own
shape/aspect filters do not reliably exclude it -- and are not meant to;
the corridor/path-intersection classification in obstacle_planner.py is
what's supposed to reject an off-path detection like this one, not the
detector itself. Don't mistake "the detector doesn't reject the chair" for
a bug in this file.

Depth estimation is a robust percentile (default 25th) over an inner ROI
of the bbox, not a raw min or a full-bbox mean -- validated by replaying
this exact technique against tub_30_26-07-28 (a real recorded approach
with real depth data): it tracked the cone's actual back-and-forth motion
(computed distance ranged ~1429mm -> ~382mm -> ~1224mm as the cone was
walked in and back out), while that tub's own OLD recorded
depth/nearest_dist_mm field (a raw per-column min, since fixed elsewhere
per this project's depth-verification history) stayed pinned at
200-235mm throughout regardless of the cone's real position -- the same
"raw min latches onto stereo noise" failure mode already diagnosed for
that field, and the reason this file does not use a raw min either.

IMPORTANT CAVEAT: none of the five tubs recorded for THIS branch
(cone_static_right, cone_approach_right, cone_negative, lane_change,
avoid_cone) include depth data at all -- they predate/didn't use the
depth-recording infrastructure fix on this branch. The depth path above is
only validated against the older tub_30 (a stationary-camera, hand-walked
cone, not a driving approach). The bbox-height fallback below is
correspondingly NOT calibrated to real-world mm -- see its docstring.
"""
import logging

import cv2
import numpy as np

from donkeycar.parts.obstacle_types import BBox, Detection

logger = logging.getLogger(__name__)


class ConeDetector:
    def __init__(self, cfg):
        self.hsv_low = np.array(getattr(cfg, 'CONE_HSV_THRESHOLD_LOW', (0, 70, 70)), dtype=np.uint8)
        self.hsv_high = np.array(getattr(cfg, 'CONE_HSV_THRESHOLD_HIGH', (18, 255, 255)), dtype=np.uint8)
        self.min_area_px = getattr(cfg, 'CONE_MIN_AREA_PX', 80)
        # height/width >= this. Real samples ranged 1.2 (point-blank,
        # frame-cropped) to 2.3 (far) -- 0.8 gives margin below the
        # smallest observed value while still rejecting wide/flat blobs.
        self.min_aspect_ratio = getattr(cfg, 'CONE_MIN_ASPECT_RATIO', 0.8)
        # a point-blank cone legitimately covers ~35% of the frame
        # (confirmed: tub_30 idx700, area 36067/(426*240)=35.2%) -- set
        # well above that so only true glare/overexposure blowouts reject.
        self.max_mask_fraction = getattr(cfg, 'CONE_MAX_MASK_FRACTION', 0.6)
        self.morph_kernel_size = getattr(cfg, 'CONE_MORPH_KERNEL_SIZE', 3)
        # continuity gate: prefer the candidate blob nearest the last
        # tracked center (same idea as lane_follower3.py's preferred_x /
        # _select_line_blob continuity bias -- reused as a concept, not
        # imported, since this detector's candidate generation, shape
        # filters, and color space all differ from the line trackers').
        self.max_jump_px = getattr(cfg, 'CONE_MAX_JUMP_PX', 120)
        self.reacquire_after_frames = getattr(cfg, 'CONE_REACQUIRE_AFTER_FRAMES', 15)

        self.depth_roi_h_frac = getattr(cfg, 'CONE_DEPTH_ROI_H_FRAC', (0.30, 0.70))
        self.depth_roi_v_frac = getattr(cfg, 'CONE_DEPTH_ROI_V_FRAC', (0.35, 0.90))
        self.depth_percentile = getattr(cfg, 'CONE_DEPTH_PERCENTILE', 25)
        self.depth_min_valid_mm = getattr(cfg, 'CONE_DEPTH_MIN_VALID_MM', 150)
        self.depth_min_valid_px = getattr(cfg, 'CONE_DEPTH_MIN_VALID_PX', 20)

        self._last_cx = None
        self._lost_frames = 0

    def detect(self, rgb, depth=None):
        """
        rgb: HxWx3 uint8 RGB frame (cam/image_array).
        depth: optional HxW uint16 mm array (cam/depth_array), already
               resized to match rgb (see oak_d.py) -- None if unavailable.

        Returns (Detection or None, debug: dict). debug always has a
        'reason' key explaining a None result, plus mask_fraction/
        candidate_count when available, for offline replay/logging.
        """
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.hsv_low, self.hsv_high)
        if self.morph_kernel_size > 1:
            kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        mask_fraction = float(np.count_nonzero(mask)) / mask.size
        if mask_fraction > self.max_mask_fraction:
            self._register_miss()
            return None, dict(reason='mask_fraction_too_large', mask_fraction=mask_fraction)

        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
        candidates = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_area_px:
                continue
            x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                          int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
            if w == 0 or (h / float(w)) < self.min_aspect_ratio:
                continue
            candidates.append((area, x, y, w, h, float(cents[i][0])))

        if not candidates:
            self._register_miss()
            return None, dict(reason='no_candidates', mask_fraction=mask_fraction)

        best = self._select(candidates)
        area, x, y, w, h, cx = best
        bbox = BBox(x, y, w, h)
        self._last_cx = cx
        self._lost_frames = 0

        distance_mm, distance_valid, source = self._estimate_distance(bbox, depth)
        det = Detection(bbox=bbox, area_px=area, distance_mm=distance_mm,
                         distance_valid=distance_valid, distance_source=source)
        return det, dict(reason='ok', mask_fraction=mask_fraction, candidate_count=len(candidates))

    def _select(self, candidates):
        if self._last_cx is not None and self._lost_frames <= self.reacquire_after_frames:
            nearest = min(candidates, key=lambda c: abs(c[5] - self._last_cx))
            if abs(nearest[5] - self._last_cx) <= self.max_jump_px:
                return nearest
            # no candidate is plausibly continuous with the last track --
            # fall through to cold-start (largest-wins) selection below,
            # same as a fresh reacquire after a genuine loss.
        return max(candidates, key=lambda c: c[0])

    def _register_miss(self):
        self._lost_frames += 1
        if self._lost_frames > self.reacquire_after_frames:
            self._last_cx = None

    def _estimate_distance(self, bbox, depth):
        if depth is not None:
            x0 = int(bbox.x + self.depth_roi_h_frac[0] * bbox.w)
            x1 = int(bbox.x + self.depth_roi_h_frac[1] * bbox.w)
            y0 = int(bbox.y + self.depth_roi_v_frac[0] * bbox.h)
            y1 = int(bbox.y + self.depth_roi_v_frac[1] * bbox.h)
            x0, x1 = max(0, x0), min(depth.shape[1], max(x1, x0 + 1))
            y0, y1 = max(0, y0), min(depth.shape[0], max(y1, y0 + 1))
            roi = depth[y0:y1, x0:x1].astype(np.float32)
            valid = roi[roi >= self.depth_min_valid_mm]
            if valid.size >= self.depth_min_valid_px:
                return float(np.percentile(valid, self.depth_percentile)), True, 'depth'
        # Bounding-box-height fallback: an ORDINAL closer-is-bigger proxy
        # only. No px->mm calibration is available -- the only tub with
        # both real depth and a driving approach (tub_30) wasn't a lane
        # approach, and the tubs that ARE lane approaches (cone_approach_
        # right etc.) have no depth to calibrate against. Callers must
        # read bbox.h directly and compare against pixel thresholds, not
        # treat this as a distance in mm. See obstacle_planner.py.
        return None, False, 'bbox_height'
