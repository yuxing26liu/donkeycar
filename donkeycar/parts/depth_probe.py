"""
File: depth_probe.py

Obstacle-avoidance roadmap, Phase 2: verify the OAK-D depth signal on real
hardware/track before anything is allowed to steer off it. This part is
read-only with respect to driving -- it never touches steering/throttle. It
only inspects cam/depth_array and reports (a) periodic sanity stats (shape,
dtype, min/max, valid-pixel fraction) and (b) the nearest object's column
and distance within a scan slice, so both can be watched live
(LOGLEVEL=DEBUG) and recorded to a tub for offline analysis.

The horizontal-slice / per-column-minimum technique in get_nearest_object()
is the same one prototyped (but never verified on hardware) in
origin/estella's ObjectAvoider. Per the obstacle-avoidance roadmap decision,
that branch's code isn't being reused, but this specific technique is kept
as reference since it's a reasonable way to reduce a depth image to "nearest
thing in front of the car" -- it just hasn't been proven against a real
depth stream yet, which is exactly what this part is for.
"""
import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)


class DepthProbe:
    '''
    Depth-sensor verification probe. Does not affect steering/throttle.

    Takes a horizontal slice of the OAK-D depth image at cfg.DEPTH_SCAN_Y /
    cfg.DEPTH_SCAN_HEIGHT, masks out invalid readings (0, or anything nearer
    than cfg.DEPTH_MIN_VALID_MM, which is lens-adjacent noise), and reports
    the nearest column's position and distance. Every
    cfg.DEPTH_LOG_EVERY_N_FRAMES frames it also logs the raw depth image's
    shape/dtype/min/max/valid-fraction at INFO level.
    '''
    def __init__(self, cfg):
        self.scan_y = cfg.DEPTH_SCAN_Y
        self.scan_height = cfg.DEPTH_SCAN_HEIGHT
        self.min_valid_depth_mm = cfg.DEPTH_MIN_VALID_MM
        self.log_every_n = getattr(cfg, 'DEPTH_LOG_EVERY_N_FRAMES', 40)
        self.frame_count = 0

    def get_nearest_object(self, depth_array):
        '''
        input: depth_array, a HxW uint16 numpy array of depth in mm (0 = no reading)
        output: (nearest_col, nearest_dist_mm), or (None, None) if nothing
                valid was found in the scan slice

        Uses each column's MEDIAN valid depth (over the scan_height rows),
        not the raw closest pixel - real on-car footage (tub_30_26-07-28)
        showed scattered single-pixel stereo noise sitting right at the
        min_valid_depth_mm floor throughout the frame, at every real
        distance, which a raw per-column min latches onto every time: a
        held-up cone's approach from ~1.5m to filling the frame produced
        almost no change in the old min-based reading (200-220mm the
        entire time), while the scan slice's true median depth dropped
        ~6857mm -> ~400mm over those same frames. A column's median is
        only trusted if at least half its rows have a valid reading, so a
        mostly-empty column with one noisy pixel can't win by accident.
        '''
        scan_slice = depth_array[self.scan_y:self.scan_y + self.scan_height, :]

        # depthai reports 0 for pixels with no valid depth reading; mask
        # those (and anything closer than min_valid_depth_mm, which is
        # lens-adjacent noise) out before taking each column's median.
        valid = scan_slice >= self.min_valid_depth_mm
        valid_counts = valid.sum(axis=0)
        enough_valid = valid_counts >= (self.scan_height // 2)
        if not np.any(enough_valid):
            return None, None

        masked = np.where(valid, scan_slice.astype(np.float32), np.nan)
        # A column with zero valid pixels is all-NaN, which nanmedian
        # warns about even though its result (NaN) is exactly right - it
        # gets overridden to inf below via enough_valid anyway, so the
        # warning is noise, not a real problem; suppress it rather than
        # let it spam the console on every frame with an empty column
        # (e.g. looking at open sky/track with nothing in range).
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            col_median = np.nanmedian(masked, axis=0)
        col_median = np.where(enough_valid, col_median, np.inf)

        nearest_col = int(np.argmin(col_median))
        nearest_dist = int(col_median[nearest_col])
        return nearest_col, nearest_dist

    def run(self, depth_array):
        '''
        input: depth_array from cam/depth_array (None if OAKD_DEPTH is off,
               or CAMERA_TYPE isn't OAKD)
        output: (nearest_col, nearest_dist_mm) -- both None if unavailable
        '''
        if depth_array is None:
            return None, None

        self.frame_count += 1
        if self.frame_count % self.log_every_n == 0:
            nonzero = depth_array[depth_array > 0]
            valid_fraction = nonzero.size / depth_array.size
            d_min = int(nonzero.min()) if nonzero.size else 0
            d_max = int(nonzero.max()) if nonzero.size else 0
            logger.info(
                f"depth frame {self.frame_count}: shape={depth_array.shape} "
                f"dtype={depth_array.dtype} valid_fraction={valid_fraction:.2f} "
                f"min={d_min}mm max={d_max}mm"
            )

        nearest_col, nearest_dist = self.get_nearest_object(depth_array)
        if nearest_dist is not None:
            logger.debug(f"nearest object: col={nearest_col} dist={nearest_dist}mm")

        return nearest_col, nearest_dist
