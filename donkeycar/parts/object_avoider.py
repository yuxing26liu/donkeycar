import numpy as np
import logging

logger = logging.getLogger(__name__)


class ObjectAvoider:
    '''
    Depth based obstacle avoidance.
    This part takes a horizontal slice of the OAK-D depth image at a set Y
    coordinate. It finds the nearest (minimum depth) column within that
    slice. If that nearest point is closer than a danger distance, it
    overrides the incoming steering/throttle to swerve away from whichever
    side (left/right half of the image) the object is on, and slows the
    car down. Otherwise it passes steering/throttle through unchanged, so
    LineFollower keeps full control.
    '''
    def __init__(self, cfg):
        self.scan_y = cfg.DEPTH_SCAN_Y  # num pixels from the top to start horiz scan
        self.scan_height = cfg.DEPTH_SCAN_HEIGHT  # num pixels high to grab from horiz scan
        self.min_valid_depth_mm = cfg.OBJECT_MIN_VALID_DEPTH_MM  # ignore readings nearer than this (lens noise)
        self.danger_distance_mm = cfg.OBJECT_DANGER_DISTANCE_MM  # trigger avoidance below this distance
        self.avoid_steering_mag = abs(cfg.OBJECT_AVOID_STEERING)  # magnitude of the override steering value
        self.avoid_throttle = cfg.OBJECT_AVOID_THROTTLE  # throttle to use while avoiding

    def get_nearest_object(self, depth_array):
        '''
        find the horizontal index of the nearest object in the scan slice
        input: depth_array, a HxW uint16 numpy array of depth in mm (0 = no reading)
        output: index of nearest column, its distance in mm (0 if nothing valid found)
        '''
        iSlice = self.scan_y
        scan_slice = depth_array[iSlice: iSlice + self.scan_height, :]

        # depthai reports 0 for pixels with no valid depth reading; mask those
        # (and anything closer than min_valid_depth_mm, which is lens-adjacent
        # noise) out before taking the per-column minimum.
        valid = scan_slice >= self.min_valid_depth_mm
        if not np.any(valid):
            return 0, 0

        masked = np.where(valid, scan_slice, np.iinfo(scan_slice.dtype).max)
        col_min = np.min(masked, axis=0)
        nearest_col = int(np.argmin(col_min))
        nearest_dist = int(col_min[nearest_col])

        return nearest_col, nearest_dist

    def run(self, depth_array, steering, throttle):
        '''
        main runloop of the obstacle avoider
        input: depth_array (uint16 mm depth image), current steering and throttle
               (e.g. from LineFollower)
        output: steering, throttle -- unchanged unless an object is closer than
                OBJECT_DANGER_DISTANCE_MM, in which case they are overridden to
                swerve away from the object.
        '''
        if depth_array is None:
            return steering, throttle

        nearest_col, nearest_dist = self.get_nearest_object(depth_array)

        if nearest_dist == 0 or nearest_dist >= self.danger_distance_mm:
            return steering, throttle

        width = depth_array.shape[1]
        center = width / 2

        # object is left of center -> steer right (away from it), and vice versa
        avoid_steering = self.avoid_steering_mag if nearest_col < center else -self.avoid_steering_mag

        logger.debug(
            f"object at col={nearest_col} dist={nearest_dist}mm < "
            f"{self.danger_distance_mm}mm; avoiding with steering={avoid_steering:.1f}"
        )

        return avoid_steering, self.avoid_throttle
