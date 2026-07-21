"""
lane_follower.py

Successor to LineFollower (line_follower.py) and RobustLineFollower
(robust_line_follower.py). Those track a single line; this is meant to
reason about the full lane so the car can:
  - drive in the right lane of a two-lane track (bounded by a solid edge
    line on the outside and a dashed yellow line on the inside), and
  - detect and swerve around obstacles that appear on the track.

Design/implementation TBD.
"""

import cv2
import numpy as np
from simple_pid import PID
import logging

logger = logging.getLogger(__name__)


class LaneFollower:
    '''
    OpenCV based lane-following + obstacle-avoidance controller.

    TODO: design and implement.
    '''

    def __init__(self, pid, cfg):
        pass

    def run(self, cam_img):
        pass
