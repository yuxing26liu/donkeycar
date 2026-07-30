#!/usr/bin/env python3
"""
replay_cone_planner.py

Offline validation harness for cone-dodger3000: replays real recorded
tubs through the ACTUAL LaneFollower, ConeDetector, and ObstaclePlanner
classes (not a reimplementation of their logic), following this project's
established practice of scoring candidates against real footage across
many tubs/lighting conditions before ever trusting them on the car (see
donkeycar/tools/tub_eval.py for the precedent this follows for
lane-following work).

IMPORTANT, project-specific reason this script reconstructs lane geometry
instead of reading it from the tub: none of the five tubs recorded for
this branch (cone_static_right, cone_approach_right, cone_negative,
lane_change, avoid_cone) have lane/yellow_x populated at all -- they were
recorded under manual control, not with LaneFollower running as the
active CV controller. This script feeds each tub's raw cam/image_array
frames through a real LaneFollower instance (constructed from the actual
deployed myconfig.py, not generic defaults) to reconstruct what its
lane/yellow_x, lane/white_x, lane/width_px would have been -- this is
the same class, same tuned thresholds, that would run on the car.

Usage:
    python scripts/replay_cone_planner.py --tub cone_approach_right
    python scripts/replay_cone_planner.py --all
    python scripts/replay_cone_planner.py --tub lane_change --lane-change-study
    python scripts/replay_cone_planner.py --tub avoid_cone --simulate-active

Does not modify any tub. Writes a JSON summary + optional annotated debug
frames to a report directory.
"""
import argparse
import glob
import importlib.util
import json
import os

import cv2
import numpy as np
from PIL import Image
from simple_pid import PID

from donkeycar.parts.cone_detector import ConeDetector
from donkeycar.parts.lane_follower3 import LaneFollower
from donkeycar.parts.obstacle_planner import ObstaclePlanner
from donkeycar.parts.obstacle_types import LaneGeometry

DEFAULT_TUBS_DIR = r"c:\Users\ryanc\OneDrive\AppData\Attachments\Desktop\tubs"
DEFAULT_MYCONFIG = r"c:\Users\ryanc\OneDrive\AppData\Attachments\Desktop\myconfig.py"

TUB_LIST = ['cone_static_right', 'cone_approach_right', 'cone_negative', 'lane_change', 'avoid_cone']


def load_myconfig(path):
    """Import the real deployed myconfig.py as a module so this replay
    uses the actual tuned LANE_*/CONE_* values, not hand-copied defaults
    that could silently drift from what's really on the car."""
    spec = importlib.util.spec_from_file_location("myconfig_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_records(tub_dir):
    recs = []
    for cf in sorted(glob.glob(os.path.join(tub_dir, "catalog_*.catalog"))):
        for line in open(cf):
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    recs.sort(key=lambda r: r['_index'])
    return recs


def load_rgb(tub_dir, rec):
    return np.array(Image.open(os.path.join(tub_dir, "images", rec['cam/image_array'])))


def load_depth(tub_dir, rec):
    if 'cam/depth_array' not in rec:
        return None
    return np.array(Image.open(os.path.join(tub_dir, "images", rec['cam/depth_array'])))


def make_lane_geometry(lf, primary_row_y):
    return LaneGeometry(
        yellow_x=lf.last_yellow_x,
        white_x=lf.last_white_x,
        width_px=lf.lane_width_px,
        white_right_of_yellow=lf.white_right_of_yellow,
        primary_row_y=primary_row_y,
    )


def replay_tub(tub_name, tubs_dir, cfg, out_dir, apply_set_lane=False, max_frames=None):
    tub_dir = os.path.join(tubs_dir, tub_name)
    recs = load_records(tub_dir)
    if max_frames:
        recs = recs[:max_frames]

    pid = PID(cfg.PID_P, cfg.PID_I, cfg.PID_D)
    lf = LaneFollower(pid, cfg)
    primary_row_y = lf.scan_rows[0]['scan_y']
    detector = ConeDetector(cfg)
    planner = ObstaclePlanner(cfg)

    summary = dict(tub=tub_name, n_frames=len(recs), states=[], transitions=[],
                   detections=0, relevant_frames=0, depth_frames=0,
                   bbox_height_series=[], distance_series=[], state_series=[],
                   set_lane_calls=[])
    prev_state = planner.state

    for rec in recs:
        rgb = load_rgb(tub_dir, rec)
        depth = load_depth(tub_dir, rec)
        if depth is not None:
            summary['depth_frames'] += 1

        lf.run(rgb)  # reconstruct lane geometry (steering/throttle from this discarded -- manual drive)
        geometry = make_lane_geometry(lf, primary_row_y)

        detection, ddebug = detector.detect(rgb, depth)
        decision = planner.step(detection, geometry)

        if detection is not None:
            summary['detections'] += 1
            summary['bbox_height_series'].append([rec['_index'], detection.bbox.h])
            if detection.distance_valid:
                summary['distance_series'].append([rec['_index'], detection.distance_mm])
        if decision.cone_in_path:
            summary['relevant_frames'] += 1
        summary['state_series'].append([rec['_index'], decision.state.value])

        if planner.state != prev_state:
            summary['transitions'].append(dict(idx=rec['_index'], frm=prev_state.value,
                                                to=planner.state.value, reason=decision.reason))
            prev_state = planner.state

        if apply_set_lane and decision.requested_lane is not None \
                and decision.requested_lane != lf.current_lane:
            lf.set_lane(decision.requested_lane)
            summary['set_lane_calls'].append(dict(idx=rec['_index'], lane=decision.requested_lane))

    summary['final_state'] = planner.state.value
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{tub_name}_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def lane_change_reacquire_study(tub_name, tubs_dir, cfg, out_dir, steering_threshold=0.35, min_run=8):
    """
    No cone involved -- studies set_lane()'s reacquisition latency against
    a real manually-driven lane-change tub. Candidate transition windows
    are INFERRED from sustained steering magnitude in the tub's own real
    recorded steering values (real telemetry, not a fabricated label) --
    reported explicitly as inferred/approximate, not ground truth.
    """
    tub_dir = os.path.join(tubs_dir, tub_name)
    recs = load_records(tub_dir)

    # find runs of sustained steering in one direction = candidate lane changes
    runs = []
    cur_sign, cur_start, cur_len = 0, None, 0
    for rec in recs:
        st = rec.get('steering', 0.0)
        sign = 1 if st > steering_threshold else (-1 if st < -steering_threshold else 0)
        if sign != 0 and sign == cur_sign:
            cur_len += 1
        else:
            if cur_sign != 0 and cur_len >= min_run:
                runs.append((cur_start, cur_start + cur_len - 1, cur_sign))
            cur_sign, cur_start, cur_len = sign, rec['_index'], 1 if sign != 0 else 0
    if cur_sign != 0 and cur_len >= min_run:
        runs.append((cur_start, cur_start + cur_len - 1, cur_sign))

    pid = PID(cfg.PID_P, cfg.PID_I, cfg.PID_D)
    lf = LaneFollower(pid, cfg)
    recs_by_idx = {r['_index']: r for r in recs}

    result = dict(tub=tub_name, inferred_transition_windows=runs, reacquisitions=[])

    # run continuously through the whole tub, and at the end of each
    # inferred window, call set_lane() to the opposite side and measure
    # how many subsequent frames until corridor geometry is valid again.
    windows_by_end = {end: sign for (_, end, sign) in runs}
    pending_switch = None
    frames_since_switch = None
    for idx in sorted(recs_by_idx):
        rec = recs_by_idx[idx]
        rgb = load_rgb(tub_dir, rec)
        lf.run(rgb)
        corridor_valid = lf.last_yellow_x is not None or lf.last_white_x is not None

        if frames_since_switch is not None:
            frames_since_switch += 1
            if corridor_valid:
                result['reacquisitions'].append(dict(
                    switch_idx=pending_switch, reacquired_idx=idx,
                    frames_to_reacquire=frames_since_switch))
                frames_since_switch = None
                pending_switch = None
            elif frames_since_switch > 60:  # gave up
                result['reacquisitions'].append(dict(
                    switch_idx=pending_switch, reacquired_idx=None,
                    frames_to_reacquire=None, note="did not reacquire within 60 frames"))
                frames_since_switch = None
                pending_switch = None

        if idx in windows_by_end and frames_since_switch is None:
            target = 'left' if lf.current_lane == 'right' else 'right'
            lf.set_lane(target)
            pending_switch = idx
            frames_since_switch = 0

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{tub_name}_lane_change_study.json"), 'w') as f:
        json.dump(result, f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tub')  # any tub directory name under --tubs-dir, not restricted to TUB_LIST
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--tubs-dir', default=DEFAULT_TUBS_DIR)
    ap.add_argument('--myconfig', default=DEFAULT_MYCONFIG)
    ap.add_argument('--out-dir', default=os.path.join(os.path.dirname(__file__), '..', '_replay_reports'))
    ap.add_argument('--simulate-active', action='store_true',
                     help='actually call lane_follower.set_lane() when the planner requests it')
    ap.add_argument('--lane-change-study', action='store_true')
    ap.add_argument('--max-frames', type=int, default=None)
    args = ap.parse_args()

    cfg = load_myconfig(args.myconfig)
    out_dir = os.path.abspath(args.out_dir)

    tubs = TUB_LIST if args.all else ([args.tub] if args.tub else [])
    for tub in tubs:
        if args.lane_change_study and tub == 'lane_change':
            r = lane_change_reacquire_study(tub, args.tubs_dir, cfg, out_dir)
            print(f"{tub}: {len(r['inferred_transition_windows'])} inferred transitions, "
                  f"{len(r['reacquisitions'])} reacquisitions measured")
        else:
            r = replay_tub(tub, args.tubs_dir, cfg, out_dir,
                            apply_set_lane=args.simulate_active, max_frames=args.max_frames)
            print(f"{tub}: {r['n_frames']} frames, {r['detections']} detections "
                  f"({r['relevant_frames']} in-path), final_state={r['final_state']}, "
                  f"transitions={len(r['transitions'])}, set_lane_calls={len(r['set_lane_calls'])}")


if __name__ == '__main__':
    main()
