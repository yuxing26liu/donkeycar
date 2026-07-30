#!/usr/bin/env python3
"""
verify_cone_tub.py

Post-run sanity check for a shadow-mode (or later, active-mode) cone-
avoidance tub. Reports what's actually present, not just "did it crash" -
missing fields, depth shape/validity, state transitions, and whether
steering/throttle ever moved in a way inconsistent with shadow mode
(never conclusive proof on its own - see the note printed at the end;
the conclusive guarantee is the code-level test in
donkeycar/tests/test_pilot_arbiter.py, which asserts this directly
against the real merged config).

Usage:
    python scripts/verify_cone_tub.py --tub /path/to/tub_NN_YY-MM-DD
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image


def load_records(tub_dir):
    recs = []
    for cf in sorted(glob.glob(os.path.join(tub_dir, "catalog_*.catalog"))):
        for line in open(cf):
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    recs.sort(key=lambda r: r['_index'])
    return recs


def field_coverage(recs, key):
    present = sum(1 for r in recs if key in r and r[key] is not None)
    return present, len(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tub', required=True)
    args = ap.parse_args()
    tub_dir = args.tub

    recs = load_records(tub_dir)
    n = len(recs)
    print(f"=== {tub_dir} ===")
    print(f"records: {n}")
    if n == 0:
        print("EMPTY TUB - nothing to verify")
        return

    manifest_path = os.path.join(tub_dir, "manifest.json")
    with open(manifest_path) as f:
        inputs = json.loads(f.readline())
        types = json.loads(f.readline())
    print(f"\nschema (from manifest.json): {list(zip(inputs, types))}")

    fields = ['cam/image_array', 'steering', 'throttle',
              'lane/yellow_x', 'lane/white_x', 'lane/width_px',
              'cam/depth_array',
              'planner/state', 'planner/reason', 'planner/requested_lane', 'planner/cone_in_path',
              'cone/bbox_x', 'cone/bbox_y', 'cone/bbox_w', 'cone/bbox_h',
              'cone/distance_mm', 'cone/distance_valid', 'cone/distance_source']

    print("\n--- field coverage ---")
    missing_entirely = []
    for f in fields:
        present, total = field_coverage(recs, f)
        pct = 100.0 * present / total
        flag = "" if present > 0 else "  <-- NEVER PRESENT"
        print(f"  {f:28s}: {present:5d}/{total} ({pct:5.1f}%){flag}")
        if present == 0:
            missing_entirely.append(f)

    if missing_entirely:
        print(f"\nWARNING: these fields were NEVER recorded: {missing_entirely}")
        print("(check OBSTACLE_AVOIDANCE_MODE != 'disabled' and CV_CONTROLLER_CLASS == 'LaneFollower' in the myconfig used)")

    # depth shape/validity
    depth_recs = [r for r in recs if r.get('cam/depth_array')]
    if depth_recs:
        shapes = set()
        valid_fracs = []
        for r in depth_recs[::max(1, len(depth_recs) // 30)]:  # sample up to ~30 frames
            d = np.array(Image.open(os.path.join(tub_dir, "images", r['cam/depth_array'])))
            shapes.add(d.shape)
            valid_fracs.append(float(np.mean(d >= 150)))
        print(f"\n--- depth ---")
        print(f"  frames with depth: {len(depth_recs)}/{n}")
        print(f"  observed shapes (sampled): {shapes}", "<-- inconsistent!" if len(shapes) > 1 else "(consistent)")
        print(f"  mean valid-pixel fraction (>=150mm), sampled: {np.mean(valid_fracs):.2%}")
    else:
        print("\n--- depth ---\n  NO depth frames recorded - check OAKD_DEPTH=True in the myconfig used for this drive")

    # planner state transitions
    state_recs = [(r['_index'], r['planner/state']) for r in recs if 'planner/state' in r]
    if state_recs:
        print(f"\n--- planner state transitions ---")
        prev = None
        transitions = []
        for idx, state in state_recs:
            if state != prev:
                transitions.append((idx, state))
                prev = state
        for idx, state in transitions:
            print(f"  idx={idx:5d} -> {state}")
        print(f"  total distinct transitions: {len(transitions)}")
        # crude oscillation flag: same state visited 4+ separate times
        from collections import Counter
        visits = Counter(s for _, s in transitions)
        oscillating = [s for s, c in visits.items() if c >= 4]
        if oscillating:
            print(f"  WARNING: possible oscillation - visited repeatedly: {oscillating}")
    else:
        print("\n--- planner state transitions ---\n  no planner/state field recorded")

    # requested_lane vs actual steering/throttle sanity (weak check only)
    requested = [(r['_index'], r.get('planner/requested_lane', '')) for r in recs]
    ever_requested = [x for x in requested if x[1]]
    print(f"\n--- lane-switch requests (shadow mode: should NEVER be applied) ---")
    print(f"  frames where planner requested a lane switch: {len(ever_requested)}/{n}")
    if ever_requested:
        print(f"  first request at idx={ever_requested[0][0]} -> {ever_requested[0][1]}")
        print("  NOTE: a non-empty requested_lane here is EXPECTED and fine in shadow mode -")
        print("  it means the planner would have switched in active mode. What matters is")
        print("  whether 'steering'/'throttle' show any abrupt change coincident with it.")
        idx0 = ever_requested[0][0]
        window = [r for r in recs if idx0 - 3 <= r['_index'] <= idx0 + 3]
        print(f"  steering/throttle around idx={idx0}:")
        for r in window:
            print(f"    idx={r['_index']}: steering={r.get('steering')}, throttle={r.get('throttle')}")

    print("\nNOTE: this script's steering/throttle check is a WEAK behavioral")
    print("indicator only - it cannot conclusively prove the arbiter never wrote")
    print("to pilot/steering/pilot/throttle from tub data alone (no pre-arbiter")
    print("value is separately recorded). The CONCLUSIVE guarantee is the")
    print("code-level test in donkeycar/tests/test_pilot_arbiter.py, which")
    print("asserts this directly against the real merged myconfig_cone.py config.")


if __name__ == '__main__':
    main()
