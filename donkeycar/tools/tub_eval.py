#!/usr/bin/env python3
"""
tub_eval.py - offline scoring harness for lane-follower candidates.

Runs a candidate CV-controller class (same (pid, cfg) constructor / run(cam_img)
interface as donkeycar/parts/lane_follower.py's LaneFollower) against recorded
tubs in Desktop/tubs, without driving the car, and reports steering-behavior
metrics - plus, for tubs recorded by a human driving (not an autopilot), how
closely the candidate's steering would have matched what the human actually
did through the same turns.

Deliberately does NOT `import donkeycar` (or anything under that package) -
this dev machine is missing a declared dependency (pyfiglet) even for basic
imports, and importing anything dotted under `donkeycar.` re-triggers that
failure via the parent package's __init__.py. Candidate modules and
myconfig.py are instead loaded directly by file path via importlib, and the
tub_v2 manifest/catalog format is parsed by hand (json + cv2, no PIL) -
matching how prior tub-analysis sessions on this project already worked
around the same two missing packages.

Run this file directly, NOT with `python -m` (that imports it as
donkeycar.tools.tub_eval, which re-triggers the pyfiglet problem):

    python donkeycar/tools/tub_eval.py eval \
        --candidate donkeycar/parts/lane_follower3.py:LaneFollower

    python donkeycar/tools/tub_eval.py compare \
        --candidate donkeycar/parts/lane_follower3.py:LaneFollower \
        --baseline donkeycar/parts/lane_follower2.py:LaneFollower
"""
import argparse
import importlib.util
import json
import logging
import os
import types

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TUBS_DIR = "C:/Users/ryanc/OneDrive/AppData/Attachments/Desktop/tubs"
DEFAULT_MYCONFIG = "C:/Users/ryanc/OneDrive/AppData/Attachments/Desktop/myconfig.py"
# Tubs recorded by a human manually driving to stay centered in the lane, not
# by an autopilot - see project memory "Closed-loop testing strategy" for why
# these (and only these) are valid ground truth for a candidate's steering
# decisions. Every other tub was recorded under some prior autopilot, so it
# can only validate detection-side behavior, not a different algorithm's
# closed-loop steering (classic covariate-shift problem in offline replay).
DEFAULT_GROUND_TRUTH_TUBS = {"two_inner_laps", "two_outer_laps"}

DEFAULT_HZ = 20  # matches DRIVE_LOOP_HZ in myconfig.py

# Manual-intervention detection (see project memory "Tub analysis workflow"):
# a real pickup/reposition shows up as a timestamp gap and/or a frame-to-frame
# thumbnail-diff spike. Calibrated 2026-07-24 across all 8 autonomous tubs
# (18,723 frame pairs): timestamp gap>300ms is clean and sparse (43 frames,
# 0.23%) - kept as the auto-exclusion trigger. Thumbnail-diff is NOT safe to
# auto-exclude on: confirmed gap-verified interventions ranged from diff=1.3
# to 52.6 (some pickups barely change the view), while normal driving without
# any gap reaches diff=34.6 at just the 99.9th percentile (tub_16's direct-sun
# frames, sharp turns) - overlapping ranges mean a diff threshold tight enough
# to catch every pickup would also strip real hard-turn frames, which would
# quietly make a candidate look calmer than it actually is. So diff is only
# ever used as a reported diagnostic (n_high_motion_frames_flagged_not_excluded)
# here, never to exclude a frame.
INTERVENTION_GAP_MS = 300
EXCLUDE_FRAMES_AFTER_GAP = 2  # the reposition frame + one more, to let re-acquisition settle
HIGH_MOTION_DIFF_FLAG = 40.0  # just above the calibrated non-intervention p99.9 (~38.9) - flag only


class FakeClock:
    """
    Deterministic stand-in for simple_pid.PID's default wall-clock time_fn.

    Replaying frames back-to-back with no real-time pacing makes dt "however
    long this run of the script happened to take" unless a fixed-interval
    clock is injected instead - see project memory "PID timing bug in test
    harnesses". Advancing by exactly 1/hz every call means two runs of the
    same candidate against the same tub always produce identical
    steering-smoothness numbers, so comparisons between candidates are
    actually comparing the code, not scheduling noise.
    """

    def __init__(self, hz=DEFAULT_HZ):
        self._t = 0.0
        self._dt = 1.0 / hz

    def __call__(self):
        self._t += self._dt
        return self._t


def load_module_from_path(path, module_name=None):
    """Import a .py file directly by path - never touches donkeycar's
    package __init__ (see module docstring for why that matters here)."""
    path = os.path.abspath(path)
    module_name = module_name or f"_tub_eval_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_candidate_spec(spec):
    """'path/to/file.py:ClassName' -> (path, class_name)."""
    if ":" not in spec:
        raise ValueError(f"Candidate spec must be 'path/to/file.py:ClassName', got {spec!r}")
    path, class_name = spec.rsplit(":", 1)
    return path, class_name


def load_candidate(spec):
    """
    Returns (class, module_cfg_overrides). A candidate file may define a
    module-level CFG_OVERRIDES = {...} dict for any config it needs that
    differs from myconfig.py (new PID gains, a new tunable constant, etc.)
    instead of requiring a separate myconfig.py per candidate - myconfig.py
    stays the single shared source of truth for shared/hardware config,
    each candidate only declares the handful of keys its own strategy
    actually needs.
    """
    path, class_name = parse_candidate_spec(spec)
    module = load_module_from_path(path)
    return getattr(module, class_name), getattr(module, "CFG_OVERRIDES", {})


def load_cfg(myconfig_path=DEFAULT_MYCONFIG, overrides=None):
    """
    Load myconfig.py fresh and return it as a plain namespace (not the
    cached module object) so per-candidate overrides from one evaluation
    run can't leak into the next. myconfig.py is self-contained - every
    name it references is defined in the same file (see project memory
    "myconfig.py location") - so this doesn't need donkeycar's normal
    cfg_cv_control.py-plus-myconfig merge step.
    """
    module = load_module_from_path(myconfig_path)
    cfg = types.SimpleNamespace(**{
        k: v for k, v in vars(module).items() if not k.startswith("__")
    })
    for key, value in (overrides or {}).items():
        setattr(cfg, key, value)
    return cfg


def build_pid(cfg, hz=DEFAULT_HZ):
    from simple_pid import PID
    return PID(Kp=cfg.PID_P, Ki=cfg.PID_I, Kd=cfg.PID_D, time_fn=FakeClock(hz))


# ---------------------------------------------------------------------------
# Tub reading (hand-rolled, deliberately not donkeycar.parts.tub_v2 - see
# module docstring)
# ---------------------------------------------------------------------------

def read_manifest(tub_dir):
    manifest_path = os.path.join(tub_dir, "manifest.json")
    with open(manifest_path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    _inputs, _types, _metadata, _session_info, path_info = lines
    return {
        "paths": path_info["paths"],
        "deleted_indexes": set(path_info.get("deleted_indexes", [])),
    }


def iter_tub_records(tub_dir):
    """Yield records (dicts) from every catalog_N.catalog file listed in
    manifest.json's 'paths', in order, skipping deleted indexes."""
    manifest = read_manifest(tub_dir)
    for catalog_name in manifest["paths"]:
        catalog_path = os.path.join(tub_dir, catalog_name)
        if not os.path.exists(catalog_path):
            logger.warning(f"{tub_dir}: manifest lists {catalog_name} but it's missing, skipping")
            continue
        with open(catalog_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("_index") in manifest["deleted_indexes"]:
                    continue
                yield record


def load_frame(tub_dir, record, image_key="cam/image_array"):
    """Read the frame's jpg and return it as an RGB numpy array - matching
    what every CV controller's run(cam_img) expects. cv2.imread reads BGR
    by default, so this converts; no PIL involved (not installed on this
    dev machine - see module docstring)."""
    filename = record[image_key]
    image_path = os.path.join(tub_dir, "images", filename)
    bgr = cv2.imread(image_path)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _thumbnail(cam_img_rgb):
    small = cv2.resize(cam_img_rgb, (32, 18), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_candidate_on_tub(candidate_cls, tub_dir, cfg, hz=DEFAULT_HZ,
                               is_ground_truth=False, max_frames=None,
                               gap_ms_threshold=INTERVENTION_GAP_MS,
                               exclude_frames_after_gap=EXCLUDE_FRAMES_AFTER_GAP,
                               high_motion_diff_flag=HIGH_MOTION_DIFF_FLAG):
    """
    Replay every frame of one tub through candidate_cls(pid, cfg).run(cam_img)
    and compute steering-behavior metrics.

    Metrics come only from the standardized run() contract - (steering,
    throttle, image, ...) - never from candidate-specific internals, so
    this works unchanged for any future lane_follower3 architecture, not
    just today's blob-tracker design. That means it can't measure raw
    per-color detection hit-rate generically (that needs each candidate's
    internal trackers, which differ by design - see the more detailed,
    lane_follower1/2-specific version of that analysis in project memory
    "Tub analysis workflow"); instead this measures the thing that actually
    matters for comparing different *algorithms* against each other: how
    the car would have driven.

    Many tubs here (see module docstring / project memory) include the car
    being manually picked up mid-recording and/or veering off track and
    being repositioned - both show up as a timestamp gap. A frame right
    after one of those isn't a fair sample of the candidate's own behavior
    (the car was just placed down, possibly mid-motion or pointed
    somewhere transient), so it - and the next `exclude_frames_after_gap`
    frames - are excluded from every metric, and frame-to-frame diff
    metrics (sign_flip_rate) never compare across the boundary. The
    candidate's run() is still *called* on excluded frames (that's
    realistic - a live car doesn't get a "skip this frame" signal either),
    just not scored.
    """
    pid = build_pid(cfg, hz=hz)
    candidate = candidate_cls(pid, cfg)

    steering_series = []
    throttle_series = []
    episode_ids = []
    human_steering_series = [] if is_ground_truth else None
    n_errors = 0
    first_error = None
    n_interventions = 0
    n_excluded_for_intervention = 0
    n_high_motion_flagged = 0

    prev_ts = None
    prev_thumb = None
    episode_id = 0
    exclude_remaining = 0
    n_frames_seen = 0

    for record in iter_tub_records(tub_dir):
        if max_frames is not None and n_frames_seen >= max_frames:
            break
        n_frames_seen += 1
        cam_img = load_frame(tub_dir, record)
        if cam_img is None:
            continue

        ts = record.get("_timestamp_ms")
        is_boundary = False
        if prev_ts is not None and ts is not None and (ts - prev_ts) > gap_ms_threshold:
            is_boundary = True
            n_interventions += 1
            episode_id += 1
            exclude_remaining = exclude_frames_after_gap
        if ts is not None:
            prev_ts = ts

        # Diagnostic-only high-motion flag - never auto-excludes a frame,
        # see module docstring / INTERVENTION_GAP_MS comment for why.
        thumb = _thumbnail(cam_img)
        if prev_thumb is not None and not is_boundary:
            diff = float(np.mean(np.abs(thumb - prev_thumb)))
            if diff > high_motion_diff_flag:
                n_high_motion_flagged += 1
        prev_thumb = thumb

        try:
            result = candidate.run(cam_img)
        except Exception as exc:  # candidate code is experimental by design
            n_errors += 1
            if first_error is None:
                first_error = f"{type(exc).__name__}: {exc} (frame _index={record.get('_index')})"
            continue

        if exclude_remaining > 0:
            exclude_remaining -= 1
            n_excluded_for_intervention += 1
            continue  # candidate still ran on this frame, just not scored

        steering, throttle = result[0], result[1]
        steering_series.append(steering)
        throttle_series.append(throttle)
        episode_ids.append(episode_id)
        if is_ground_truth:
            human_steering_series.append(record.get("steering", 0.0))

    return _summarize(tub_dir, len(steering_series), n_errors, first_error,
                       steering_series, throttle_series, human_steering_series,
                       episode_ids, n_interventions, n_excluded_for_intervention,
                       n_high_motion_flagged)


def _summarize(tub_dir, n_frames, n_errors, first_error,
               steering_series, throttle_series, human_steering_series,
               episode_ids, n_interventions, n_excluded_for_intervention,
               n_high_motion_flagged):
    metrics = {
        "tub": os.path.basename(tub_dir),
        "n_frames": n_frames,
        "n_errors": n_errors,
        "first_error": first_error,
        "n_interventions_detected": n_interventions,
        "n_frames_excluded_for_intervention": n_excluded_for_intervention,
        "n_high_motion_frames_flagged_not_excluded": n_high_motion_flagged,
    }
    if n_frames == 0:
        metrics["ok"] = False
        return metrics

    steering = np.asarray(steering_series, dtype=float)
    throttle = np.asarray(throttle_series, dtype=float)
    episode_ids = np.asarray(episode_ids)

    metrics["ok"] = n_errors == 0
    metrics["mean_abs_steering"] = float(np.mean(np.abs(steering)))
    metrics["full_lock_fraction"] = float(np.mean(np.abs(steering) > 0.99))

    # Sign-flip rate only across consecutive frames in the SAME episode - a
    # jump across an excluded intervention boundary is the car being
    # repositioned, not the candidate's own steering behavior.
    if len(steering) > 1:
        same_episode = np.diff(episode_ids) == 0
        if np.any(same_episode):
            sign_diffs = np.diff(np.sign(steering))[same_episode]
            metrics["sign_flip_rate"] = float(np.mean(sign_diffs != 0))
        else:
            metrics["sign_flip_rate"] = 0.0
    else:
        metrics["sign_flip_rate"] = 0.0

    metrics["mean_throttle"] = float(np.mean(throttle))
    metrics["frac_stopped"] = float(np.mean(throttle <= 0.001))

    if human_steering_series is not None:
        human_steering = np.asarray(human_steering_series, dtype=float)
        metrics["mean_abs_steering_error_vs_human"] = float(
            np.mean(np.abs(steering - human_steering)))

    return metrics


def evaluate_across_tubs(candidate_spec, tubs_dir=None,
                          myconfig_path=DEFAULT_MYCONFIG, overrides=None,
                          ground_truth_tubs=None,
                          tub_names=None, hz=DEFAULT_HZ, max_frames=None):
    candidate_cls, module_overrides = load_candidate(candidate_spec)
    # candidate's own CFG_OVERRIDES apply first; anything passed explicitly
    # here (e.g. from the CLI) wins over that - see load_candidate docstring.
    merged_overrides = dict(module_overrides)
    merged_overrides.update(overrides or {})
    overrides = merged_overrides
    overrides.setdefault("OVERLAY_IMAGE", False)  # not needed for scoring, only slows replay down

    # Validation scope comes from myconfig.py (EVAL_TUBS_DIR / EVAL_TUBS /
    # EVAL_GROUND_TRUTH_TUBS) unless overridden by an explicit argument/CLI
    # flag. Decided 2026-07-24: the older autonomous tubs (tub_4..tub_31)
    # were all recorded under earlier, faulty lane-follower iterations, so
    # they no longer gate anything - the trusted validation set is the two
    # human-driven lap recordings (ground truth for steering) plus the most
    # recent on-car tub as a limited reference. With no EVAL_TUBS configured
    # anywhere, falls back to scanning every folder in the tubs dir, the old
    # behavior.
    scope_cfg = load_cfg(myconfig_path, overrides)
    if tubs_dir is None:
        tubs_dir = getattr(scope_cfg, "EVAL_TUBS_DIR", DEFAULT_TUBS_DIR)
    if tub_names is None:
        tub_names = getattr(scope_cfg, "EVAL_TUBS", None)
    if ground_truth_tubs is None:
        ground_truth_tubs = set(getattr(scope_cfg, "EVAL_GROUND_TRUTH_TUBS",
                                         DEFAULT_GROUND_TRUTH_TUBS))

    if tub_names is None:
        tub_names = sorted(
            d for d in os.listdir(tubs_dir)
            if os.path.isdir(os.path.join(tubs_dir, d))
            and os.path.exists(os.path.join(tubs_dir, d, "manifest.json"))
        )

    per_tub = {}
    for name in tub_names:
        tub_dir = os.path.join(tubs_dir, name)
        cfg = load_cfg(myconfig_path, overrides)  # fresh cfg per tub - no cross-run state leakage
        is_gt = name in ground_truth_tubs
        per_tub[name] = evaluate_candidate_on_tub(
            candidate_cls, tub_dir, cfg, hz=hz, is_ground_truth=is_gt, max_frames=max_frames)

    return {
        "candidate": candidate_spec,
        "per_tub": per_tub,
        "worst_case": _worst_case(per_tub),
    }


def _worst_case(per_tub):
    """Aggregate as worst-case across tubs, not average - a candidate that
    helps most tubs and wrecks one is a regression, not a win (see project
    memory "Scan-geometry generalization" / "Hit-rate needs a false-positive
    guard" for why this project doesn't trust averaged metrics alone)."""
    ok_tubs = {name: m for name, m in per_tub.items() if m.get("ok")}
    if not ok_tubs:
        return {"any_errors": True}
    steer_errs = [m["mean_abs_steering_error_vs_human"] for m in ok_tubs.values()
                  if "mean_abs_steering_error_vs_human" in m]
    return {
        "any_errors": any(not m.get("ok") for m in per_tub.values()),
        "max_mean_abs_steering": max(m["mean_abs_steering"] for m in ok_tubs.values()),
        "max_full_lock_fraction": max(m["full_lock_fraction"] for m in ok_tubs.values()),
        "max_sign_flip_rate": max(m["sign_flip_rate"] for m in ok_tubs.values()),
        "max_frac_stopped": max(m["frac_stopped"] for m in ok_tubs.values()),
        "max_steering_error_vs_human": max(steer_errs) if steer_errs else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(report):
    print(f"\n=== {report['candidate']} ===")
    for name, m in report["per_tub"].items():
        if not m.get("ok", False):
            print(f"  {name}: FAILED ({m['n_errors']} errors) - {m.get('first_error')}")
            continue
        line = (f"  {name}: mean|steer|={m['mean_abs_steering']:.3f} "
                f"full_lock={m['full_lock_fraction']*100:.1f}% "
                f"sign_flips={m['sign_flip_rate']*100:.1f}% "
                f"stopped={m['frac_stopped']*100:.1f}%")
        if "mean_abs_steering_error_vs_human" in m:
            line += f" steer_err_vs_human={m['mean_abs_steering_error_vs_human']:.3f}"
        print(line)
    print(f"  WORST CASE: {report['worst_case']}")


def compare_reports(candidate_report, baseline_report, tolerance=0.02):
    """
    Flag regressions: any tub where the candidate is worse than baseline by
    more than `tolerance` on a metric that matters. This is the pass/fail
    gate an iterate-many-candidates loop should key off of, not raw
    improvement - see project memory "Hit-rate needs a false-positive
    guard" / "Scan-geometry generalization" for why "better on average"
    isn't the bar this project uses.
    """
    regressions = []
    cand_tubs = candidate_report["per_tub"]
    base_tubs = baseline_report["per_tub"]
    for name in base_tubs:
        c, b = cand_tubs.get(name), base_tubs[name]
        if c is None or not c.get("ok"):
            regressions.append(f"{name}: candidate failed to run")
            continue
        if not b.get("ok"):
            continue  # baseline itself failed here, nothing to compare against
        for metric in ("full_lock_fraction", "sign_flip_rate", "frac_stopped",
                       "mean_abs_steering_error_vs_human"):
            if metric not in c or metric not in b:
                continue
            if c[metric] > b[metric] + tolerance:
                regressions.append(
                    f"{name}: {metric} regressed ({b[metric]:.3f} -> {c[metric]:.3f})")
    return regressions


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tubs-dir", default=None,
                         help="Override EVAL_TUBS_DIR from myconfig (fallback: Desktop/tubs)")
    common.add_argument("--myconfig", default=DEFAULT_MYCONFIG)
    common.add_argument("--tub", action="append", dest="tub_names",
                         help="Limit to this tub name (repeatable); default = EVAL_TUBS from "
                              "myconfig, else every tub in the tubs dir")
    common.add_argument("--max-frames", type=int, default=None,
                         help="Cap frames per tub, for a quick smoke test")
    common.add_argument("--output", help="Write the full JSON report here")

    p_eval = sub.add_parser("eval", parents=[common],
                             help="Score one candidate across tubs")
    p_eval.add_argument("--candidate", required=True,
                         help="path/to/file.py:ClassName")

    p_compare = sub.add_parser("compare", parents=[common],
                                help="Score a candidate against a baseline and flag regressions")
    p_compare.add_argument("--candidate", required=True, help="path/to/file.py:ClassName")
    baseline_group = p_compare.add_mutually_exclusive_group(required=True)
    baseline_group.add_argument("--baseline", help="path/to/file.py:ClassName (re-evaluated fresh)")
    baseline_group.add_argument("--baseline-report",
                                 help="Path to a JSON report from a previous 'eval --output ...' run - "
                                      "skips recomputing the baseline, so it only needs to be run once "
                                      "and every candidate compares against the same saved numbers")
    p_compare.add_argument("--tolerance", type=float, default=0.02)

    args = parser.parse_args()

    if args.command == "eval":
        report = evaluate_across_tubs(args.candidate, tubs_dir=args.tubs_dir,
                                       myconfig_path=args.myconfig,
                                       tub_names=args.tub_names,
                                       max_frames=args.max_frames)
        _print_report(report)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)

    elif args.command == "compare":
        candidate_report = evaluate_across_tubs(
            args.candidate, tubs_dir=args.tubs_dir, myconfig_path=args.myconfig,
            tub_names=args.tub_names, max_frames=args.max_frames)
        if args.baseline_report:
            with open(args.baseline_report) as f:
                baseline_report = json.load(f)
        else:
            baseline_report = evaluate_across_tubs(
                args.baseline, tubs_dir=args.tubs_dir, myconfig_path=args.myconfig,
                tub_names=args.tub_names, max_frames=args.max_frames)
        _print_report(baseline_report)
        _print_report(candidate_report)
        regressions = compare_reports(candidate_report, baseline_report, tolerance=args.tolerance)
        if regressions:
            print(f"\nREGRESSIONS ({len(regressions)}):")
            for r in regressions:
                print(f"  - {r}")
        else:
            print("\nNo regressions vs baseline - candidate survives.")
        if args.output:
            with open(args.output, "w") as f:
                json.dump({"candidate": candidate_report, "baseline": baseline_report,
                           "regressions": regressions}, f, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
