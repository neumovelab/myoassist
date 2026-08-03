#!/usr/bin/env python3
"""Control Optimization evaluation entry point.

Streamlined replacement for the legacy Tkinter GUI eval. Reads a JSON config that
points to a results directory + parameter file + pickle, runs the rollout via
`CtrlOptimGaitEvaluator`, and renders a single unified composite figure.

Usage:
    python -m ctrl_optim.run_eval --config path/to/eval_config.json
    python -m ctrl_optim.run_eval --results-dir <dir> [overrides...]

If `--config` is omitted, command-line overrides are used as a fallback. Any value
present in the JSON config is overridden by an explicit CLI flag.

Defaults are tuned for the preoptimized `exo_4param_1_25ms` example.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Optional

import numpy as np

from ctrl_optim.eval.gait_evaluator import (
    CtrlOptimEvalConfig,
    CtrlOptimGaitEvaluator,
)
from myoassist_utils.eval_utils import (
    CompositeInputs,
    build_composite,
    load_cma_fitness,
)

REFERENCE_NPZ = "rl_train/reference_data/segmented.npz"


def _load_reference():
    """Load the shared segmented gait reference (parity with RL eval)."""
    if not os.path.exists(REFERENCE_NPZ):
        return None
    try:
        ref = np.load(REFERENCE_NPZ, allow_pickle=True)
        return {k: ref[k] for k in ref.files}
    except Exception as e:
        print(f"Warning: could not load reference data ({REFERENCE_NPZ}): {e}")
        return None


def _find_in_dir(results_dir: str, suffix: str) -> Optional[str]:
    matches = sorted(glob.glob(os.path.join(results_dir, f"*{suffix}")))
    return matches[0] if matches else None


def _resolve_paths(results_dir: str, param_file: Optional[str], pkl_file: Optional[str],
                   param_type: str) -> tuple[str, Optional[str]]:
    """Auto-locate param/pkl files in results_dir if not explicitly given."""
    if not param_file:
        suffix = "_Best.txt" if param_type == "Best" else "_BestLast.txt"
        param_file = _find_in_dir(results_dir, suffix)
        if not param_file:
            raise FileNotFoundError(f"No *{suffix} found in {results_dir}")
    if not pkl_file:
        pkl_file = _find_in_dir(results_dir, "_Pickle.pkl")
    return param_file, pkl_file


def _build_config_from_args(args, cfg_dict: dict) -> tuple[CtrlOptimEvalConfig, str, Optional[str], str]:
    results_dir = args.results_dir or cfg_dict.get("results_dir")
    if not results_dir:
        raise ValueError("results_dir must be set via --results-dir or in the JSON config.")

    param_file, pkl_file = _resolve_paths(
        results_dir,
        args.param_file or cfg_dict.get("param_file"),
        args.pkl_file or cfg_dict.get("pkl_file"),
        args.param_type or cfg_dict.get("param_type", "Best"),
    )

    output_dir = args.output_dir or cfg_dict.get("output_dir") or os.path.join(results_dir, "eval_output")
    os.makedirs(output_dir, exist_ok=True)

    # Build evaluator config. CLI overrides JSON; JSON overrides hard defaults.
    def pick(name, default):
        cli_val = getattr(args, name, None)
        if cli_val is not None:
            return cli_val
        return cfg_dict.get(name, default)

    eval_config = CtrlOptimEvalConfig(
        param_file=param_file,
        sim_time=pick("sim_time", 10.0),
        target_velocity=pick("target_velocity", 1.25),
        mode=pick("mode", "2D"),
        init_pose=pick("init_pose", "walk_left"),
        slope_deg=pick("slope_deg", 0.0),
        delayed=pick("delayed", False),
        exo_bool=pick("exo_bool", True),
        fixed_exo=pick("fixed_exo", False),
        use_4param_spline=pick("use_4param_spline", True),
        max_torque=pick("max_torque", 1.0),
        model=pick("model", "tutorial"),
        n_points=pick("n_points", 4),
        msk_key=cfg_dict.get("msk_key"),
        device_key=cfg_dict.get("device_key"),
        camera_speed=cfg_dict.get("camera_speed"),
        camera_distance=cfg_dict.get("camera_distance", 3.5),
        camera_elevation=cfg_dict.get("camera_elevation", -10.0),
        camera_height=cfg_dict.get("camera_height"),
        camera_azimuth=cfg_dict.get("camera_azimuth"),
        render_width=cfg_dict.get("render_width", 1920),
        render_height=cfg_dict.get("render_height", 960),
        show_actuators=cfg_dict.get("show_actuators", True),
    )
    return eval_config, output_dir, pkl_file, results_dir


def main():
    parser = argparse.ArgumentParser(description="Control Optimization evaluation.")
    parser.add_argument("--config", type=str, help="Path to JSON eval config.")
    parser.add_argument("--results-dir", type=str)
    parser.add_argument("--param-file", type=str)
    parser.add_argument("--pkl-file", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--param-type", choices=["Best", "BestLast"], default=None)
    parser.add_argument("--sim-time", type=float, default=None)
    parser.add_argument("--target-velocity", type=float, default=None)
    parser.add_argument("--mode", choices=["2D", "3D"], default=None)
    parser.add_argument("--init-pose", default=None)
    parser.add_argument("--slope-deg", type=float, default=None)
    parser.add_argument("--exo-bool", type=lambda v: v.lower() == "true", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-show", action="store_true", help="Skip pop-out window.")
    parser.add_argument("--export-video", action="store_true",
                        help="Record the full rollout and write rollout.mp4 to the output dir.")
    args = parser.parse_args()

    cfg_dict: dict = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg_dict = json.load(f)

    eval_config, output_dir, pkl_file, results_dir = _build_config_from_args(args, cfg_dict)

    print("=" * 60)
    print("Control Optimization Evaluation")
    print("=" * 60)
    print(f"Results dir: {results_dir}")
    print(f"Param file:  {os.path.basename(eval_config.param_file)}")
    print(f"Pickle file: {os.path.basename(pkl_file) if pkl_file else '(none)'}")
    print(f"Output dir:  {output_dir}")
    print("=" * 60)

    # 1. Rollout (optionally record full video)
    if args.export_video or cfg_dict.get("export_video"):
        eval_config.export_video = True
    video_path = os.path.join(output_dir, "replay.mp4") if eval_config.export_video else None
    evaluator = CtrlOptimGaitEvaluator(eval_config)
    gait_data, skeleton_frame = evaluator.run(video_path=video_path)

    # Persist gait data in RL schema so downstream analyzers can reuse it
    gait_json_path = os.path.join(output_dir, "gait_evaluated_data.json")
    gait_data.save_json_data(gait_json_path)
    print(f"  Gait data saved to {gait_json_path}")

    # 2. CMA fitness — prefer outcmaes/*_fit.dat, fall back to pkl es.fit.hist
    cma_fitness = None
    try:
        fb, fm, fw = load_cma_fitness(results_dir, pkl_file)
        if fb is not None:
            cma_fitness = (fb, fm, fw)
            extras = "best+median+worst" if fm is not None else "best only"
            print(f"  Loaded CMA fitness history ({len(fb)} generations, {extras})")
    except Exception as e:
        print(f"Warning: could not load CMA fitness: {e}")

    # 3. Composite figure (with shared human gait reference for kinematics)
    metadata = {
        "Param file": os.path.basename(eval_config.param_file),
        "Model": eval_config.model,
        "Mode": eval_config.mode,
        "Init pose": eval_config.init_pose,
        "Sim time": f"{eval_config.sim_time:.1f} s",
        "Target velocity": f"{eval_config.target_velocity:.2f} m/s",
        "Slope": f"{eval_config.slope_deg:.1f} deg",
        "Exo": ("on" if eval_config.exo_bool else "off"),
    }
    if cma_fitness is not None:
        metadata["CMA generations"] = str(len(cma_fitness[0]))

    composite_path = os.path.join(output_dir, "composite.png")
    build_composite(
        CompositeInputs(
            gait_data=gait_data,
            skeleton_frame=skeleton_frame,
            ref_data=_load_reference(),
            cma_fitness=cma_fitness,
            title=os.path.basename(results_dir),
            metadata=metadata,
        ),
        save_path=composite_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
