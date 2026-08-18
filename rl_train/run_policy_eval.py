"""RL policy evaluation entry point.

Runs evaluation rollouts for every entry in `session_config.json:evaluate_param_list`
and produces a unified composite figure per rollout. Replay video is always
generated (used as the source for the skeleton snapshot). Legacy per-panel PNGs
are off by default; enable with `--legacy-plots`.

Usage:
    python -m rl_train.run_policy_eval <log_dir> [--legacy-plots] [--no-show] [--regen]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import warnings

import numpy as np


@contextlib.contextmanager
def _silence_stdout():
    """Redirect stdout to /dev/null for noisy third-party init blocks. Stderr stays."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def _parse_args():
    p = argparse.ArgumentParser(description="RL policy evaluation.")
    p.add_argument("log_dir", nargs="?", default=None)
    p.add_argument("--legacy-plots", action="store_true", help="Also write the legacy per-panel PNGs.")
    p.add_argument("--no-show", action="store_true", help="Skip the pop-out composite window.")
    p.add_argument("--regen", action="store_true", help="Regenerate evaluated gait data even if it already exists.")
    p.add_argument(
        "--varying",
        action="store_true",
        help="Override evaluate_param_list with a single SINUSOIDAL "
        "varying-speed rollout (0.8-1.4 m/s) and emit the "
        "speed-tracking composite.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override num_timesteps for every rollout. The configs ship 200 steps (~6.7 s, "
        "about 5 strides), which is too few strides to put an error bar on a per-phase "
        "quantity such as when in the cycle the exo peaks. Raising it costs only rollout "
        "time and needs no config edit, so it also works on already-trained sessions.",
    )
    p.add_argument(
        "--cmap",
        choices=("rainbow", "teal", "bluered"),
        default="rainbow",
        help="Speed colour map for varying-speed composites.",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    log_dir = args.log_dir or input("Enter the log directory: ")
    show_composite = not args.no_show

    # Silence the noisy tight_layout warning emitted by some legacy panels.
    warnings.filterwarnings("ignore", message=".*tight_layout.*")

    from rl_train.utils.data_types import DictionableDataclass
    from rl_train.utils.train_log_handler import TrainLogHandler
    from rl_train.utils.train_checkpoint_data_imitation import ImitationTrainCheckpointData
    from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
    from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
    from rl_train.analyzer.train_log_analyzer import TrainLogAnalyzer
    from rl_train.analyzer.gait_analyze import GaitAnalyzer
    from rl_train.analyzer.gait_evaluate import GaitData, ImitationGaitEvaluator
    from myoassist_utils.eval_utils import build_composite, CompositeInputs, CMAPS

    with open(os.path.join(log_dir, "session_config.json"), "r") as f:
        config_dict = json.load(f)
    config = DictionableDataclass.create(ImitationTrainSessionConfig, config_dict)

    if args.varying:
        base = dict(config.evaluate_param_list[0]) if config.evaluate_param_list else {}
        base.update(
            {
                "velocity_mode": "SINUSOIDAL",
                "min_target_velocity": 0.8,
                "max_target_velocity": 1.4,
                "target_velocity_period": 20.0,
                "num_timesteps": 1200,
            }
        )
        base.setdefault("cam_distance", 3.0)
        base.setdefault("cam_type", "follow")
        base.setdefault("visualize_activation", True)
        base.setdefault("realtime_plotting_info", [])
        config.evaluate_param_list = [base]
        print("  --varying: SINUSOIDAL 0.8-1.4 m/s (period 20 s, 1200 steps)")

    if args.steps is not None:
        # Applied after --varying so an explicit --steps wins over that preset's 1200.
        config.evaluate_param_list = [dict(p, num_timesteps=args.steps) for p in config.evaluate_param_list]
        print(f"  --steps: {args.steps} control steps per rollout")

    print("=" * 60)
    print("RL Policy Evaluation")
    print("=" * 60)
    print(f"Log dir:       {log_dir}")
    print(f"Rollouts:      {len(config.evaluate_param_list)}")
    print(f"Composite:     enabled (show={show_composite})")
    print(f"Legacy plots:  {'enabled' if args.legacy_plots else 'disabled'}")
    print("=" * 60)

    for idx, evaluate_param in enumerate(config.evaluate_param_list):
        analyze_result_dir = os.path.join(log_dir, f"analyze_results_{idx:02d}")
        os.makedirs(analyze_result_dir, exist_ok=True)
        print(f"\n[Rollout {idx + 1}/{len(config.evaluate_param_list)}] -> {analyze_result_dir}")

        log_handler = TrainLogHandler(log_dir)
        log_handler.load_log_data(ImitationTrainCheckpointData)

        if args.legacy_plots:
            TrainLogAnalyzer(log_handler).plot_reward(result_dir=analyze_result_dir, show_plot=False)

        gait_data_name = "gait_evaluated_data.json"
        gait_data_path = os.path.join(analyze_result_dir, gait_data_name)
        is_regen = args.regen or not os.path.exists(gait_data_path)

        with _silence_stdout():
            gait_evaluator = ImitationGaitEvaluator(log_handler, config)
            gait_evaluator.load_reference_data()
            gait_evaluator.initialize_env()
            if is_regen:
                gait_data_path = gait_evaluator.evaluate(
                    result_dir=analyze_result_dir,
                    file_name=gait_data_name,
                    velocity_mode=MyoAssistLegBase.VelocityMode[evaluate_param["velocity_mode"]],
                    target_velocity_period=evaluate_param["target_velocity_period"],
                    max_timestep=evaluate_param["num_timesteps"],
                    min_target_velocity=evaluate_param["min_target_velocity"],
                    max_target_velocity=evaluate_param["max_target_velocity"],
                    terminate_when_done=True,
                )

        gait_data = GaitData()
        gait_data.read_json_data(gait_data_path)
        segmented_ref = np.load("rl_train/reference_data/segmented.npz", allow_pickle=True)
        segmented_ref_data = {k: segmented_ref[k] for k in segmented_ref.files}

        replay_path = os.path.join(analyze_result_dir, "replay.mp4")
        with _silence_stdout():
            gait_evaluator.replay(
                gait_data_path,
                replay_path,
                cam_distance=evaluate_param["cam_distance"],
                use_activation_visualization=evaluate_param["visualize_activation"],
                cam_type=evaluate_param["cam_type"],
                realtime_plotting_info=evaluate_param.get("realtime_plotting_info", []),
                video_fps=config.env_params.control_framerate,
            )
        print(f"  Replay saved to {replay_path}")

        gait_analyzer = GaitAnalyzer(gait_data, segmented_ref_data, show_plot=False)
        if len(gait_analyzer.get_gait_segment_index(is_right_foot_based=True)) < 1:
            print("  Warning: not enough gait data to plot — skipping composite.")
            continue

        if args.legacy_plots:
            with _silence_stdout():
                gait_analyzer.plot_entire_result(result_dir=analyze_result_dir, is_right_foot_based=True)
                gait_analyzer.plot_exo_segmented_data(result_dir=analyze_result_dir)
                gait_analyzer.plot_segmented_kinematics_result(result_dir=analyze_result_dir)
                gait_analyzer.plot_left_right_comparison(result_dir=analyze_result_dir)
                gait_analyzer.plot_right_ref_comparison(result_dir=analyze_result_dir)
                gait_analyzer.plot_segmented_muscle_data(result_dir=analyze_result_dir, is_plot_right=True)
                gait_analyzer.joint_angle_by_velocity(result_dir=analyze_result_dir)

        # Fresh wide-aspect snapshot rendered directly from the env (no cropping)
        skeleton_frame = None
        try:
            with _silence_stdout():
                skeleton_frame = gait_evaluator.render_snapshot(
                    gait_data_path,
                    cam_distance=evaluate_param["cam_distance"],
                )
        except Exception as e:
            print(f"  Warning: snapshot render failed: {e}")

        time_steps_curve = [d.num_timesteps for d in log_handler.log_datas]
        returns_curve = [d.average_reward_per_episode for d in log_handler.log_datas]

        metadata = {
            "Model checkpoint": os.path.basename(log_handler.get_path2save_model(log_handler.log_datas[-1].num_timesteps)),
            "Env ID": config.env_params.env_id,
            "Total timesteps": f"{log_handler.log_datas[-1].num_timesteps:,}",
            "Eval timesteps": str(evaluate_param["num_timesteps"]),
            "Velocity mode": evaluate_param["velocity_mode"],
            "Target velocity": f"{evaluate_param['min_target_velocity']:.2f}–{evaluate_param['max_target_velocity']:.2f} m/s",
        }

        # Varying-speed rollout -> speed-aware composite (per-stride teal kinematics
        # + speed-tracking / cadence / step-length row).
        is_varying = (
            evaluate_param["velocity_mode"] != "UNIFORM"
            or evaluate_param["min_target_velocity"] != evaluate_param["max_target_velocity"]
        )

        composite_path = os.path.join(analyze_result_dir, "composite.png")
        build_composite(
            CompositeInputs(
                gait_data=gait_data,
                skeleton_frame=skeleton_frame,
                ref_data=segmented_ref_data,
                return_curve=(time_steps_curve, returns_curve),
                title=os.path.basename(log_dir),
                metadata=metadata,
            ),
            save_path=composite_path,
            show=show_composite,
            speed_varying=is_varying,
            fs=config.env_params.control_framerate,
            cmap=CMAPS[args.cmap],
        )


if __name__ == "__main__":
    main()
