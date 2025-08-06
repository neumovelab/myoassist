
# Parse command line arguments for log_dir
import sys
if len(sys.argv) > 1:
    log_dir = sys.argv[1]
else:
    log_dir = ""

if log_dir == "":
    log_dir = input("Enter the log directory: ")
    # log_dir = "docs/assets/tutorial_rl_models/train_session_20250728-161129_tutorial_partial_obs" # partial obs
    # log_dir = "docs/assets/tutorial_rl_models/train_session_20250729-005528_tutorial_full_obs" # Full obs
show_plot = False
import os

import numpy as np
from rl_train.utils.data_types import DictionableDataclass
from rl_train.utils.data_types import DictionableDataclass

import os
from rl_train.utils.train_log_handler import TrainLogHandler
from rl_train.utils.train_checkpoint_data_imitation import ImitationTrainCheckpointData
import json
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass
with open(os.path.join(log_dir, "session_config.json"), 'r') as f:
    config_dict = json.load(f)
config = DictionableDataclass.create(ImitationTrainSessionConfig, config_dict)

for (idx, evaluate_param) in enumerate(config.evaluate_param_list):
    analyze_result_dir = os.path.join(log_dir,f"analyze_results_{idx:02d}")
    if not os.path.exists(analyze_result_dir):
        os.makedirs(analyze_result_dir)

    log_handler = TrainLogHandler(log_dir)
    log_handler.load_log_data(ImitationTrainCheckpointData)

    from rl_train.utils.data_types import DictionableDataclass
    DictionableDataclass.to_dict(log_handler.log_datas[-1])

    import sys
    sys.modules.pop('package.train_log_analyzer', None)
    from rl_train.analyzer.train_log_analyzer import TrainLogAnalyzer
    train_log_analyzer = TrainLogAnalyzer(log_handler)
    train_log_analyzer.plot_reward(result_dir=analyze_result_dir, show_plot=show_plot)


    


    from rl_train.envs.myoassist_leg_base import MyoAssistLegBase

        

    import sys
    from rl_train.analyzer.gait_analyze import GaitAnalyzer
    from rl_train.analyzer.gait_evaluate import GaitData


    gait_data_name = f"gait_evaluated_data.json"
    if os.path.exists(os.path.join(analyze_result_dir, gait_data_name)):
        user_input = input(f"Regenerate evaluate data? ({gait_data_name}) (y/n(anything))")
    else:
        user_input = "y"
    is_regen_evaluating_data = True if user_input == "y" else False

    from rl_train.analyzer.gait_evaluate import ImitationGaitEvaluator
    gait_evaluator = ImitationGaitEvaluator(log_handler, config)
    gait_evaluator.load_reference_data()
    gait_evaluator.initialize_env()
    if is_regen_evaluating_data:
        gait_data_path = gait_evaluator.evaluate(result_dir=analyze_result_dir,
                                                file_name=gait_data_name,
                                                velocity_mode=MyoAssistLegBase.VelocityMode[evaluate_param["velocity_mode"]],
                                                target_velocity_period=evaluate_param["target_velocity_period"],
                                                max_timestep=evaluate_param["num_timesteps"],
                                                min_target_velocity=evaluate_param["min_target_velocity"],
                                                max_target_velocity=evaluate_param["max_target_velocity"],
                                                terminate_when_done=True
                                                )
    else:
        gait_data_path = os.path.join(analyze_result_dir, gait_data_name)

    gait_data = GaitData()
    gait_data.read_json_data(gait_data_path)
    segmented_ref_data = np.load("rl_train/reference_data/segmented.npz", allow_pickle=True)
    segmented_ref_data = {key: segmented_ref_data[key] for key in segmented_ref_data.files}


    gait_evaluator.replay(gait_data_path, os.path.join(analyze_result_dir, "replay.mp4"),
                                                cam_distance=evaluate_param["cam_distance"],
                                                # max_time_step=evaluate_param["num_timesteps"],
                                                use_activation_visualization=evaluate_param["visualize_activation"],
                                                cam_type=evaluate_param["cam_type"],
                                                use_realtime_floating=False
                                                )

    gait_analyzer = GaitAnalyzer(gait_data, segmented_ref_data, show_plot)


    if len(gait_analyzer.get_gait_segment_index(is_right_foot_based=True)) < 1:
        print("="*10 + "Warning" + "="*10)
        print("Warning! Not enough gait data to plot. Skipping plotting.")
        print("="*10 + "Warning" + "="*10)

        continue


    gait_analyzer.plot_entire_result(result_dir=analyze_result_dir,is_right_foot_based=True)

    gait_analyzer.plot_exo_segmented_data(result_dir=analyze_result_dir)

    gait_analyzer.plot_segmented_kinematics_result(result_dir=analyze_result_dir)

    gait_analyzer.plot_left_right_comparison(result_dir=analyze_result_dir)

    gait_analyzer.plot_right_ref_comparison(result_dir=analyze_result_dir)

    gait_analyzer.plot_segmented_muscle_data(result_dir=analyze_result_dir, is_plot_right=True)


    gait_analyzer.joint_angle_by_velocity(result_dir=analyze_result_dir)
