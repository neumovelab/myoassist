from rl_train.utils.train_log_handler import TrainLogHandler
import os
import json
from rl_train.utils.data_types import DictionableDataclass
from rl_train.analyzer.train_log_analyzer import TrainLogAnalyzer
from rl_train.train.train_configs.config import TrainSessionConfigBase
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass



from rl_train.analyzer.gait_evaluate import ImitationGaitEvaluator


from rl_train.analyzer.gait_analyze import GaitAnalyzer
from rl_train.analyzer.gait_evaluate import GaitData
from rl_train.utils.train_checkpoint_data_imitation import ImitationTrainCheckpointData
from enum import Enum
import numpy as np

class TrainAnalyzer:
    class SequenceElement(Enum):
        REWARD_PLOT = 0
        EVALUATE = 1
        ANALYZE = 2
        # Should have detail analysis

    def analyze_in_sequence(self, log_dir:str, show_plot:bool):
        with open(os.path.join(log_dir, "session_config.json"), 'r') as f:
            config_dict = json.load(f)
            config = DictionableDataclass.create(TrainSessionConfigBase, config_dict)

            if config.env_params.env_id == 'myoAssistLeg-v0':
                config = DictionableDataclass.create(TrainSessionConfigBase, config_dict)
            elif config.env_params.env_id in ['myoAssistLegImitation-v0', 'myoAssistLegImitationExo-v0']:
                # print("Imitation train session config")
                config = DictionableDataclass.create(ImitationTrainSessionConfig, config_dict)
                # print("\nAll fields in RewardWeights:")
                # for field in fields(config.env_params.reward_keys_and_weights):
                #     print(f"{field.name}: {getattr(config.env_params.reward_keys_and_weights, field.name)}")

        for (eval_idx, evaluate_param) in enumerate(config.evaluate_param_list):
            log_handler = TrainLogHandler(log_dir)
            log_handler.load_log_data(ImitationTrainCheckpointData)

            train_analyzer_report = {
                "num_timesteps":log_handler.log_datas[-1].num_timesteps,
                "exceptions":[],
            }

            analyze_result_dir = os.path.join(log_dir,f"analyze_results_{log_handler.log_datas[-1].num_timesteps}_{eval_idx:02d}")
            if not os.path.exists(analyze_result_dir):
                os.makedirs(analyze_result_dir)

            # plot train logs
            
            train_log_analyzer = TrainLogAnalyzer(log_handler)
            train_log_analyzer.plot_reward(result_dir=analyze_result_dir, show_plot=show_plot)

            train_log_analyzer.plot_reward_dict(result_dir=analyze_result_dir, show_plot=show_plot, mult_weights=False)
            train_log_analyzer.plot_reward_dict(result_dir=analyze_result_dir, show_plot=show_plot, mult_weights=True)
            # train_log_analyzer.plot_reward_weights(result_dir=analyze_result_dir)


            
        
            config.env_params.min_target_velocity = evaluate_param["min_target_velocity"]
            config.env_params.max_target_velocity = evaluate_param["max_target_velocity"]
            from rl_train.envs.myoassist_leg_base import MyoAssistLegBase

            gait_data_name = f"gait_evaluated_data_{eval_idx:02d}.json"

            if os.path.exists(os.path.join(analyze_result_dir, gait_data_name)):
                user_input = input(f"Regenerate evaluate data? ({gait_data_name}) (y/n(anything))")
            else:
                user_input = "y"
            is_regen_evaluating_data = True if user_input == "y" else False


            gait_evaluator = ImitationGaitEvaluator(log_handler, config)
            gait_evaluator.load_reference_data()
            gait_evaluator.initialize_env()

            if is_regen_evaluating_data:
                gait_data_path = gait_evaluator.evaluate(result_dir=analyze_result_dir,
                                                        file_name=gait_data_name,
                                                        velocity_mode=MyoAssistLegBase.VelocityMode[evaluate_param["velocity_mode"]],
                                                        target_velocity_period=evaluate_param["target_velocity_period"],
                                                        max_timestep=evaluate_param["num_timesteps"]
                                                        )
            else:
                # it will(should) not happen during the training
                gait_data_path = os.path.join(analyze_result_dir, gait_data_name)
                
            
            gait_data = GaitData()
            gait_data.read_json_data(gait_data_path)
            
            # Only load reference data and perform analysis for imitation learning environments
            if config.env_params.env_id in ['myoAssistLegImitation-v0', 'myoAssistLegImitationExo-v0']:
                try:
                    segmented_ref_data = np.load("reference_data/segmented.npz", allow_pickle=True)
                    segmented_ref_data = {key: segmented_ref_data[key] for key in segmented_ref_data.files}
                    exception_report_list = self.analyze(gait_data, segmented_ref_data, analyze_result_dir, show_plot)
                except FileNotFoundError:
                    print("Warning: Reference data file not found. Skipping gait analysis.")
                    exception_report_list = []
            else:
                # For base environments, skip reference data analysis
                print("Base environment detected. Skipping reference data analysis.")
                exception_report_list = []

            train_analyzer_report["exceptions"].extend(exception_report_list)


            # for cam_type in ["average_speed", "follow"]:
            file_name = f'replay_{eval_idx:02d}.mp4'
            frames = gait_evaluator.replay(gait_data_path, os.path.join(analyze_result_dir, file_name),
                                                cam_distance=evaluate_param["cam_distance"],
                                                max_time_step=evaluate_param["num_timesteps"],
                                                use_activation_visualization=evaluate_param["visualize_activation"],
                                                cam_type=evaluate_param["cam_type"],
                                                use_realtime_floating=False
                                                )
            
            train_analyzer_report_path = os.path.join(analyze_result_dir, "train_analyzer_report.json")
            with open(train_analyzer_report_path, 'w') as f:
                json.dump(train_analyzer_report, f)
        
    def analyze(self, gait_data, segmented_ref_data, analyze_result_dir, show_plot:bool):
        exception_report_list = []

        gait_analyzer = GaitAnalyzer(gait_data, segmented_ref_data, show_plot)

        try:
            gait_analyzer.plot_entire_result(result_dir=analyze_result_dir, is_right_foot_based=True)
        except Exception as e:
            exception_report_list.append({"function_name": "plot_entire_result_right", "exception": str(e)})

        try:
            gait_analyzer.plot_entire_result(result_dir=analyze_result_dir, is_right_foot_based=False)
        except Exception as e:
            exception_report_list.append({"function_name": "plot_entire_result_left", "exception": str(e)})

        try:
            gait_analyzer.plot_segmented_kinematics_result(result_dir=analyze_result_dir)
        except Exception as e:
            exception_report_list.append({"function_name": "plot_segmented_kinematics_result", "exception": str(e)})

        try:
            gait_analyzer.plot_left_right_comparison(result_dir=analyze_result_dir)
        except Exception as e:
            exception_report_list.append({"function_name": "plot_left_right_comparison", "exception": str(e)})

        try:
            gait_analyzer.plot_right_ref_comparison(result_dir=analyze_result_dir)
        except Exception as e:
            exception_report_list.append({"function_name": "plot_right_ref_comparison", "exception": str(e)})

        try:
            gait_analyzer.plot_contact_data(result_dir=analyze_result_dir)
        except Exception as e:
            exception_report_list.append({"function_name": "plot_contact_data", "exception": str(e)})

        try:
            gait_analyzer.plot_segmented_muscle_data(result_dir=analyze_result_dir, is_plot_right=True)
        except Exception as e:
            exception_report_list.append({"function_name": "plot_segmented_muscle_data_right", "exception": str(e)})

        try:
            gait_analyzer.plot_segmented_muscle_data(result_dir=analyze_result_dir, is_plot_right=False)
        except Exception as e:
            exception_report_list.append({"function_name": "plot_segmented_muscle_data_left", "exception": str(e)})

        return exception_report_list
