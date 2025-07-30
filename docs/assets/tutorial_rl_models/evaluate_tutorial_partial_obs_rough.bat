python myoassist_rl/rl_train/train_ppo.py ^
--config.env_params.terrain_type "random" ^
--config.env_params.terrain_params "0.03" ^
--config_file_path myoassist_rl/rl_train/train_configs/imitation_tutorial_22_separated_net_partial_obs.json ^
--config.env_params.prev_trained_policy_path docs/assets/tutorial_rl_models/train_session_20250728-161129_tutorial_partial_obs/session_name_models/session_name_step_19939328 ^
--flag_realtime_evaluate
