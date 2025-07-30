python myoassist_rl/rl_train/train_ppo.py ^
--config_file_path myoassist_rl/rl_train/train_configs/imitation_tutorial_22_separated_net_partial_obs.json ^
--config.env_params.num_envs 1 --config.ppo_params.n_steps 256 ^
--config.ppo_params.batch_size 256 --config.logger_params.logging_frequency 1 ^
--config.logger_params.evaluate_frequency 2 --flag_rendering ^
--config.env_params.terrain_type "slope" ^
--config.env_params.terrain_params "-0.2"