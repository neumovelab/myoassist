python myoassist_rl/rl_train/train_ppo.py ^
--config_file_path myoassist_rl/rl_train/train_configs/imitation_tutorial_22_separated_net_partial_obs.json ^
--config.total_timesteps 64 ^
--config.env_params.num_envs 1 --config.ppo_params.n_steps 16 ^
--config.ppo_params.batch_size 16 --config.logger_params.logging_frequency 1 ^
--config.logger_params.evaluate_frequency 1 --flag_rendering ^

@REM --config.env_params.terrain_type "random" ^
@REM --config.env_params.terrain_params "0.2"

@REM --config.env_params.terrain_type "slope" ^
@REM --config.env_params.terrain_params "0.2"