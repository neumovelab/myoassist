# MyoAssist
## Pure RL
```bash
.venv\Scripts\python.exe myoassist_rl/rl_train/train_ppo.py --config_file_path myoassist_rl/rl_train/train_configs/base.json
```
## Imitation Learning
```bash
.venv\Scripts\python.exe myoassist_rl/rl_train/train_ppo.py --config_file_path myoassist_rl/rl_train/train_configs/imitation.json
```
### Imitation Learning with Dephy

- separated net
```bash
.venv\Scripts\python.exe myoassist_rl/rl_train/train_ppo.py --config_file_path myoassist_rl/rl_train/train_configs/imitation_dephy_separated_net.json
```
- uni net
```bash
.venv\Scripts\python.exe myoassist_rl/rl_train/train_ppo.py --config_file_path myoassist_rl/rl_train/train_configs/imitation_dephy_uni_net.json
```

## Train with pre-trained policy
```bash
--config.env_params.prev_trained_policy_path [path to previous trained policy]
```
reset value network
```bash
--config.custom_policy_params.reset_value_net_after_load
```
## Test training with rendering (Add this line after command)
```bash
--config.env_params.num_envs 1 --config.ppo_params.n_steps 32 --config.ppo_params.batch_size 16 --config.logger_params.logging_frequency 1 --config.logger_params.evaluate_frequency 2 --flag_rendering
```
## Enable Lumbar joint
```bash
--config.env_params.enable_lumbar_joint
```
## disable Lumbar joint
```bash
--no-config.env_params.enable_lumbar_joint
```