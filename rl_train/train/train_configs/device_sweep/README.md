# Device sweep — imitation training, one config per bilateral exo

Eight configs, one per bilateral powered device. They differ only in `device_key` and in the joint
the exo sub-policy observes, so any difference in outcome is attributable to the device.

## Train

```bash
python rl_train/run_train.py \
  --config_file_path rl_train/train/train_configs/device_sweep/imitation_22_Tutorial_L1_h128_e32_sidenet_mirror0p1_actpen10.json \
  --config.total_timesteps 30000000 \
  --config.env_params.num_envs 32
```

## Evaluate

```bash
python -m rl_train.run_policy_eval rl_train/results/<train_session_...> --no-show --steps 1000 --regen
python tools/score_exo_policy.py     rl_train/results/<train_session_...>/analyze_results_00
python tools/plot_kinematics_exo.py  rl_train/results/<train_session_...>/analyze_results_00 -o out.png
```

Use `--steps 1000` (~30 strides). The 200-step default is ~5 strides, too few for a per-phase
quantity such as when in the cycle the exo peaks.

## Regenerate the configs

They are generated, not hand-written:

```bash
python tools/make_device_sweep_configs.py --exo-shared-side-net --mirror-coef 0.1 \
  --human-net 128 --exo-net 32 --muscle-activation-penalty 10
```

`--out-dir` writes elsewhere. `python tools/make_device_sweep_configs.py --help` lists the other
options.

## The eight devices

| device | transmission | assisted joint | exo obs (qpos / qvel) |
|---|---|---|---|
| `Tutorial_L1` | joint | ankle | `[0,2]` / `[8,10]` |
| `DephyExoBoot_L1` | joint | ankle | `[0,2]` / `[8,10]` |
| `Humotech_L1` | joint | ankle | `[0,2]` / `[8,10]` |
| `OpenExo_L1` | joint | ankle | `[0,2]` / `[8,10]` |
| `STRIDE_L2` | tendon + linkage | ankle | `[0,2]` / `[8,10]` |
| `UTAnkleExo_L2` | tendon + linkage | ankle | `[0,2]` / `[8,10]` |
| `Hippo_L1` | joint | **hip** | `[2,4]` / `[10,12]` |
| `HMEDI_L1` | tendon | **hip** | `[2,4]` / `[10,12]` |

`python -m assist_sim list` offers 13 devices for `myolegs22`. The other five are a different
class: `Anatomics_L1` and `KFoot_L1` have no device actuator, and `NEUankle_L1`,
`OpenSourceLeg_A_L1` and `OpenSourceLeg_KA_L1` are unilateral prostheses with a truncated muscle
set, so they need their own action layout and cannot use the mirror penalty or the per-side network.

## The config

* **Weight-shared per-side exo network** (`exo_actor_r` / `exo_actor_l`) — one network applied to
  each leg with that leg's own inputs first, so `Exo_L(s) = Exo_R(mirror(s))` holds by construction.
  The `NET` arm of Abdolhosseini et al. 2019, *On Learning Symmetric Locomotion*.
* **`mirror_coef 0.1`** — Yu et al. 2018 mirror penalty on the policy (`rl_train/train/mirror_ppo.py`).
* **`muscle_activation_penalty 10`**, squared, and no device cost.

Rank candidates with `tools/score_exo_policy.py`, not with `mirror_loss`: a policy can lower the
mirror loss by driving both exo outputs toward zero.
