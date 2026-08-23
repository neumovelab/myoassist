# Prosthesis — imitation training on amputee models

Four configs, one per prosthetic device in the assist_sim registry. These are the counterpart to
[`device_sweep/`](../device_sweep/README.md), not a row in it: there the only thing that varies is
a drop-in exo on an intact 22-muscle model, whereas here the device performs an amputation and
changes the human model itself.

## Train

```bash
python rl_train/run_train.py \
  --config_file_path rl_train/train/train_configs/prosthesis/imitation_22_OpenSourceLeg_A_L1_h128_d32_actpen10.json \
  --config.total_timesteps 30000000 \
  --config.env_params.num_envs 32
```

## Evaluate

Same as the exo sweep:

```bash
python -m rl_train.run_policy_eval rl_train/results/<train_session_...> --no-show --steps 1000 --regen
```

`tools/score_exo_policy.py` assumes a left/right exo pair, so it does not apply to a unilateral
device.

## Regenerate the configs

They are generated, not hand-written:

```bash
python tools/make_prosthesis_configs.py --human-net 128 --device-net 32 --muscle-activation-penalty 10
```

Nothing about the layout is tabulated in the generator: the muscle count, the actuator names, the
prosthetic joints that stand in for the amputated human ones, and the joint-limit sensors that
survive the amputation are all read from the composed model. `--help` lists the other options.

## The four devices

| device | amputation | actuators | obs | prosthetic DOF | policy |
|---|---|---|---|---|---|
| `OpenSourceLeg_A_L1` | transtibial, right | 18 muscle + 1 motor | 40 | `OSL_A_L1_osl_ankle_angle_r` | `HumanExoActorCriticPolicy` |
| `NEUankle_L1` | transtibial, right | 18 muscle + 1 motor | 40 | `NEUankle_L1_neuankle_ankle_angle_r` | `HumanExoActorCriticPolicy` |
| `OpenSourceLeg_KA_L1` | transfemoral, right | 15 muscle + 2 motor | 37 | `OSL_KA_L1_osl_knee_angle_r`, `..._ankle_angle_r` | `HumanExoActorCriticPolicy` |
| `KFoot_L1` | transtibial, right | 18 muscle + 0 motor | 42 | `KFoot_L1_df_ankle_angle_r`, `..._pf_...` | `HumanActorCriticPolicy` |

`KFoot_L1` is a passive spring foot with no motor, so there is nothing for a device sub-policy to
drive. Its config runs the muscle-only `myoAssistLegImitation-v0` env and carries no
`net_indexing_info`, which the muscle-only policy does not read. The spring stiffnesses are fixed
here; `ctrl_optim` can optimize them (see
[Amputee and Prosthetic Control](../../../../docs/controller-optimization/Amputee_Prosthetic_Control.md)).

## What differs from an exo config, and why

**The observation layout is per-device.** The amputation removes muscles, so the `act` block is 18
or 15 entries rather than 22, and every absolute index after it shifts. `myolegs22 +
OpenSourceLeg_A_L1` has no `ankle_angle_r` and no `mtp_angle_r` at all.

**The prosthetic DOF replaces the human one in observation.** `ankle_angle_r` is substituted by the
device's own joint (and by *two* joints for the K-Foot, which carries a separate dorsiflexion and
plantarflexion spring), so the policy still sees the ankle it has.

**Forward progress is the dominant objective, not imitation.** This is the one weight that has to
differ from an intact config, and the reason is that "imitation" does not mean the same thing here.
On an intact model, tracking the reference *is* walking, so an imitation-dominant reward produces
gait — measured on the `Tutorial_L1` control, the imitation block is 84% of the weighted total
against forward's 3.7%, and it walks. On an amputee the imitation term covers only the pelvis and
the intact leg, so its optimum is reachable **while the prosthetic leg drags**. Measured on the
30M prosthesis runs: imitation 77.4%, forward 5.7%, and the learned policies either dragged the
prosthetic foot or folded the residual knee. `forward_reward` is therefore 20 here against the
template's 1.

**The imitation reward covers the pelvis, the intact leg, and the residual limb.** `short_reference_gait.npz` is
motion capture of an intact walker. The device joint is not the human joint it replaces — the OSL
ankle is limited to ±0.52 rad against the human's −1.13…0.35, and the transfemoral knee runs
[0, 2.09] against the human's [−2.53, 0], the opposite sign convention — so those keys are dropped
rather than remapped. The whole amputated side goes with them: without an ankle push-off the
residual limb's hip and knee cannot produce healthy kinematics either.

Three groups, three treatments:

| joints | imitation reward | termination check |
|---|---|---|
| pelvis, intact leg | template weight | yes, at 0.4 rad |
| residual limb (own hip; own knee, transtibially) | weak posture weight (0.2) | **no** |
| replaced by the device (ankle; knee+ankle, transfemorally) | **no** | **no** |

The residual limb needs the reward term: with the whole side dropped, nothing in the reward
referred to those joints and the policy folded the knee. It must not be in the termination check:
a residual limb without an ankle push-off cannot produce healthy swing-phase kinematics, and
`knee_angle_r` was the single most frequent cause of episode termination on the first runs. The
two are separable because `env_params.out_of_trajectory_joint_keys` narrows what
`MyoAssistLegImitation.step` watches; left empty it falls back to the whole imitation dict, which
is what the intact configs still do.

`out_of_trajectory_threshold` is 0.4 here against the intact configs' 0.2, because the intact leg
compensates for the missing push-off and does not track a healthy walker as tightly.

`reference_data_keys` still *places* the residual limb from the reference at reset. Dropping a
joint from the reward is not the same as starting it from an arbitrary pose.

This is measured. The first 30M runs kept the affected side in the imitation dict at threshold 0.2:

| run | eval survival | distance | prosthetic-foot strikes |
|---|---|---|---|
| `OpenSourceLeg_A_L1` | 23 steps (0.8 s) | 1.1 m | 0 |
| `NEUankle_L1` | 22 steps (0.7 s) | 0.4 m | 0 |
| `Tutorial_L1` (intact control) | 1000 steps (33 s, capped) | 39.4 m | 62 |

`knee_angle_r` was the largest tracking error and the most frequent termination cause, and both
prosthesis runs flattened at ~15M steps (episode length 4.1 → 4.7) while the intact control kept
improving (4.25 → 7.51). Removing the trajectory wall at evaluation time did *not* rescue the
trained policy — 12/12 episodes then fell instead — so the wall is not the whole story on its own;
it is the reward and the terminator together, which is why both are changed.

**Both OpenSourceLeg compositions open 9 cm in the air, and this compensates for it.**

`myoassist_utils/compose.py` seats a composed model on the ground by finding its lowest point, and
`_model_ground_candidates` estimates a primitive's underside as `centre_z − max(geom_size)`. For a
capsule `size` is `(radius, half-length)` about its *local z axis*, so a horizontal capsule reaches
only its radius below centre. The OpenSourceLeg foot contacts are capsules of radius 0.012 m and
half-length 0.11 m laid flat, so the estimate puts their underside 9.8 cm too low and the seater
lifts the whole model onto a point that is not there: no ground contact at the keyframe at all, the
real capsules floating at +0.093 m, the intact foot at +0.088 m. Every other device's foot contact
is a mesh, which is measured from its transformed vertices and is unaffected.

That pipeline is shared with the CO framework, so the fix belongs there and not here. On the RL
side this is compensated instead, in two places:

* `MyoAssistLegImitation._standing_pelvis_height` measures the standing height by lowering the
  pelvis when the keyframe reports no ground contact, and `_height_corrected_reference` then
  shifts the reference down by 8.3 cm. Without it every episode began with the pelvis above the
  height at which the feet reach the floor — `reset` takes its pose from the reference — so the
  model free-fell into each episode and the imitation term rewarded holding it up there.
* `safe_height` is derived per composition as `standing height − 0.21`, giving 0.613 for the two
  OpenSourceLeg models against 0.7 elsewhere. It is an absolute `pelvis_ty`, so the shipped 0.7
  would leave a model standing at 0.823 only 0.12 m of fall margin against the intact configs'
  0.21 m.

Both are keyed off the measured geometry, so if the compose estimate is ever corrected upstream
they go inert on their own: the keyframe will report contact, no shift will be applied, and the
derived `safe_height` returns to 0.7.

**The device command is capped, and its effort is priced.** A device actuator here can overpower
its own joint limit. `OpenSourceLeg_A_L1` drives its ankle with 168 N·m into a joint with
`damping=0` and `armature=0`; a MuJoCo joint limit is a soft constraint whose stiffness scales with
the DOF's effective inertia, and with none the joint runs far past its own ±0.52 rad range and
stays there. Measured on a trained policy, holding everything else fixed:

| | survival | ankle outside its limits | worst overshoot |
|---|---|---|---|
| device on, full 168 N·m | 34 steps | 92% of steps | +4.29 rad |
| device off | 50 steps | 31% | +0.31 rad |
| +armature 0.05, +damping 2, device on | 33 steps | 57% | +1.59 rad |
| +armature 0.05, +damping 2, device off | 53 steps | 26% | +0.30 rad |

Ground reaction alone leaves it 0.31 rad over, the same regime as the intact model's passive toe
joint (0.36 rad) — which walks fine. The motor is what breaks it, and adding joint dynamics does
not help while the motor is on. For reference the intact model's own load-bearing ankle never
leaves its range at all (0.0% of steps).

Two knobs follow. `device_ctrl_scale` (0.15 here, 1.0 everywhere else) narrows `actuator_ctrlrange`
at setup so the policy commands at most 25 N·m; across a cap sweep that is where the overshoot
reaches 0.47 rad. `exo_activation_penalty` is 1.0 rather than 0, so the torque is no longer free —
at zero the command simply sat at saturation.

Both are workarounds on the RL side. The real fix is in the device model: the ankle needs enough
inertia to absorb its own actuator, or the device needs a control rate above this env's 30 Hz.

**`reset_keyframe_joint_keys` names the prosthetic joints.** `reset` seeds the next episode from
`sim.data.qpos`, so a DOF the reference does not write carries its value across the episode
boundary — normally the value it held while the model was falling. On an intact model that is only
the passive toe joints. Here it is the joint the device actuator drives: measured on
`OpenSourceLeg_A_L1` before this field existed, successive episodes started at +1.43 and +1.94 rad
against a ±0.52 rad limit. The field is opt-in so the intact configs keep the reset behaviour their
published results were trained under.

**`mirror_coef` is 0.** The mirror penalty needs a left/right actuator permutation, and
`rl_train/train/policies/mirror.py` raises on an asymmetric model — correctly, since there is no
partner for `soleus_l`. The weight-shared per-side network (`exo_actor_r`/`exo_actor_l`) is out for
the same reason; a single `exo_actor` drives the one-sided device.

**`joint_limit_sensor_keys` is set explicitly.** Every prosthesis here drops `r_mtp_sensor` with the
amputated forefoot, and the constraint-force penalty reads each name unguarded, so the class default
would raise on the first step.

## Imitation as a scaffold

The shipped configs anneal the imitation weights to zero between 20% and 60% of the run, ramping
`forward_reward` and `muscle_activation_penalty` up over the same window, and train for 100M steps
rather than 30M.

The reason is that imitation cannot be the objective here. The reference is a healthy walker and
says nothing about the affected side, so what survives the filtering is a term that describes the
pelvis and the intact leg. Its optimum is reachable without walking: on the 30M runs
`OpenSourceLeg_KA_L1` reached 34 m by hopping on the intact leg, with the prosthetic foot recording
0.00 N of ground force in 20 of 20 episodes.

What remains once imitation is gone is forward progress against effort, and that pair can tell the
two apart. Measured on the 30M policies, muscle effort per metre travelled:

| policy | effort per metre |
|---|---|
| `OpenSourceLeg_KA_L1` hopping on one leg | 0.237 |
| `KFoot_L1` using the prosthesis | 0.117 |
| `NEUankle_L1` using the prosthesis | 0.110 |
| `Tutorial_L1` intact, walking | 0.131 |

Hopping is about twice as expensive per metre, so the effort term already disfavours it. It won
anyway at `forward_reward` 20 against `muscle_activation_penalty` 10, which is why the annealed
target is 5 against 10.

Once `qpos_imitation_rewards` reaches zero the out-of-trajectory check goes inert
(`MyoAssistLegImitation.step`): enforcing a reference the policy is no longer paid to follow would
keep the constraint that annealing exists to remove. `safe_height` carries termination from there.

This is opt-in per config. The intact configs declare no `reward_curriculum`, and a test asserts
they do not — the argument for annealing rests on the reference being unable to describe the model,
which is not true of them.

## Reading the results

An amputee gait is asymmetric by nature, so a symmetry-based score is not the right instrument. The
prosthetic side also carries no imitation term, which makes total reward not comparable against an
intact run — compare prosthesis configs against each other, not against the exo sweep.
