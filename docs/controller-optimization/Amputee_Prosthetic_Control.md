# Amputee and Prosthetic Control

This page shows how to optimize the reflex controller on an amputee model that wears a prosthetic device. It covers three features: the amputee reflex mode (`--reflex_mode amp`), prosthetic ankle stiffness optimization (`--optimize_stiffness`) for a passive foot, and the impedance controller for a powered prosthesis (`--optimize_impedance`).

## Amputee models

An amputee model is a standard leg MSK composed with a prosthetic device. The device performs the amputation. It removes the muscles and bones below the amputation, reshapes the residual limb, and attaches the prosthesis. You do not build a separate amputee MSK model.

The framework ships four prosthetic devices:

| Device | Type | Amputation |
|--------|------|------------|
| `KFoot_L1` | Passive foot | Transtibial (below knee), right |
| `NEUankle_L1` | Powered ankle | Transtibial (below knee), right |
| `OpenSourceLeg_A_L1` | Powered ankle (Open-Source Leg) | Transtibial (below knee), right |
| `OpenSourceLeg_KA_L1` | Powered knee and ankle (Open-Source Leg) | Transfemoral (above knee), right |

The transtibial devices remove the leg below the knee and attach to the residual tibia. The transfemoral `OpenSourceLeg_KA_L1` removes the leg below the hip and attaches to the residual femur.

Compose an amputee environment the same way as any other. Give a leg MSK key and a prosthetic device key:

```json
{ "msk": "myolegs22", "device": "KFoot_L1" }
```

Run `python -m assist_sim list` for the valid keys. See [Defining an Environment](../getting-started/defining-an-environment.md).

## Amputee reflex mode

The standard reflex controller drives both legs from one muscle set. An amputee model has fewer muscles on the prosthetic side, so the standard controller does not run on it. The amputee reflex mode solves this.

Set `--reflex_mode amp` to run the reflex controller on an amputee model. This mode does two things:

1. It uses the bilateral layout. Each leg gets its own reflex parameter block. See [Reflex Control](Reflex_Control_Overview.md) for the layout.
2. It tolerates the prosthetic side. It skips the reflex terms for muscles that the amputation removed. It reads a prosthetic joint in place of each biological joint that the amputation removed. For a spring foot, the ankle angle is the sum of the dorsiflexion and plantarflexion joints. For a powered ankle or knee, it is the angle of the one prosthetic joint. It also handles the absent toe joint and the prosthetic foot placement.

Use `amp` only with an amputee device. For an intact model, use `bilat` for independent legs, or the default symmetric mode.

The `amp_kfoot` example runs the 22-muscle model with the passive K-Foot:

```bash
cd ctrl_optim
python run_optim.py amp_kfoot
```

## Prosthetic ankle stiffness optimization

A passive prosthetic foot has a spring ankle. The `KFoot_L1` foot uses two spring joints on one axis. `df_ankle_angle_r` carries dorsiflexion. `pf_ankle_angle_r` carries plantarflexion. Each joint has its own stiffness.

Set `--optimize_stiffness` to add these two stiffnesses to the search. The flag appends two parameters to the CMA-ES vector: one for plantarflexion, then one for dorsiflexion. The optimizer tunes them together with the reflex controller.

How it works:

- The two parameters are normalized to `[0, 1]`.
- On each reset, the framework denormalizes them and writes `model.jnt_stiffness` for the two ankle joints. It edits the live model. It does not recompile.
- The stiffness ranges are 30 to 300 Nm/rad for plantarflexion, and 100 to 1000 Nm/rad for dorsiflexion.
- The two parameters are the last two entries of the parameter vector.

Only the passive `KFoot_L1` foot has the two spring-ankle joints, so `--optimize_stiffness` applies to it. The powered devices (`NEUankle_L1`, `OpenSourceLeg_A_L1`, `OpenSourceLeg_KA_L1`) drive a motor instead of a spring, so this flag does not apply to them. Use `--optimize_impedance` for them (see the next section).

The `kfoot_stiffness` example adds stiffness optimization to the amputee reflex:

```bash
cd ctrl_optim
python run_optim.py kfoot_stiffness
```

## Powered prosthesis impedance control

A powered prosthesis has a motor at the knee, the ankle, or both. Set `--optimize_impedance` to drive these motors with a four-phase finite-state impedance controller, and to add its gains to the search. The controller follows the `MyoOSLController` of MyoSuite, which uses the defaults of the Open-Source Leg project. The code is in `ctrl_optim/ctrl/prosthetic/impedance_ctrl.py`.

The controller is not specific to one device. It finds the powered prosthetic joints in the composed model. A powered prosthetic joint is a knee or ankle joint of the device that a motor (a non-muscle actuator) drives, and that replaces a biological joint that the amputation removed. An exoskeleton motor on an intact joint does not match.

| Device | Powered joints | Torque capacity | Parameters |
|--------|----------------|-----------------|------------|
| `OpenSourceLeg_A_L1` | Ankle | 168 Nm | 12 |
| `NEUankle_L1` | Ankle | 50 Nm | 12 |
| `OpenSourceLeg_KA_L1` | Knee and ankle | 142 Nm (knee), 168 Nm (ankle) | 24 |

The torque capacity is the actuator gain times the largest control magnitude of the actuator.

### Phases and transitions

The controller has four phases. They always follow in this order: early stance, late stance, early swing, late swing, then early stance again. The controller moves at most one phase per control step. It moves to the next phase when the sensors meet the transition condition of the current phase:

| Phase | Next phase when |
|-------|-----------------|
| Early stance | The load is above 0.25 body weight **and** the ankle dorsiflexion is more than 6 deg |
| Late stance | The load is below 0.15 body weight |
| Early swing | The knee flexion is more than 50 deg, **or** the knee flexion velocity is less than 3 deg/s |
| Late swing | The load is above 0.4 body weight, **or** the knee flexion is less than 30 deg |

The load is the vertical ground force under the prosthetic foot, in body weights. A powered joint gives its own angle. For a transtibial device (a powered ankle only), the knee angle comes from the knee of the person on the prosthetic side. For the transfemoral device, it comes from the prosthetic knee.

The thresholds are the `initDefaults` values of `MyoOSLController`. They stay fixed during the optimization. `MyoOSLController` goes to the next phase when one condition is true. Here, the early-stance transition needs both conditions, as in the state machine of the Open-Source Leg project. With one condition, early stance starts in late swing, when the knee flexion becomes less than 30 deg. The ankle target of late swing (15 deg) is above the 6 deg threshold, so the controller then goes through all four phases in the air before heel strike.

The first phase follows the start pose of the reflex controller. With `walk_left`, the right leg starts in early swing.

### Torque law

In each phase, each powered joint has a stiffness K, a damping B, and a target angle theta_eq. The joint torque is:

```
tau  = K * (theta_eq - theta) - B * dtheta    (clipped to the torque capacity)
ctrl = sign * tau / gain
```

The controller uses one angle convention for all devices. Knee flexion is positive, and 0 is full extension. Ankle dorsiflexion is positive, and 0 is neutral. The model joints do not all use this convention. For example, the biological knee of the 22- and 26-muscle models has negative flexion. The controller does not assume a sign for each device. It measures the sign (+1 or -1) from the model: hip flexion, knee extension, and ankle dorsiflexion turn about the same lateral axis. The prosthetic joints of the three shipped devices already use the controller convention.

The phases update once per control step (20 ms with `--delayed 0`). The torque law runs before each physics step (2 ms), with the phase of that control step. A motor driver also runs its impedance loop faster than its state machine. The prosthetic foot is light (about 0.003 kg m2 about the ankle). A torque that stays constant for a full control step is not an impedance on this foot. In a test, such a torque moved the unloaded foot far past its range of motion.

### Parameters

The flag adds 12 parameters per powered joint to the end of the CMA-ES vector: 12 for a powered ankle, 24 for a powered knee and ankle. The order is joint (knee before ankle), then phase, then K, B, and theta_eq.

Each parameter is normalized to `[0, 1]`, as the other device parameters are. On each reset, the framework maps it linearly onto its range:

| Gain | Knee range | Ankle range |
|------|------------|-------------|
| Stiffness K | 0 to 400 Nm/rad | 0 to 400 Nm/rad |
| Damping B | 0 to 10 Nm s/rad | 0 to 1 Nm s/rad |
| Target angle theta_eq | The joint range of the device | The joint range of the device |

A new optimization starts these parameters at the `initDefaults` gains. If you load an amputee reflex result that does not have these parameters (`--param_path`), the framework adds the default gains to it.

You cannot use `--optimize_impedance` together with `--ExoOn 1` or `--optimize_stiffness`.

### Examples

There is one example for each powered device. Each example uses the amputee reflex mode on the 22-muscle model:

```bash
cd ctrl_optim
python run_optim.py osl_a_impedance      # OpenSourceLeg_A_L1, powered ankle
python run_optim.py osl_ka_impedance     # OpenSourceLeg_KA_L1, powered knee and ankle
python run_optim.py neuankle_impedance   # NEUankle_L1, powered ankle
```

`run_eval.py` reads `--optimize_impedance` from the training configuration in the results directory, so it replays the run with the same controller. The prosthesis removes the biological ankle (and the biological knee, for a transfemoral device). The evaluation figure shows no curve for these joints.

## Reinforcement learning on the same models

The reflex controller is not the only way to drive an amputee model. `rl_train` ships one
imitation-training config per prosthetic device, generated by `tools/make_prosthesis_configs.py`:

```bash
python rl_train/run_train.py \
  --config_file_path rl_train/train/train_configs/prosthesis/imitation_22_OpenSourceLeg_A_L1_h128_d32_actpen10.json
```

The amputation changes the muscle set and the joint names, so those configs cannot be derived from
the intact ones by swapping `device_key`. The reference motion is also a healthy walker, and its
prosthetic-side joints are dropped rather than remapped. See the
[directory README](https://github.com/neumovelab/myoassist/blob/main/rl_train/train/train_configs/prosthesis/README.md)
for what differs and why.

## Notes

- The amputee mode is verified in 2D on the K-Foot and on the three powered devices. The 3D lineage (`myolegs26`) also composes and runs, with and without the impedance controller.
- The impedance controller does not run on the 80-muscle model (`myolegs`). With a prosthetic device, that composed model has no right ankle joint-limit sensor (`r_ankle_sensor`). The transfemoral model also has no right knee joint-limit sensor (`r_knee_sensor`). The cost function reads these sensors.
- An amputee gait is asymmetric by nature. The symmetry cost still applies, so read its value with that in mind. See [Cost Functions](Understanding_Cost.md).
- The prosthetic foot bears load during its own stance phase. At the `walk_left` start pose the prosthetic (right) leg trails, so its ground force reads near zero at that instant.
