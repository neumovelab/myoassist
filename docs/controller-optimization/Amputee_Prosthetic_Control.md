# Amputee and Prosthetic Control

This page shows how to optimize the reflex controller on an amputee model that wears a prosthetic device. It covers two features: the amputee reflex mode (`--reflex_mode amp`) and prosthetic ankle stiffness optimization (`--optimize_stiffness`).

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
2. It tolerates the prosthetic side. It skips the reflex terms for muscles that the amputation removed. It reads the prosthetic ankle as the sum of its dorsiflexion and plantarflexion joints. It also handles the absent toe joint and the prosthetic foot placement.

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

Only the passive `KFoot_L1` foot has the two spring-ankle joints, so `--optimize_stiffness` applies to it. The powered ankles (`NEUankle_L1`, `OpenSourceLeg_A_L1`, `OpenSourceLeg_KA_L1`) drive a motor instead of a spring, so this flag does not apply to them.

The `kfoot_stiffness` example adds stiffness optimization to the amputee reflex:

```bash
cd ctrl_optim
python run_optim.py kfoot_stiffness
```

## Notes

- The amputee mode is verified in 2D on the K-Foot. The 3D lineage (`myolegs26`) also composes and runs.
- An amputee gait is asymmetric by nature. The symmetry cost still applies, so read its value with that in mind. See [Cost Functions](Understanding_Cost.md).
- The prosthetic foot bears load during its own stance phase. At the `walk_left` start pose the prosthetic (right) leg trails, so its ground force reads near zero at that instant.
