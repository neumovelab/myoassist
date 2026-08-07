# Evaluating Results

This guide explains how to evaluate a finished Controller Optimization (CO) run
with the central evaluation pipeline: `ctrl_optim/run_eval.py` →
`ctrl_optim/eval/gait_evaluator.py`.

## Overview

`run_eval` takes an optimized reflex controller (a `_Best.txt` / `_BestLast.txt`
parameter file from an `optim_results` folder), rolls it out in the same composed
environment it was optimized in, and produces:

- **`gait_evaluated_data.json`** — the full rollout in the RL `GaitData` schema
  (joint `qpos`/`qvel`, actuator force/ctrl, sensor traces, target velocity), so the
  RL analyzers can consume CO output.
- **`composite.png`** (and a matching `.svg`) — a single summary figure.
- **`replay.mp4`** — an optional follow-camera replay video.

The rollout is driven by `CtrlOptimGaitEvaluator` (config `CtrlOptimEvalConfig`);
the figure is assembled by `myoassist_utils/eval_utils.py`
(`build_composite` / `CompositeInputs`) — the *same* builder the RL eval uses; the
follow camera comes from `ctrl_optim/eval/camera_setup.py`. The legacy Tkinter GUI
eval has been removed.

## Running an evaluation

Run from the repository root as a module.

```bash
# With a JSON config
python -m ctrl_optim.run_eval --config ctrl_optim/eval/configs/example_config.json

# Or point directly at a results directory (CLI flags fill the rest)
python -m ctrl_optim.run_eval --results-dir ctrl_optim/results/preoptimized/exo_4param_125_0729_1339
```

Given a results directory, `run_eval` auto-locates the parameter file
(`*_Best.txt` or `*_BestLast.txt`, chosen by `param_type`) and the CMA-ES pickle
(`*_Pickle.pkl`) inside it. A pop-out window shows the composite figure unless you
pass `--no-show`.

## JSON configuration

Copy `ctrl_optim/eval/configs/example_config.json`, rename it, and fill in your
values. Any field with a default can be omitted. Keys the pipeline reads:

| Key | Meaning | Default |
|-----|---------|---------|
| `results_dir` | folder holding the optimized run | (required) |
| `param_type` | `"Best"` or `"BestLast"` — which param file to load | `"Best"` |
| `param_file` | explicit param `.txt`; auto-located from `param_type` if null | auto |
| `pkl_file` | explicit CMA-ES `_Pickle.pkl`; auto-located if null | auto |
| `output_dir` | where outputs are written | `<results_dir>/eval_output` |
| `sim_time` | rollout length (s) | `10.0` |
| `target_velocity` | target walking speed (m/s), used for readouts | `1.25` |
| `mode` | `"2D"` or `"3D"` | `"2D"` |
| `init_pose` | starting keypose | `"walk_left"` |
| `delayed` | biological neural delays | `false` |
| `exo_bool` | exoskeleton enabled | `true` |
| `fixed_exo` | hold exo params fixed | `false` |
| `use_4param_spline` | 4-param vs n-point exo spline | `true` |
| `max_torque` | max exo torque | `1.0` |
| `n_points` | n-point spline points (when not 4-param) | `4` |
| `msk_key` | human MSK registry key | `"myolegs22"` |
| `device_key` | assist_sim device registry key | `"Tutorial_L1"` |
| `terrain` | terrain spec (path or inline); drives the slope | null → flat |
| `camera_speed` / `camera_distance` / `camera_elevation` / `camera_height` / `camera_azimuth` | follow-camera setup | see config |
| `render_width` / `render_height` | render resolution | `1920` × `960` |
| `show_actuators` | draw actuators in the render | `true` |
| `export_video` | also write `replay.mp4` | `false` |
| `video_fps` | replay frame rate | `100` |

The environment is defined the same way as everywhere else in MyoAssist — a
`{msk, device, terrain}` env-spec of raw registry keys (see
**[Defining an Environment](../getting-started/defining-an-environment.md)**). The
course grade is the single source of truth: the incline is derived from a
`slope` `terrain`, not a separate flag.

## CLI overrides

Instead of (or on top of) a config, you can pass flags; an explicit flag always
overrides the JSON value:

```bash
python -m ctrl_optim.run_eval --results-dir <dir> \
    --param-type BestLast --sim-time 20 --target-velocity 1.25 \
    --mode 2D --exo-bool true --export-video --no-show
```

Available flags: `--config`, `--results-dir`, `--param-file`, `--pkl-file`,
`--output-dir`, `--param-type {Best,BestLast}`, `--sim-time`, `--target-velocity`,
`--mode {2D,3D}`, `--init-pose`, `--exo-bool`, `--export-video`, `--no-show`.

## The composite figure

`composite.png` is a black-and-white multi-panel summary:

- **Environment snapshot** — a mid-rollout skeleton render.
- **CMA-ES fitness** — best / median / worst cost per generation, read from
  `outcmaes/*_fit.dat` in the results folder (falling back to `es.fit.hist` in the
  pickle for older runs).
- **Kinematics** — segmented right-leg hip / knee / ankle angles (mean ± SD)
  overlaid on the shared human gait reference (`rl_train/reference_data/segmented.npz`).
- **Kinetics** — active joint moments (Nm) per gait cycle.
- **Activation** — a muscle-activation grid plus exo torque.
- **Timeseries** — right-leg joint angles, pelvis height, and foot-contact sensors.

## Output structure

By default outputs land in `<results_dir>/eval_output/`:

```
<results_dir>/eval_output/
├── gait_evaluated_data.json   # rollout data in the RL GaitData schema
├── composite.png              # summary figure
├── composite.svg              # vector version of the same figure
└── replay.mp4                 # only when export_video / --export-video is set
```

## Quick visualization

For a quick video without the full analysis figure, use the `run_ctrl.py`
simulation script instead. See **[Running Reflex Control](Running_Reflex_Control.md)**.
