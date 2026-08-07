# Defining an Environment

In MyoAssist an **environment** is a human musculoskeletal (MSK) model + an
assistive **device** + a **terrain**, composed into a single MuJoCo model. The
*same* definition drives both pipelines — reflex **Controller Optimization (CO)**
and **Reinforcement Learning (RL)** — so you describe an environment once and run
it either way.

## The environment spec

An environment is three fields — **raw registry keys**, not free-form paths:

```json
{ "msk": "myolegs22", "device": "Humotech_L1", "terrain": { "terrain": "slope", "deg": 8 } }
```

| Field | What it is | Examples |
|-------|------------|----------|
| `msk` | human MSK model | `myolegs22`, `myolegs26` |
| `device` | assistive device | `DephyExoBoot_L1`, `Humotech_L1`, `OpenSourceLeg_A_L1`, `Tutorial_L1` |
| `terrain` | the ground (optional) | omit → flat; an inline config; or a terrains JSON path |

Run **`python -m assist_sim list`** to see every MSK / device that is installed
and which pairs are compatible.

### MSK models

| Key | Description |
|-----|-------------|
| `myolegs22` | 22-muscle **2D** (sagittal-plane) lower limb |
| `myolegs26` | 26-muscle **3D** lower limb |

The muscle count and 2D/3D control mode are **derived from `msk`** — you never set
them separately.

### Devices

Devices include ankle / knee exoskeletons (e.g. `DephyExoBoot_L1`, `Humotech_L1`,
`OpenExo_L1`, `UTAnkleExo_L2`, `HMEDI_L1`), robotic prosthetic legs (e.g.
`OpenSourceLeg_A_L1`, `OpenSourceLeg_KA_L1`), and a passive `Tutorial_L1` used for
demos and baselines. `python -m assist_sim list` prints the authoritative,
installed set; `python -m assist_sim validate <msk> <device>` checks one pair.

### Terrain

Leave `terrain` unset (or `null`) for a flat, effectively-infinite ground plane.
Otherwise give one of:

**A uniform surface** (one plane or one heightfield):

| `terrain` | Result |
|-----------|--------|
| `{ "terrain": "flat" }` | a flat plane |
| `{ "terrain": "slope", "deg": 8 }` | a constant 8° incline (a tilted plane) |
| `{ "terrain": "random", "amplitude": 0.06 }` | a rough heightfield, ≤ 6 cm relief |
| `{ "terrain": "sinusoidal", "amplitude": 0.05, "period": 1.0 }` | rolling waves |

**A tiled grid** — a [`myoassist_terrains`](https://github.com/neumovelab/myoassist.terrains)
config with a `grid` + per-cell `tiles` (types: `flat`, `slope`, `stairs`,
`pyramid_stairs`, `rough`, `boulders`, `stepping_stones`, `discrete_obstacles`,
`gap`), optionally filled by `randomization`. Give it inline or as a path to a
JSON file:

```json
"terrain": {
  "grid": { "rows": 1, "cols": 3, "tile_size": [4.0, 4.0] },
  "border": { "width": 0.5 },
  "tiles": [
    { "row": 0, "col": 0, "type": "flat",   "params": { "height": 0.0 } },
    { "row": 0, "col": 1, "type": "slope",  "params": { "angle_deg": 8.0, "axis": "x" } },
    { "row": 0, "col": 2, "type": "stairs", "params": { "n_steps": 5, "step_height": 0.1, "axis": "x" } }
  ]
}
```

The course **grade is the single source of truth**: a `slope` terrain *is* the
incline, and the evaluation camera, cost, and readouts derive the angle from it —
there is no separate slope flag.

> **Reflex CO is for steady-state locomotion.** A constant slope is fine, but the
> reflex controller is not meant to *optimize over* variable terrain (rough,
> stairs, mixed tiles). Use those with RL, or for visualization.

## Using an environment spec

### Controller Optimization (reflex)

```bash
# raw flags
python -m ctrl_optim.optim.train --msk myolegs22 --device Humotech_L1 \
    --terrain '{"terrain":"slope","deg":8}' --sim_time 20 -eff --ExoOn 1 ...

# or a shared env-spec file
python -m ctrl_optim.optim.train --env-spec docs/examples/env_exo_slope.json --sim_time 20 -eff --ExoOn 1 ...
```

There is no `--model`, `--musc_model`, or `--tgt_slope` — the model is defined by
`--msk`/`--device` and the grade by the terrain.

### Reinforcement Learning

Set the same three fields on `env_params` in your training-config JSON:

```json
"env_params": { "msk_key": "myolegs22", "device_key": "Humotech_L1", "terrain": null }
```

### Programmatically

```python
from myoassist_utils.env_spec import EnvSpec

spec = EnvSpec.load("docs/examples/env_exo_slope.json")
# or: EnvSpec(msk="myolegs22", device="Humotech_L1", terrain={"terrain": "slope", "deg": 8})

spec.validate()          # checks keys against the registry; raises with valid options on a bad key
xml = spec.compose()     # -> a loadable MuJoCo MJCF string
spec.compose(export_path="my_env.xml")   # also write a standalone, from_xml_path-loadable file
```

## Discovering and validating

```bash
python -m assist_sim list                       # every installed MSK / device + compatibility
python -m assist_sim validate myolegs22 Humotech_L1
```

`EnvSpec.validate()` does the same check in code and raises a `ValueError` listing
the valid options when a key is unknown or the MSK/device pair is incompatible.

## Ready-to-use examples

The [`docs/examples/`](../examples/) directory has runnable env-specs:

| File | Environment |
|------|-------------|
| `env_exo_flat.json` | `myolegs22` + `Humotech_L1`, flat |
| `env_exo_slope.json` | `myolegs22` + `Humotech_L1`, 8° slope |
| `env_prosthesis_rough.json` | `myolegs22` + `OpenSourceLeg_A_L1`, rough heightfield |
| `env_tiled_course.json` | `myolegs22` + `Humotech_L1`, a flat → slope → stairs tiled course |
| `env_tiled_random.json` | `myolegs22` + `OpenExo_L1`, a randomized 3×3 tiled grid |
