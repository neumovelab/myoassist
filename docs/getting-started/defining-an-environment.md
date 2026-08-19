# Defining an Environment

In MyoAssist, an **environment** is a human musculoskeletal (MSK) model, an assistive
**device**, and a **terrain**, composed into one MuJoCo model. The same definition
drives both pipelines: reflex **Controller Optimization (CO)** and **Reinforcement
Learning (RL)**.

## The environment spec

An environment has three fields. Each field is a **raw registry key**:

```json
{ "msk": "myolegs22", "device": "Humotech_L1", "terrain": { "terrain": "slope", "deg": 8 } }
```

| Field | What it is | Examples |
|-------|------------|----------|
| `msk` | human MSK model | `myolegs22`, `myolegs26` |
| `device` | assistive device | `DephyExoBoot_L1`, `Humotech_L1`, `OpenSourceLeg_A_L1`, `Tutorial_L1` |
| `terrain` | the ground (optional) | omit for flat; an inline config; or a terrains JSON path |

Run **`python -m assist_sim list`** to see every MSK and device that is installed,
and which pairs are compatible.

### MSK models

| Key | Muscles | Description |
|-----|---------|-------------|
| `myolegs22` | 22 | **2D** (sagittal-plane) |
| `myolegs26` | 26 | **3D** |
| `myolegs` | 80 | **3D**. The key of the 80-muscle model is `myolegs`, not `myolegs80`. |
| `myofullbody` | 416 | **3D**, full body with arms and torso muscles |

The muscle count and the 2D or 3D control mode come from `msk`. You do not set them
separately.

For a 3D key, the CO pipeline replaces the free root with named sagittal DOFs, because the
reflex controller reads the pelvis state from them. This is automatic. See
[Root frame](https://myoassist.neumove.org/modeling/msk-models/#root-frame).

### Discovering and validating

`python -m assist_sim list` prints the
authoritative installed set. `python -m assist_sim validate <msk> <device>` checks
one pair.

```bash
python -m assist_sim list                       # every installed MSK / device + compatibility
python -m assist_sim validate myolegs22 Humotech_L1
```

`EnvSpec.validate()` does the same check in code. It raises a `ValueError` that lists
the valid options when a key is unknown or the MSK and device pair is incompatible.

> **Note:**  
>  
> Environment validation **must** use the correct raw registry keys or the returned result will be inaccurate.  
>  
> **Example:**  
> ```bash
> python -m assist_sim validate myolegs22 Dephy_L1
> INVALID: myolegs22 x Dephy_L1
>
> python -m assist_sim validate myolegs22 DephyExoboot_L1
> INVALID: myolegs22 x DephyExoboot_L1
>
> python -m assist_sim validate myolegs22 DephyExoBoot_L1
> OK: myolegs22 x DephyExoBoot_L1
>    human:  myolegs22 (composed MjSpec, 38 bodies)
>    config: ...\Lib\site-packages\assist_sim\models\DephyExoBoot\L1config.yaml
> ```

### Terrain

Leave `terrain` unset (or `null`) for a flat, effectively-infinite ground plane.
Otherwise give one of the following.

A **uniform surface** (one plane or one heightfield):

| `terrain` | Result |
|-----------|--------|
| `{ "terrain": "flat" }` | a flat plane |
| `{ "terrain": "slope", "deg": 8 }` | a constant 8° incline (a tilted plane) |
| `{ "terrain": "random", "amplitude": 0.06 }` | a rough heightfield, up to 6 cm relief |
| `{ "terrain": "sinusoidal", "amplitude": 0.05, "period": 1.0 }` | rolling waves |

Those are the common cases. The full field set, including `resolution`, `extent`,
`safe_zone_radius` and `seed`, is in
[Uniform Terrains](https://myoassist.neumove.org/modeling/terrains/uniform/). Read `resolution` and `extent`
together: they set the heightfield cell size, which is what the roughness actually
feels like underfoot.

A **tiled grid**: a [terrain config](https://myoassist.neumove.org/modeling/terrains/configuration/) with a `grid`
and per-cell `tiles`. Tile types are `flat`, `slope`, `stairs`,
`pyramid_stairs`, `rough`, `boulders`, `stepping_stones`, `discrete_obstacles`, and
`gap`. You can fill empty cells with `randomization`. Give the config inline or as a
path to a JSON file:

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

The terrain sets the course grade. A `slope` terrain is the incline, and the
evaluation camera, cost, and readouts derive the angle from it. There is no separate
slope flag.

> **Reflex CO is for steady-state locomotion.** A constant terrain is manageable. The CO
> pipeline and reflex controller do not optimize over highly variable terrain (rough, stairs,
> mixed tiles). Use variable terrain with RL.

## Using an environment spec

### Controller Optimization (reflex)

```bash
# raw flags
python -m ctrl_optim.optim.train --msk myolegs22 --device Humotech_L1 \
    --terrain '{"terrain":"slope","deg":8}' --sim_time 20 -eff --ExoOn 1 ...

# or a shared env-spec file
python -m ctrl_optim.optim.train --env-spec docs/examples/env_exo_slope.json --sim_time 20 -eff --ExoOn 1 ...
```

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
xml = spec.compose()     # returns a loadable MuJoCo MJCF string
spec.compose(export_path="my_env.xml")   # also writes a standalone, loadable file
```

## Ready-to-use examples

The `docs/examples/` directory in the
[myoassist repo](https://github.com/neumovelab/myoassist/tree/main/docs/examples) has
runnable env-specs:

| File | Environment |
|------|-------------|
| `env_exo_flat.json` | `myolegs22` + `Humotech_L1`, flat |
| `env_exo_slope.json` | `myolegs22` + `Humotech_L1`, 8° slope |
| `env_prosthesis_rough.json` | `myolegs22` + `OpenSourceLeg_A_L1`, rough heightfield |
| `env_tiled_course.json` | `myolegs22` + `Humotech_L1`, a flat, slope, and stairs tiled course |
| `env_tiled_random.json` | `myolegs22` + `OpenExo_L1`, a randomized 3×3 tiled grid |
