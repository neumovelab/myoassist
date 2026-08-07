# Terrain Types

In RL, terrain is part of the [environment spec](../getting-started/defining-an-environment.md):
set `env_params.terrain` in your training config. It is composed into the model
when the environment is built — there is no separate runtime terrain manager.

```json
"env_params": {
  "msk_key": "myolegs22",
  "device_key": "Humotech_L1",
  "terrain": { "terrain": "random", "amplitude": 0.06 }
}
```

`terrain` accepts one of:

- `null` → a flat ground plane (the default).
- an **inline uniform config** (one plane or one heightfield).
- a **path to a [`myoassist_terrains`](https://github.com/neumovelab/myoassist.terrains) JSON config** for tiled / complex courses.

## Uniform terrains

| `terrain` | Result |
|-----------|--------|
| `{ "terrain": "flat" }` | flat plane |
| `{ "terrain": "slope", "deg": 8 }` | constant 8° incline |
| `{ "terrain": "random", "amplitude": 0.06 }` | rough heightfield, ≤ 6 cm relief |
| `{ "terrain": "sinusoidal", "amplitude": 0.05, "period": 1.0 }` | rolling waves |

The `random` and `sinusoidal` heightfields keep a smooth **safe zone** near the
origin (flattened around the reset point) so the agent doesn't spawn mid-obstacle.

## Tiled terrains

For multi-terrain courses, give `terrain` a `myoassist_terrains` config with a
`grid` and per-cell `tiles` (types: `flat`, `slope`, `stairs`, `pyramid_stairs`,
`rough`, `boulders`, `stepping_stones`, `discrete_obstacles`, `gap`), optionally
filled by `randomization`:

```json
"terrain": {
  "grid": { "rows": 3, "cols": 3, "tile_size": [8.0, 8.0] },
  "border": { "width": 0.5 },
  "tiles": [ { "row": 1, "col": 1, "type": "flat", "params": { "height": 0.0 } } ],
  "randomization": { "seed": 17, "weights": { "rough": 0.4, "stairs": 0.3, "slope": 0.3 } }
}
```

See the [`myoassist_terrains`](https://github.com/neumovelab/myoassist.terrains)
repo for the full tile schema and parameters, and
[Defining an Environment](../getting-started/defining-an-environment.md) for how
the same `terrain` field is used by both the RL and CO pipelines.
