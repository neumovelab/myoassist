# Environment spec examples

Each `.json` here is a complete **environment spec** — a human MSK model + an
assistive device + a terrain — that both the CO and RL pipelines can consume. See
[Defining an Environment](../getting-started/defining-an-environment.md) for the
full field reference.

| File | MSK | Device | Terrain |
|------|-----|--------|---------|
| `env_exo_flat.json` | `myolegs22` | `Humotech_L1` | flat |
| `env_exo_slope.json` | `myolegs22` | `Humotech_L1` | 8° slope |
| `env_prosthesis_rough.json` | `myolegs22` | `OpenSourceLeg_A_L1` | rough heightfield (6 cm) |
| `env_tiled_course.json` | `myolegs22` | `Humotech_L1` | tiled 1×3 course: flat → slope → stairs |
| `env_tiled_random.json` | `myolegs22` | `OpenExo_L1` | randomized 3×3 tiled grid |

## Use one

```bash
# Controller Optimization (reflex)
python -m ctrl_optim.optim.train --env-spec docs/examples/env_exo_slope.json --sim_time 20 -eff --ExoOn 1 ...
```

```python
# Programmatically
from myoassist_utils.env_spec import EnvSpec
spec = EnvSpec.load("docs/examples/env_exo_slope.json").validate()
xml = spec.compose()   # -> loadable MJCF string
```

To make your own, copy one and change the keys — `python -m assist_sim list`
shows every valid MSK / device, and the terrain field (uniform or tiled) is
documented in [Defining an Environment](../getting-started/defining-an-environment.md).
