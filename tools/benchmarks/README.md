# Training throughput benchmarks

Scripts used to size RL training runs and to find the thread setting now baked into
`rl_train/run_train.py`. They are diagnostics, not part of the training path.

| script | what it measures |
|---|---|
| `bench_throughput.py` | end-to-end PPO steps/sec at given `num_envs` (`BENCH_CFG` selects the config) |
| `bench_threads.py` | steps/sec vs OpenMP/MKL thread count, plus how many epochs actually run before `target_kl` stops them |
| `profile_vecenv.py` | `SubprocVecEnv` env-stepping throughput alone, no PPO update |

Run from the repo root with the project venv, e.g. `python tools/benchmarks/bench_threads.py 1 4 8 16`.

## Findings (neu-multicore3: Threadripper PRO 7985WX, 64c/128t; imitation config, `num_envs=64`)

Thread count dominates. The untuned default lets torch fan every op across all cores while
the env workers compete for the same cores:

| OMP/MKL threads | steps/sec |
|---:|---:|
| default (64) | 819 |
| 1 | 2286 |
| 4 | 2731 |
| **8** | **2979** |
| 16 | 2809 |

Where the time goes at `num_envs=64`: env stepping alone runs at ~6555 steps/sec, so a
32768-step rollout spends ~5 s in MuJoCo and the rest in the PPO update — the update, not
the environment, is the bottleneck.

Two things bound the update, and neither is worth trading learning behaviour for:

- `n_epochs` is 30 nominally, but `target_kl = 0.01` stops it at **~7.5 epochs** (30 of a
  nominal 120 gradient steps per rollout). Lowering `n_epochs` to ~10 would be honest
  bookkeeping but would not speed anything up.
- `frame_skip = 40` (`physics_sim_framerate` 1200 / `control_framerate` 30) means 40
  `mj_step` calls per RL step, at 36.2 us each. That sets the ~691 steps/sec single-env
  ceiling. It is a physics parameter: changing it changes results and breaks comparability
  with existing checkpoints.

`SubprocVecEnv` spawn is slow on Windows (~132 s for 64 envs, ~213 s for 96), which
dominates short benchmarks but amortises over a real run.
