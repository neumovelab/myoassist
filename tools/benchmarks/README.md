# Benchmarks

Computational cost of the 22-, 26-, 80- and 416-muscle models: simulation speed, PPO training
throughput, model build time and controller-optimization (CO) throughput. All models carry the
Tutorial_L1 exoskeleton unless a script is run with `--device none`. The scripts are
diagnostics, not part of the training path. Each script's docstring gives its protocol.

| script | what it measures |
|---|---|
| `sim.py` | MuJoCo physics steps/s and real-time factor while every model replays one shared 3D walking reference; one pinned core or N concurrent processes; `--device none` for the bare MSK; `--xml` for reference models read from installed packages (MyoSuite's MyoLeg, Gymnasium's Humanoid-v5 model with its random-policy rollout); `--validate` checks the replay against closed-loop walking of a trained 22-muscle policy |
| `train.py` | PPO training env steps/s vs `num_envs`, split into rollout collection and update; generates one imitation config per model; `--threads` sweeps the thread count; `--env-only` steps the vectorized env without a policy |
| `build.py` | model build time (compose + MuJoCo compile), with and without `MYOASSIST_CACHE_DIR`, with and without the device |
| `co.py` | CO rollouts/s and physics steps/s in a real CMA-ES run, serial and in parallel, with the exo controlled or not |
| `report.py` | prints every result in a directory as Markdown tables |
| `run_all.py` | runs all of the above into one directory, then the report |
| `common.py` | shared helpers |

Run from the repo root with the project venv:

```
python tools/benchmarks/run_all.py                    # the full suite, ~8 h on 64 cores
python tools/benchmarks/run_all.py --quick            # a few minutes: checks that every step runs
python tools/benchmarks/run_all.py --checkpoint <session_dir> <session_dir>/trained_models/<model>.zip
python tools/benchmarks/report.py tools/benchmarks/results/<run_dir>
```

`--checkpoint` adds the replay validation and the training runs that start from a walking
policy. Results go to `tools/benchmarks/results/` (git-ignored). Each JSON records the
hardware, package versions, commit and any modified tracked files, so a result can be matched to
the code that produced it.

Machine notes:

- On Windows, use the High performance power plan. Under Balanced, single-core timings of the
  416-muscle model split into two clusters about 25% apart.
- `EvalParallel2`, the process pool CO uses, fails on Windows above 61 processes and the run
  hangs, so `co.py` caps the pool there.
- `SubprocVecEnv` spawn is slow on Windows (about 130 s for 64 envs). It dominates short
  benchmarks but amortises over a real run, and `train.py` excludes it.

## Thread count for PPO training

Measured on a Threadripper PRO 7985WX (64 cores, 128 threads) with
`imitation_tutorial_22_separated_net_partial_obs.json` and `num_envs=64`, before the thread cap
in `rl_train/run_train.py` existed. Reproduce with
`train.py --config <that config> --num-envs 64 --threads 1 4 8 16`.

| OMP/MKL threads | steps/s |
|---:|---:|
| default (64) | 819 |
| 1 | 2286 |
| 4 | 2731 |
| **8** | **2979** |
| 16 | 2809 |

Thread count dominates: the untuned default lets torch fan every op across all cores while the
env workers compete for the same cores. `run_train.py` now caps it at 8
(`MYOASSIST_NUM_THREADS`).

At `num_envs=64`, env stepping alone runs at about 6555 steps/s, so a 32768-step rollout spends
about 5 s in MuJoCo and the rest in the PPO update. The update, not the environment, is the
bottleneck. Two things bound it, and neither is worth trading learning behaviour for:

- `n_epochs` is 30 nominally, but `target_kl = 0.01` stops it at about 7.5 epochs (30 of a
  nominal 120 gradient steps per rollout). Lowering `n_epochs` to about 10 would be honest
  bookkeeping but would not speed anything up.
- `frame_skip = 40` (`physics_sim_framerate` 1200 / `control_framerate` 30) means 40 `mj_step`
  calls per RL step, at 36.2 us each. That sets the ~691 steps/s single-env ceiling. It is a
  physics parameter: changing it changes results and breaks comparability with existing
  checkpoints.
