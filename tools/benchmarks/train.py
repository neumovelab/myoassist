"""PPO training throughput per MSK, split into rollout collection and update.

Builds the env, model and logging callback exactly as `rl_train/run_train.py` does (same
`EnvironmentHandler` calls, same OpenMP/BLAS/torch thread cap), adds a timing callback, and runs
a few PPO iterations. Reported per run, excluding the first iteration (subprocess spawn, warmup):

  * env steps/s over whole iterations: what a training run achieves;
  * collection steps/s: rollout only (parallel env stepping + batched policy inference);
  * update seconds per iteration and the gradient epochs it actually ran (`target_kl` ends the
    epoch loop early, by an amount that depends on the learning, so it differs across models);
  * mean episode length: from scratch the imitation env ends most episodes within a step or
    two, so collection pays for a reset almost every step.

Configs: by default one imitation config per `--msk`, generated into results/train_configs/ from
`imitation_tutorial_22_separated_net_full_obs.json`, changing only what the model forces:
  * `msk_key`, and `planar_root` for the 3D models (the env reads the root through the named
    pelvis DOFs, see `EnvParams.planar_root`);
  * the observed joints: the 3D models add the frontal and transverse DOFs a 3D walker needs
    (hip adduction and rotation, pelvis list and rotation, lateral pelvis velocity);
  * `joint_limit_sensor_keys`: the Rajagopal-lineage models have no mtp limit sensors;
  * `net_indexing_info`, rebuilt from those counts. Every sub-network keeps the template's full
    observation; the human actor drives all muscles, the exo actor the two device actuators.
PPO hyperparameters, network widths, rewards and the seed stay identical. These configs exist to
measure steps per second, not to train a good 3D walker. `--config` runs an existing config
instead (for example a session's `session_config.json`, with `--init-policy` to continue from
its checkpoint: the long-episode regime of a policy that already walks).

Other modes:
  * `--threads 1 4 8 16`: sweep the OpenMP/BLAS/torch thread count (default: run_train.py's,
    `MYOASSIST_NUM_THREADS` or 8);
  * `--env-only`: step the vectorized env with random actions, no policy and no PPO update.

Usage (repo root, project venv):
    python tools/benchmarks/train.py                                   # 4 models, 32 envs, 3 reps
    python tools/benchmarks/train.py --num-envs 1 32 64 128
    python tools/benchmarks/train.py --msk myolegs22 --num-envs 64 --threads 1 4 8 16
    python tools/benchmarks/train.py --config <session>/session_config.json --init-policy <zip>
"""

import argparse
import copy
import json
import os
import pathlib
import sys
import time

import common

TEMPLATE = common.REPO / "rl_train/train/train_configs/imitation_tutorial_22_separated_net_full_obs.json"
CONFIG_DIR = common.RESULTS / "train_configs"
EXTRA_POS_3D = ["hip_adduction_l", "hip_adduction_r", "hip_rotation_l", "hip_rotation_r", "pelvis_list", "pelvis_rotation"]
EXTRA_VEL_3D = EXTRA_POS_3D + ["pelvis_tz"]


# --- configs ----------------------------------------------------------------------------------


def obs_ranges(n_pos, n_vel, n_mus, n_sens):
    import numpy as np

    edges = np.cumsum([0, n_pos, n_vel, n_mus, n_sens, 1])
    comments = [f"{n_pos} qpos", f"{n_vel} qvel", f"{n_mus} muscle activation", f"{n_sens} foot force", "target velocity"]
    return [{"type": "range", "range": [int(edges[i]), int(edges[i + 1])], "comment": c} for i, c in enumerate(comments)]


def make_config(msk):
    """The imitation config for `msk` (see the module docstring); returns its path."""
    import mujoco
    import numpy as np
    from myoassist_utils.compose import compose_env_model
    from rl_train.envs.myoassist_leg_base import MyoAssistLegBase

    cfg = json.loads(TEMPLATE.read_text())
    env = cfg["env_params"]
    is_3d = msk != "myolegs22"
    m = mujoco.MjModel.from_xml_string(compose_env_model(msk, env["device_key"], planar_root=is_3d))
    n_muscles = common.MUSCLES[msk]
    muscles = np.where(m.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)[0]
    assert len(muscles) == n_muscles and list(muscles) == list(range(n_muscles)), f"{msk}: muscles not first"
    assert m.nu == n_muscles + 2, f"{msk}: expected exactly two device actuators after the muscles"

    env["msk_key"] = msk
    env["planar_root"] = is_3d
    if is_3d:
        env["observation_joint_pos_keys"] = env["observation_joint_pos_keys"] + EXTRA_POS_3D
        env["observation_joint_vel_keys"] = env["observation_joint_vel_keys"] + EXTRA_VEL_3D
    joints = {m.joint(j).name for j in range(m.njnt)}
    missing = [k for k in env["observation_joint_pos_keys"] + env["observation_joint_vel_keys"] if k not in joints]
    assert not missing, f"{msk}: observed joints not in the model: {missing}"
    sensors = {m.sensor(i).name for i in range(m.nsensor)}
    env["joint_limit_sensor_keys"] = [s for s in MyoAssistLegBase.JOINT_LIMIT_SENSOR_NAMES if s in sensors]
    assert all(s in sensors for s in env["observation_joint_sensor_keys"]), f"{msk}: foot sensors missing"

    obs = obs_ranges(
        len(env["observation_joint_pos_keys"]),
        len(env["observation_joint_vel_keys"]),
        n_muscles,
        len(env["observation_joint_sensor_keys"]),
    )
    nets = cfg["policy_params"]["custom_policy_params"]["net_indexing_info"]
    nets["human_actor"]["observation"] = copy.deepcopy(obs)
    nets["human_actor"]["action"] = [
        {
            "type": "range_mapping",
            "range_net": [0, n_muscles],
            "range_action": [0, n_muscles],
            "comment": f"{n_muscles} muscle activations",
        }
    ]
    nets["exo_actor"]["observation"] = copy.deepcopy(obs)
    nets["exo_actor"]["action"] = [
        {"type": "range_mapping", "range_net": [0, 2], "range_action": [n_muscles, n_muscles + 2], "comment": "2 for exo R, L"}
    ]
    nets["common_critic"]["observation"] = copy.deepcopy(obs)

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    out = CONFIG_DIR / f"imitation_bench_{n_muscles}.json"
    out.write_text(json.dumps(cfg, indent=4) + "\n")
    print(
        f"  {out.relative_to(common.REPO)}  obs={obs[-1]['range'][1]}  act={m.nu}  limit_sensors={len(env['joint_limit_sensor_keys'])}"
    )
    return out


# --- worker -----------------------------------------------------------------------------------


def load_config(path, num_envs):
    from rl_train.envs.environment_handler import EnvironmentHandler

    env_id = json.loads(pathlib.Path(path).read_text())["env_params"]["env_id"]
    config = EnvironmentHandler.get_session_config_from_path(
        str(path), EnvironmentHandler.get_config_type_from_session_id(env_id)
    )
    config.env_params.num_envs = num_envs
    config.logger_params.evaluate_frequency = 10**9  # evaluation renders video; not part of training throughput
    return config


def worker_ppo(args, config):
    import numpy as np
    import torch
    from rl_train.envs.environment_handler import EnvironmentHandler
    from rl_train.utils.train_log_handler import TrainLogHandler
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList

    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

    class TimingCallback(BaseCallback):
        """Wall time of each rollout and of the update that follows it."""

        def __init__(self):
            super().__init__()
            self.rollout_start, self.rollout_end, self.n_updates_at_start = [], [], []
            self.episode_lengths, self._running = [], None

        def _on_rollout_start(self):
            self.rollout_start.append(time.perf_counter())
            self.n_updates_at_start.append(self.model._n_updates)

        def _on_step(self):
            dones = self.locals["dones"]
            if self._running is None:
                self._running = np.zeros(len(dones), dtype=int)
            self._running += 1
            self.episode_lengths.extend(self._running[dones].tolist())
            self._running[dones] = 0
            return True

        def _on_rollout_end(self):
            self.rollout_end.append(time.perf_counter())

    t0 = time.perf_counter()
    env = EnvironmentHandler.create_environment(config, is_rendering_on=False)
    spawn_s = time.perf_counter() - t0
    model = EnvironmentHandler.get_stable_baselines3_model(config, env, trained_model_path=args.init_policy)
    n_steps = model.n_steps  # a loaded checkpoint keeps its saved PPO hyperparameters, not the config's
    log_dir = common.RESULTS / "train_sessions" / f"{pathlib.Path(args.config).stem}_n{args.num_envs}_r{args.rep}_{os.getpid()}"
    timing = TimingCallback()
    callbacks = CallbackList([EnvironmentHandler.get_callback(config, TrainLogHandler(str(log_dir))), timing])
    model.learn(total_timesteps=args.num_envs * n_steps * args.iterations, callback=callbacks, log_interval=10**9)
    t_end = time.perf_counter()
    n_updates_end = model._n_updates
    env.close()

    starts, ends = np.array(timing.rollout_start), np.array(timing.rollout_end)
    collect_s = ends - starts
    update_s = np.append(starts[1:], t_end) - ends
    epochs = np.diff(np.append(timing.n_updates_at_start, n_updates_end))
    s = slice(1, None)  # drop the first iteration
    steps_per_iter = args.num_envs * n_steps
    n_iter = len(starts) - 1
    return {
        "n_steps": n_steps,
        "batch_size": model.batch_size,
        "n_epochs": model.n_epochs,
        "target_kl": model.target_kl,
        "spawn_s": round(spawn_s, 2),
        "iterations_timed": n_iter,
        "env_steps_per_s": round(steps_per_iter * n_iter / float((collect_s[s] + update_s[s]).sum()), 1),
        "collect_steps_per_s": round(steps_per_iter * n_iter / float(collect_s[s].sum()), 1),
        "collect_s_per_iter": round(float(collect_s[s].mean()), 3),
        "update_s_per_iter": round(float(update_s[s].mean()), 3),
        "epochs_per_iter": round(float(epochs[s].mean()), 2),
        "update_s_per_epoch": round(float(update_s[s].sum() / max(epochs[s].sum(), 1)), 4),
        "mean_episode_len": round(float(np.mean(timing.episode_lengths)), 2) if timing.episode_lengths else None,
        "episodes": len(timing.episode_lengths),
    }


def worker_env_only(args, config):
    """Vectorized env stepping with random actions: no policy, no update."""
    import numpy as np
    from rl_train.envs.environment_handler import EnvironmentHandler

    t0 = time.perf_counter()
    env = EnvironmentHandler.create_environment(config, is_rendering_on=False)
    spawn_s = time.perf_counter() - t0
    vec = hasattr(env, "num_envs")  # a single env comes back as a plain gym env
    rng = np.random.default_rng(args.rep)
    low, high = env.action_space.low, env.action_space.high
    n = args.num_envs

    def act():
        return rng.uniform(low, high, size=(n, *low.shape)) if vec else rng.uniform(low, high)

    def step(k):
        for _ in range(k):
            if vec:
                env.step(act())
            else:
                _, _, terminated, truncated, _ = env.step(act())
                if terminated or truncated:
                    env.reset()

    env.reset()
    step(args.env_steps // 10)  # warmup
    t0 = time.perf_counter()
    step(args.env_steps)
    wall = time.perf_counter() - t0
    env.close()
    return {
        "spawn_s": round(spawn_s, 2),
        "env_steps_timed": args.env_steps * n,
        "env_steps_per_s": round(args.env_steps * n / wall, 1),
    }


def worker(args):
    config = load_config(args.config, args.num_envs)
    res = worker_env_only(args, config) if args.env_only else worker_ppo(args, config)
    common.emit(
        {
            "config": str(args.config),
            "msk": config.env_params.msk_key,
            "device": config.env_params.device_key,
            "init_policy": args.init_policy,
            "env_only": args.env_only,
            "num_envs": args.num_envs,
            "threads": int(os.environ["OMP_NUM_THREADS"]),
            "rep": args.rep,
            **res,
        }
    )


# --- driver -----------------------------------------------------------------------------------


def driver(args):
    configs = args.config or [str(make_config(msk)) for msk in args.msk]
    cases = [(c, n, t) for t in args.threads for c in configs for n in args.num_envs]
    runs = []
    for rep in range(args.reps):
        for cfg, n, threads in cases:
            env = dict(os.environ)
            for var in common.THREAD_VARS:
                env[var] = str(threads)
            cmd = [sys.executable, __file__, "--worker", "--config", cfg, "--num-envs", str(n), "--rep", str(rep)]
            cmd += ["--iterations", str(args.iterations), "--env-steps", str(args.env_steps)]
            cmd += ["--init-policy", args.init_policy] if args.init_policy else []
            cmd += ["--env-only"] if args.env_only else []
            r = common.run_worker(cmd, env=env)
            runs.append(r)
            line = f"rep {rep} {pathlib.Path(cfg).stem:<28} n={n:>3} threads={threads:>2} steps/s={r['env_steps_per_s']:8.1f}"
            if not args.env_only:
                line += (
                    f" collect={r['collect_steps_per_s']:8.1f}/s update={r['update_s_per_iter']:6.2f}s "
                    f"epochs={r['epochs_per_iter']:5.2f} ep_len={r['mean_episode_len']}"
                )
            print(line + f" spawn={r['spawn_s']}s", flush=True)

    keys = ["env_steps_per_s"]
    if not args.env_only:
        keys += ["collect_steps_per_s", "update_s_per_iter", "epochs_per_iter", "update_s_per_epoch", "mean_episode_len"]
    summary = {}
    for cfg, n, threads in cases:
        rs = [r for r in runs if r["config"] == cfg and r["num_envs"] == n and r["threads"] == threads]
        summary[f"{pathlib.Path(cfg).stem}@n{n}@t{threads}"] = {
            "msk": rs[0]["msk"],
            "num_envs": n,
            "threads": threads,
            **{k: common.stats([r[k] for r in rs]) for k in keys},
        }
    settings = {
        "configs": configs,
        "num_envs": args.num_envs,
        "threads": args.threads,
        "iterations": args.iterations,
        "reps": args.reps,
        "init_policy": args.init_policy,
        "env_only": args.env_only,
    }
    used = {c: json.loads(pathlib.Path(c).read_text()) for c in configs}
    common.write_results("train", settings, summary, runs, args.out, config_contents=used)
    for key, s in summary.items():
        print(f"  {key:<40} steps/s {s['env_steps_per_s']['mean']:8.1f} ± {s['env_steps_per_s']['sd']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--msk", nargs="+", default=common.MSKS, help="models to generate configs for")
    ap.add_argument("--config", nargs="+", default=None, help="existing config paths instead of generated ones")
    ap.add_argument("--num-envs", type=int, nargs="+", default=[32])
    ap.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[int(os.environ.get("MYOASSIST_NUM_THREADS", "8"))],
        help="OpenMP/BLAS/torch threads",
    )
    ap.add_argument("--iterations", type=int, default=6, help="PPO iterations per run; the first is not timed")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--init-policy", default=None, help="checkpoint .zip to continue from")
    ap.add_argument("--env-only", action="store_true", help="random actions, no policy, no update")
    ap.add_argument("--env-steps", type=int, default=200, help="--env-only: timed steps per env")
    ap.add_argument("--out", default=None)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--rep", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        assert len(args.config) == 1 and len(args.num_envs) == 1
        args.config, args.num_envs = args.config[0], args.num_envs[0]
        worker(args)
    else:
        driver(args)


if __name__ == "__main__":
    main()
