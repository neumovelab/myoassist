"""SubprocVecEnv scaling for the imitation env, with and without thread limits."""

import os
import time
import json


def main():
    import numpy as np
    import torch
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from myosuite.utils import gym
    from rl_train.envs.environment_handler import EnvironmentHandler
    from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig
    from myoassist_utils.env_spec import EnvSpec

    lim = os.environ.get("LIMIT_THREADS") == "1"
    if lim:
        torch.set_num_threads(1)
    print(
        json.dumps({"limit_threads": lim, "torch_threads": torch.get_num_threads(), "OMP": os.environ.get("OMP_NUM_THREADS")}),
        flush=True,
    )

    CFG = "rl_train/train/train_configs/imitation_tutorial_22_separated_net_partial_obs.json"
    cfg = EnvironmentHandler.get_session_config_from_path(CFG, ExoImitationTrainSessionConfig)
    ref = EnvironmentHandler.load_reference_data(cfg)
    mp = (
        EnvSpec(msk=cfg.env_params.msk_key, device=cfg.env_params.device_key, terrain=cfg.env_params.terrain)
        .validate()
        .compose()
    )
    ga = {"seed": cfg.env_params.seed, "model_path": mp, "env_params": cfg.env_params, "is_evaluate_mode": False}
    if ref is not None:
        ga["reference_data"] = ref

    res = {"limit_threads": lim}
    for n in [8, 16, 32, 64, 96]:
        try:
            t_spawn = time.perf_counter()
            venv = SubprocVecEnv([lambda: gym.make(cfg.env_params.env_id, **ga).unwrapped for _ in range(n)])
            venv.reset()
            spawn = time.perf_counter() - t_spawn
            acts = np.stack([venv.action_space.sample() for _ in range(n)])
            for _ in range(3):
                venv.step(acts)
            STEPS = 40
            t0 = time.perf_counter()
            for _ in range(STEPS):
                venv.step(acts)
            dt = time.perf_counter() - t0
            tot = STEPS * n / dt
            print(f"  n={n:>3}  {round(tot):>7} steps/s total  {round(tot / n):>5}/env  spawn={spawn:.1f}s", flush=True)
            res[f"n{n}"] = {"total": round(tot), "per_env": round(tot / n), "spawn_s": round(spawn, 1)}
            venv.close()
            time.sleep(2)
        except Exception as e:
            print(f"  n={n} FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)

    with open(os.environ.get("PROFILE_OUT", "profile_vec.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
