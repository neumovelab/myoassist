import os

# Cap the OpenMP-family thread pools before numpy or torch is imported -- they read
# these at import time, so this block has to stay above every other import.
#
# PPO's update, not the environment, dominates wall time here: at num_envs=64 a rollout
# spends ~5 s stepping MuJoCo and the rest in the update. Left at its default, torch
# fans every op out across all cores while the env workers compete for those same cores,
# and the networks ([64, 64]) are far too small for that to pay off. Measured on a
# 64-core / 128-thread box, imitation config, num_envs=64, steps/sec:
#
#     threads     1      4      8     16    default (64)
#     steps/s  2286   2731   2979   2809            819
#
# 8 is the peak and is used as the default; the win over the untuned default is ~3.6x.
# The right value is machine-dependent, so MYOASSIST_NUM_THREADS overrides it, and an
# OMP_NUM_THREADS you set yourself is left alone.
_MYOASSIST_THREADS = os.environ.get("MYOASSIST_NUM_THREADS", "8")
for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_var, _MYOASSIST_THREADS)

import numpy as np  # noqa: E402  -- must follow the thread-pool caps above
import rl_train.train.train_configs.config as myoassist_config  # noqa: E402
import rl_train.utils.train_log_handler as train_log_handler  # noqa: E402
from rl_train.utils.data_types import DictionableDataclass  # noqa: E402
import json  # noqa: E402
from datetime import datetime  # noqa: E402
from rl_train.envs.environment_handler import EnvironmentHandler  # noqa: E402
import subprocess  # noqa: E402
import torch  # noqa: E402

# The env vars above are enough on Linux and Windows, where torch's CPU backend reads
# OMP_NUM_THREADS at import. macOS torch does not, so set it explicitly too; both paths
# then agree on the resolved value, whether it came from the default or an override.
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))


def get_git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("ascii").strip()
        return {"commit": commit, "branch": branch}
    except Exception:
        # Provenance metadata only -- a run outside a git checkout still has to start.
        return {"commit": "unknown", "branch": "unknown"}


# Version information
VERSION = {
    "version": "0.3.0",  # MAJOR.MINOR.PATCH
    **get_git_info(),
}


def ppo_evaluate_with_rendering(config):
    seed = 1234
    np.random.seed(seed)

    env = EnvironmentHandler.create_environment(config, is_rendering_on=True, is_evaluate_mode=True)
    model = EnvironmentHandler.get_stable_baselines3_model(config, env)

    EnvironmentHandler.updateconfig_from_model_policy(config, model)

    obs, info = env.reset()
    for _ in range(config.evaluate_param_list[0]["num_timesteps"]):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, truncated, info = env.step(action)
        if truncated:
            obs, info = env.reset()

    env.close()


def ppo_train_with_parameters(config, train_time_step, is_rendering_on, train_log_handler):
    seed = 1234
    np.random.seed(seed)

    env = EnvironmentHandler.create_environment(config, is_rendering_on)
    model = EnvironmentHandler.get_stable_baselines3_model(config, env)

    EnvironmentHandler.updateconfig_from_model_policy(config, model)

    session_config_dict = DictionableDataclass.to_dict(config)
    session_config_dict["env_params"].pop("reference_data", None)

    session_config_dict["code_version"] = VERSION
    with open(os.path.join(log_dir, "session_config.json"), "w", encoding="utf-8") as file:
        json.dump(session_config_dict, file, ensure_ascii=False, indent=4)

    custom_callback = EnvironmentHandler.get_callback(config, train_log_handler)

    model.learn(
        reset_num_timesteps=False, total_timesteps=train_time_step, log_interval=1, callback=custom_callback, progress_bar=True
    )
    env.close()
    print("learning done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_file_path", type=str, default="", help="path to train config file")
    parser.add_argument(
        "--flag_rendering", type=bool, default=False, action=argparse.BooleanOptionalAction, help="rendering(True/False)"
    )
    parser.add_argument(
        "--flag_realtime_evaluate",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="realtime evaluate(True/False)",
    )

    args, unknown_args = parser.parse_known_args()
    if args.config_file_path is None:
        raise ValueError("config_file_path is required")

    default_config = EnvironmentHandler.get_session_config_from_path(
        args.config_file_path, myoassist_config.TrainSessionConfigBase
    )
    DictionableDataclass.add_arguments(default_config, parser, prefix="config.")
    args = parser.parse_args()

    config_type = EnvironmentHandler.get_config_type_from_session_id(default_config.env_params.env_id)
    config = EnvironmentHandler.get_session_config_from_path(args.config_file_path, config_type)

    DictionableDataclass.set_from_args(config, args, prefix="config.")

    log_dir = os.path.join("rl_train", "results", f"train_session_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    train_log_handler = train_log_handler.TrainLogHandler(log_dir)

    if args.flag_realtime_evaluate:
        ppo_evaluate_with_rendering(config)
    else:
        ppo_train_with_parameters(
            config,
            train_time_step=config.total_timesteps,
            is_rendering_on=args.flag_rendering,
            train_log_handler=train_log_handler,
        )
