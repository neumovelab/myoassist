"""Sweep torch/OMP thread count for the PPO update, and measure how many epochs
actually run before target_kl early-stopping. Env stepping is unaffected by these
threads (it is MuJoCo in subprocesses); the update is where they matter, and during
the update the env workers are idle, so >1 thread may pay off.
"""

import json
import os
import re
import subprocess
import sys
import time
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]  # tools/benchmarks/<this file>
# Prefer the project venv's interpreter, but fall back to whatever is running this so the
# script also works from an already-activated environment or a non-Windows checkout.
_venv = REPO / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
PY_EXE = str(_venv) if _venv.exists() else sys.executable
CFG = "rl_train/train/train_configs/imitation_tutorial_22_separated_net_partial_obs.json"
NUM_ENVS, N_STEPS, ROLLOUTS = 64, 512, 4
BATCH = 8192
MINIBATCHES = NUM_ENVS * N_STEPS // BATCH

results = []
for threads in [int(a) for a in sys.argv[1:]] or [1, 4, 8, 16]:
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[v] = str(threads)
    total = NUM_ENVS * N_STEPS * ROLLOUTS
    cmd = [
        PY_EXE,
        "rl_train/run_train.py",
        "--config_file_path",
        CFG,
        "--config.total_timesteps",
        str(total),
        "--config.env_params.num_envs",
        str(NUM_ENVS),
        "--config.ppo_params.n_steps",
        str(N_STEPS),
        "--config.logger_params.logging_frequency",
        "1000",
        "--config.logger_params.evaluate_frequency",
        "1000",
    ]
    print(f"\n=== OMP/MKL threads = {threads} ===", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    wall = time.time() - t0
    steps = [int(m) for m in re.findall(r"\|\s+total_timesteps\s+\|\s+(\d+)\s+\|", p.stdout)]
    elapsed = [int(m) for m in re.findall(r"\|\s+time_elapsed\s+\|\s+(\d+)\s+\|", p.stdout)]
    nupd = [int(m) for m in re.findall(r"\|\s+n_updates\s+\|\s+(\d+)\s+\|", p.stdout)]
    early = len(re.findall(r"Early stopping at step (\d+)", p.stdout))
    stops = [int(m) for m in re.findall(r"Early stopping at step (\d+)", p.stdout)]
    fps = None
    if len(steps) >= 2 and elapsed[-1] > elapsed[0]:
        fps = (steps[-1] - steps[0]) / (elapsed[-1] - elapsed[0])
    grad_per_rollout = (nupd[-1] - nupd[0]) / max(len(nupd) - 1, 1) if len(nupd) >= 2 else None
    r = {
        "threads": threads,
        "steady_fps": round(fps, 1) if fps else None,
        "wall_sec": round(wall, 1),
        "grad_steps_per_rollout": round(grad_per_rollout, 1) if grad_per_rollout else None,
        "epochs_actually_run": round(grad_per_rollout / MINIBATCHES, 1) if grad_per_rollout else None,
        "early_stop_epochs": stops[-4:],
        "rc": p.returncode,
    }
    if p.returncode != 0:
        r["stderr_tail"] = p.stderr[-400:]
    results.append(r)
    print(json.dumps(r, indent=2), flush=True)

(REPO / "bench_threads.json").write_text(json.dumps(results, indent=2))
print("\n=== SUMMARY (nominal 30 epochs x %d minibatches = %d grad steps) ===" % (MINIBATCHES, 30 * MINIBATCHES), flush=True)
for r in results:
    print(
        f"  threads={r['threads']:>3}  fps={r['steady_fps']}  grad_steps/rollout={r['grad_steps_per_rollout']}  "
        f"epochs_run={r['epochs_actually_run']}  rc={r['rc']}",
        flush=True,
    )
