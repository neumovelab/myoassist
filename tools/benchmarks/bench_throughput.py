"""Throughput benchmark: steps/sec for the speed-control config at several num_envs.

Runs a fixed number of PPO rollouts per setting and reports steady-state throughput
(measured after the first rollout, so subprocess spawn and warmup are excluded).
"""

import json
import subprocess
import sys
import time
import re
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]  # tools/benchmarks/<this file>
# Prefer the project venv's interpreter, but fall back to whatever is running this so the
# script also works from an already-activated environment or a non-Windows checkout.
_venv = REPO / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
PY_EXE = str(_venv) if _venv.exists() else sys.executable
CFG = os.environ.get("BENCH_CFG", "rl_train/train/train_configs/imitation_tutorial_22_separated_net_partial_obs.json")
N_STEPS = 512
ROLLOUTS = 5

results = []
for num_envs in [int(a) for a in sys.argv[1:]] or [64, 96]:
    total = num_envs * N_STEPS * ROLLOUTS
    cmd = [
        PY_EXE,
        "rl_train/run_train.py",
        "--config_file_path",
        CFG,
        "--config.total_timesteps",
        str(total),
        "--config.env_params.num_envs",
        str(num_envs),
        "--config.ppo_params.n_steps",
        str(N_STEPS),
        "--config.logger_params.logging_frequency",
        "1000",
        "--config.logger_params.evaluate_frequency",
        "1000",
    ]
    print(
        f"\n=== cfg={pathlib.Path(CFG).stem} num_envs={num_envs}  total_timesteps={total}  ({ROLLOUTS} rollouts) ===",
        flush=True,
    )
    t0 = time.time()
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    wall = time.time() - t0
    # SB3 logs cumulative fps and total_timesteps per iteration; take the last pair.
    fps = [int(m) for m in re.findall(r"\|\s+fps\s+\|\s+(\d+)\s+\|", p.stdout)]
    iters = [int(m) for m in re.findall(r"\|\s+total_timesteps\s+\|\s+(\d+)\s+\|", p.stdout)]
    elapsed = [int(m) for m in re.findall(r"\|\s+time_elapsed\s+\|\s+(\d+)\s+\|", p.stdout)]
    steady = None
    if len(iters) >= 2 and len(elapsed) >= 2 and elapsed[-1] > elapsed[0]:
        steady = (iters[-1] - iters[0]) / (elapsed[-1] - elapsed[0])
    r = {
        "num_envs": num_envs,
        "total_timesteps": total,
        "wall_sec": round(wall, 1),
        "sb3_fps_cumulative": fps[-1] if fps else None,
        "steady_fps_excl_first_rollout": round(steady, 1) if steady else None,
        "returncode": p.returncode,
    }
    if p.returncode != 0:
        r["stderr_tail"] = p.stderr[-600:]
    results.append(r)
    print(json.dumps(r, indent=2), flush=True)

(REPO / os.environ.get("BENCH_OUT", "bench_results.json")).write_text(json.dumps(results, indent=2))
print("\n=== SUMMARY ===", flush=True)
for r in results:
    print(
        f"  num_envs={r['num_envs']:>3}  steady={r['steady_fps_excl_first_rollout']}  "
        f"cumulative={r['sb3_fps_cumulative']}  wall={r['wall_sec']}s  rc={r['returncode']}",
        flush=True,
    )
