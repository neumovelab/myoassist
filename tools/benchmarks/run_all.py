"""Run the whole model cost benchmark suite into one directory, then print the report.

Steps (each a separate script; see its docstring for the protocol):
  sim         1 pinned core, activation 0 / 0.05 / 0.5, 20 runs          -> sim_c1.json
  sim_bare    the same without the device, activation 0.05                -> sim_c1_bare.json
  validate    replay vs closed-loop policy (needs --checkpoint)            -> validate.json
  sim_conc    N concurrent unpinned processes, 3 runs per N                -> sim_c<N>.json
  train       PPO from scratch, 4 models x num_envs, 3 runs                -> train_scratch.json
  train_ckpt  the checkpoint's config, from scratch and from the policy    -> train_ckpt_{scratch,trained}.json
  build       build time with and without the device, cache off / hit      -> build.json
  co          CMA-ES rollouts, ExoOn 1 and 0, serial and parallel          -> co.json

Process and env counts follow the machine: N and num_envs are half the physical cores, all
physical cores and all logical CPUs; CO uses 8, half and all physical cores, capped at 61 on
Windows (see co.py). Use the High performance power plan on Windows (see README.md); this script
does not change it.

Usage (repo root, project venv):
    python tools/benchmarks/run_all.py                                      # everything, ~8 h on 64 cores
    python tools/benchmarks/run_all.py --checkpoint <session_dir> <session_dir>/trained_models/<model>.zip
    python tools/benchmarks/run_all.py --only sim build --quick             # a fast smoke run
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from datetime import datetime

import common
import psutil

HERE = pathlib.Path(__file__).resolve().parent
STEPS = ["sim", "sim_bare", "validate", "sim_conc", "train", "train_ckpt", "build", "co"]


def counts():
    phys, logical = psutil.cpu_count(logical=False), psutil.cpu_count(logical=True)
    parallel = sorted({phys // 2, phys, logical})
    co_cap = 61 if os.name == "nt" else logical
    co_procs = sorted({0, 8, min(phys // 2, co_cap), min(phys, co_cap)})
    return parallel, co_procs


def plan(args, out):
    pin = sys.platform != "darwin"  # macOS has no CPU affinity API
    nopin = [] if pin else ["--no-pin"]
    parallel, co_procs = counts()
    q = args.quick
    sim_reps = ["--reps", "2", "--duration", "2"] if q else []
    py = sys.executable
    steps = {
        "sim": [[py, HERE / "sim.py", "--activation", "0", "0.05", "0.5", *sim_reps, *nopin, "--out", out / "sim_c1.json"]],
        "sim_bare": [[py, HERE / "sim.py", "--device", "none", *sim_reps, *nopin, "--out", out / "sim_c1_bare.json"]],
        "sim_conc": [
            [
                py,
                HERE / "sim.py",
                "--concurrent",
                str(n),
                "--no-pin",
                "--reps",
                "1" if q else "3",
                *sim_reps[2:],
                "--out",
                out / f"sim_c{n}.json",
            ]
            for n in ([2] if q else parallel)
        ],
        "train": [
            [py, HERE / "train.py", "--num-envs", *map(str, [1, 4] if q else [1, *parallel]), "--reps", "1" if q else "3"]
            + (["--iterations", "2"] if q else [])
            + ["--out", out / "train_scratch.json"]
        ],
        "build": [
            [py, HERE / "build.py", "--device", common.DEVICE, "none", "--reps", "1" if q else "10"]
            + (["--warm", "1"] if q else [])
            + ([] if pin else ["--core", "-1"])
            + ["--out", out / "build.json"]
        ],
        "co": [
            [py, HERE / "co.py", "--exo", "on", "off", "--procs", *map(str, [0, 4] if q else co_procs)]
            + (["--generations", "2", "--reps", "1"] if q else [])
            + ["--out", out / "co.json"]
        ],
    }
    if args.checkpoint:
        session, model = map(pathlib.Path, args.checkpoint)
        steps["validate"] = [
            [py, HERE / "sim.py", "--validate", session, model, *sim_reps, *nopin, "--out", out / "validate.json"]
        ]
        ckpt = [py, HERE / "train.py", "--config", session / "session_config.json"]
        ckpt += ["--num-envs", *map(str, [1, 4] if q else [1, *parallel]), "--reps", "1" if q else "2"]
        ckpt += ["--iterations", "2"] if q else []
        steps["train_ckpt"] = [
            ckpt + ["--out", out / "train_ckpt_scratch.json"],
            ckpt + ["--init-policy", model, "--out", out / "train_ckpt_trained.json"],
        ]
    chosen = args.only or [s for s in STEPS if s not in (args.skip or [])]
    missing = [s for s in chosen if s not in steps]
    assert not missing or not args.only, f"{missing} need --checkpoint"
    return [(s, [str(c) for c in cmd]) for s in STEPS if s in chosen and s in steps for cmd in steps[s]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="results directory (default results/run_<time>)")
    ap.add_argument("--checkpoint", nargs=2, metavar=("SESSION_DIR", "MODEL_ZIP"), help="a trained 22-muscle policy")
    ap.add_argument("--only", nargs="+", choices=STEPS)
    ap.add_argument("--skip", nargs="+", choices=STEPS)
    ap.add_argument("--quick", action="store_true", help="few, short runs: checks that every step works")
    args = ap.parse_args()
    out = pathlib.Path(args.out) if args.out else common.RESULTS / f"run_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    steps = plan(args, out)
    log = []
    for name, cmd in steps:
        print(f"\n=== {name}: {' '.join(cmd[1:])}", flush=True)
        t0 = time.time()
        subprocess.run(cmd, cwd=common.REPO, check=True)
        log.append({"step": name, "cmd": cmd, "minutes": round((time.time() - t0) / 60, 1)})
        (out / "run_all.json").write_text(json.dumps({"quick": args.quick, "steps": log}, indent=2))
    print(f"\n=== report: {out}\n", flush=True)
    subprocess.run([sys.executable, str(HERE / "report.py"), str(out)], cwd=common.REPO, check=True)


if __name__ == "__main__":
    main()
