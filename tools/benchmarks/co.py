"""Controller-optimization (CO) throughput per MSK: rollouts/s and physics steps/s in a real CMA-ES run.

Runs the CMA-ES loop of `ctrl_optim/optim/train.py` (single mode) for a few generations: the
same parser, environment dict, bounds, initial parameters, CMA options and `EvalParallel2` pool,
evaluating every candidate with the unmodified `func_Walk_FitCost`. One rollout is one candidate:
build the reflex env (compose + MuJoCo compile + controller setup), reset, check the initial pose,
step until the walker falls or `--sim_time` runs out, compute the cost. Instrumentation only
wraps `myoLeg_reflex.__init__` (build time) and `run_reflex_step_Cost` (step count and time).
Thread settings are left at their defaults, as train.py leaves them.

The optimization starts from the default initial parameters, so the walker falls early (well
under a second of simulated time) and a rollout's wall time is mostly the env build. The runs
measure throughput, not whether CO learns.

Settings: the tutorial CO config (`training_configs/tutorial.sh`) without `--param_path`, with
only `--msk` changed, and `--exo off` for `--ExoOn 0` (the exo stays in the model, uncontrolled:
the CO code needs its actuators, so there is no device-free run). CO supports 22/26/80 muscles,
not the full-body model. One departure from train.py: its initial spline timing is applied to
the 80-muscle model too (see `setup`).

Per run, excluding the first generation (pool start-up and first imports):
  * rollouts/s: candidates evaluated per wall second of `eval_all`;
  * physics steps/s: mj_step calls per wall second, aggregated over all processes;
  * stepping rate: mj_step calls per second inside the step loop only, per process (no build);
  * build time per rollout, simulated time per rollout, share of candidates rejected at the
    initial-pose check (built but not simulated).
train.py also re-evaluates the best candidate once per generation in the main process; that
time is recorded separately (`best_check_s`) and is not in the rates above.

Process counts: 0 is the serial in-process loop (`EvalParallel2(..., 0)`, popsize 8); N > 0 is a
pool of N processes with popsize N, as the shipped configs pair them (8/8, 32/32).
`EvalParallel2` is a `multiprocessing.Pool`, and on Windows a pool of 62 or more processes fails
("need at most 63 handles") and the run hangs, so N is capped at 61 there.

Usage (repo root, project venv):
    python tools/benchmarks/co.py                                       # 22/26/80, 0/8/32/60 procs, cache off/hit
    python tools/benchmarks/co.py --exo on off --procs 0 32 --cache hit
    python tools/benchmarks/co.py --msk myolegs22 --procs 0 4 --generations 3 --reps 1
"""

import argparse
import os
import shutil
import sys
import time

import common
import numpy as np

OPTIM = common.REPO / "ctrl_optim" / "optim"
sys.path.insert(0, str(OPTIM))  # train.py imports optim_utils / config / cost_functions from here

# Module level, so that pool workers (which re-import this file) get the same instrumentation.
from cost_functions.walk_cost import func_Walk_FitCost  # noqa: E402
from ctrl_optim.ctrl.reflex.reflex_interface import myoLeg_reflex  # noqa: E402

CO_MSKS = ["myolegs22", "myolegs26", "myolegs"]  # ctrl_optim/optim/config/environment.py msk_to_musc
CACHE_ENV = "MYOASSIST_CACHE_DIR"
WINDOWS_POOL_MAX = 61
# training_configs/tutorial.sh, without --param_path / --save_path; --msk is set per run.
TUTORIAL_ARGS = [
    "--device", "Tutorial_L1", "--sim_time", "20", "--pose_key", "walk_left", "--num_strides", "5",
    "--delayed", "0", "--optim_mode", "single", "--reflex_mode", "uni", "--tgt_vel", "1.25",
    "--trunk_err_type", "ref_diff", "--tgt_sym_th", "0.1", "--tgt_grf_th", "1.5", "-eff", "--ExoOn", "{exo}",
    "--use_4param_spline", "--max_torque", "100.0", "--sigma_gain", "10",
]  # fmt: skip

_STATS = {}
_orig_init = myoLeg_reflex.__init__
_orig_step = myoLeg_reflex.run_reflex_step_Cost


def _timed_init(self, *args, **kwargs):
    t0 = time.perf_counter()
    _orig_init(self, *args, **kwargs)
    _STATS["build_s"] = time.perf_counter() - t0
    _STATS["frame_skip"] = self.frame_skip
    _STATS["dt"] = self.dt


def _counted_step(self):
    t0 = time.perf_counter()
    out = _orig_step(self)
    _STATS["step_s"] += time.perf_counter() - t0
    _STATS["control_steps"] += 1
    return out


myoLeg_reflex.__init__ = _timed_init
myoLeg_reflex.run_reflex_step_Cost = _counted_step


def _reset_stats():
    _STATS.clear()
    _STATS.update(control_steps=0, step_s=0.0)


def timed_rollout(params, *args):
    """`func_Walk_FitCost` plus the build/step counters of this rollout."""
    _reset_stats()
    t0 = time.perf_counter()
    cost = func_Walk_FitCost(params, *args)
    _STATS["rollout_s"] = time.perf_counter() - t0
    return float(cost), dict(_STATS)


def co_args(exo):
    return [a.format(exo=int(exo == "on")) for a in TUTORIAL_ARGS]


def setup(msk, exo):
    """What train.py main() builds before its CMA loop, for the tutorial config."""
    import optim_utils.bounds as bounds_mod
    from config import create_environment_dict, get_optimization_type, initParser
    from optim_utils.bounds import get_bounds

    argv, sys.argv = sys.argv, ["train.py", "--msk", msk, *co_args(exo)]
    input_args = initParser()
    sys.argv = argv
    env_dict = create_environment_dict(input_args)
    bounds_mod.input_args = input_args
    control_mode = "2D" if input_args.musc_model == "22" else "3D"
    bound_start, bound_end = get_bounds(input_args.musc_model, control_mode)
    params_0 = np.ones(len(bound_start))
    if input_args.ExoOn and input_args.use_4param_spline:
        # train.py's initial spline timing. train.py applies it to 22/26 only; with its all-ones
        # spline every 80-muscle candidate fails the spline check in check_pose_validity and is
        # never simulated, so the same default is applied to 80 here.
        params_0[-4:] = [0.5, 0.467, 0.90, 0.075]
    params_0 = np.clip(params_0, bound_start, bound_end)
    one_step = np.genfromtxt(OPTIM / "ref_data" / "ref_kinematics_radians.csv", delimiter=",")
    one_EMG = np.genfromtxt(OPTIM / "ref_data" / "ref_EMG.csv", delimiter=",")
    fargs = (
        get_optimization_type(input_args),
        one_step,
        one_EMG,
        input_args.trunk_err_type,
        input_args.tgt_vel,
        input_args.num_strides,
        input_args.tgt_sym_th,
        input_args.tgt_grf_th,
        env_dict,
    )
    return input_args, params_0, (bound_start, bound_end), fargs


def worker(args):
    import cma
    from cma.optimization_tools import EvalParallel2

    assert (os.environ.get(CACHE_ENV) is not None) == (args.cache == "hit"), "cache env does not match the case"
    input_args, params_0, bounds, fargs = setup(args.msk, args.exo)
    popsize = 8 if args.procs == 0 else args.procs
    opts = cma.CMAOptions()
    opts.set("popsize", popsize)
    opts.set("CMA_stds", np.ones(len(params_0)) * input_args.sigma_gain)
    opts.set("bounds", list(bounds))
    opts["CMA_active"] = True
    opts["tolfun"] = 1e-11
    opts["tolx"] = 1e-11
    opts["ftarget"] = None
    opts["seed"] = args.rep + 1
    opts["verbose"] = -9
    opts["verb_log"] = 0  # train.py logs to files; not part of the evaluation cost
    es = cma.CMAEvolutionStrategy(params_0, 0.01, inopts=opts)

    gens = []
    t_pool = time.perf_counter()
    with EvalParallel2(timed_rollout, number_of_processes=args.procs) as eval_all:
        pool_s = time.perf_counter() - t_pool
        for _ in range(args.generations):
            X = es.ask()
            t0 = time.perf_counter()
            res = eval_all(X, args=fargs)
            wall = time.perf_counter() - t0
            es.tell(X, [c for c, _ in res])
            _reset_stats()
            t1 = time.perf_counter()
            func_Walk_FitCost(es.result.xbest, *fargs, cost_print=True)  # train.py's per-generation check
            gens.append({"wall_s": wall, "best_check_s": time.perf_counter() - t1, "rollouts": [s for _, s in res]})

    timed = gens[1:]  # the first generation pays for pool start-up and first imports
    rollouts = [r for g in timed for r in g["rollouts"]]
    wall = sum(g["wall_s"] for g in timed)
    simulated = [r["control_steps"] * r["dt"] for r in rollouts]
    stepped = [r for r in rollouts if r["control_steps"] > 0]
    common.emit(
        {
            "msk": args.msk,
            "exo": args.exo,
            "procs": args.procs,
            "popsize": popsize,
            "cache": args.cache,
            "rep": args.rep,
            "n_params": len(params_0),
            "frame_skip": rollouts[0]["frame_skip"],
            "dt": rollouts[0]["dt"],
            "pool_start_s": round(pool_s, 3),
            "first_gen_s": round(gens[0]["wall_s"], 3),
            "generations_timed": len(timed),
            "rollouts_timed": len(rollouts),
            "rollouts_per_s": round(len(rollouts) / wall, 3),
            "physics_steps_per_s": round(sum(r["control_steps"] * r["frame_skip"] for r in rollouts) / wall, 1),
            "sim_s_per_wall_s": round(sum(simulated) / wall, 3),
            "stepping_physics_steps_per_s": round(
                sum(r["control_steps"] * r["frame_skip"] for r in stepped) / sum(r["step_s"] for r in stepped), 1
            )
            if stepped
            else None,
            "build_s_mean": round(float(np.mean([r["build_s"] for r in rollouts])), 4),
            "rollout_s_mean": round(float(np.mean([r["rollout_s"] for r in rollouts])), 4),
            "sim_s_per_rollout_mean": round(float(np.mean(simulated)), 3),
            "frac_rejected_at_pose_check": round(1 - len(stepped) / len(rollouts), 3),
            "best_check_s_mean": round(float(np.mean([g["best_check_s"] for g in timed])), 3),
            "best_cost": float(es.result.fbest),
        }
    )


def run_case(args, msk, exo, procs, cache, rep, cache_dir):
    env = {k: v for k, v in os.environ.items() if k != CACHE_ENV}
    if cache == "hit":
        env[CACHE_ENV] = str(cache_dir)
    cmd = [sys.executable, __file__, "--worker", "--msk", msk, "--exo", exo, "--procs", str(procs), "--cache", cache]
    cmd += ["--rep", str(rep), "--generations", str(args.generations)]
    # A hung pool (see the module docstring) has to fail the benchmark, not idle it.
    return common.run_worker(cmd, env=env, timeout=args.timeout)


def driver(args):
    from myoassist_utils.compose import compose_env_model

    if os.name == "nt":
        assert max(args.procs) <= WINDOWS_POOL_MAX, f"EvalParallel2 hangs above {WINDOWS_POOL_MAX} processes on Windows"
    cache_dir = common.RESULTS / "co_cache"
    shutil.rmtree(cache_dir, ignore_errors=True)
    for msk in args.msk:  # fill the cache with the CO build (planar root) of every model
        compose_env_model(msk, common.DEVICE, planar_root=True, cache_dir=cache_dir)

    cases = [(msk, e, n, c) for msk in args.msk for e in args.exo for n in args.procs for c in args.cache]
    runs = []
    for rep in range(args.reps):
        order = cases[rep % len(cases) :] + cases[: rep % len(cases)]  # rotate to spread drift
        for msk, exo, n, cache in order:
            r = run_case(args, msk, exo, n, cache, rep, cache_dir)
            runs.append(r)
            print(
                f"rep {rep} {msk:<10} exo={exo:<3} procs={n:>3} cache={cache:<3} rollouts/s={r['rollouts_per_s']:7.2f} "
                f"physics steps/s={r['physics_steps_per_s']:9.0f} stepping={r['stepping_physics_steps_per_s']} "
                f"build={r['build_s_mean']:.3f}s sim/rollout={r['sim_s_per_rollout_mean']:.2f}s "
                f"rejected={r['frac_rejected_at_pose_check']:.2f}",
                flush=True,
            )
    shutil.rmtree(cache_dir, ignore_errors=True)

    keys = (
        "rollouts_per_s",
        "physics_steps_per_s",
        "sim_s_per_wall_s",
        "stepping_physics_steps_per_s",
        "build_s_mean",
        "rollout_s_mean",
        "sim_s_per_rollout_mean",
        "frac_rejected_at_pose_check",
        "best_check_s_mean",
        "first_gen_s",
    )
    summary = {}
    for msk, exo, n, cache in cases:
        rs = [r for r in runs if (r["msk"], r["exo"], r["procs"], r["cache"]) == (msk, exo, n, cache)]
        summary[f"{msk}@exo_{exo}@procs{n}@{cache}"] = {
            "msk": msk,
            "exo": exo,
            "procs": n,
            "cache": cache,
            **{k: common.stats([r[k] for r in rs]) for k in keys},
        }
    settings = {
        "msk": args.msk,
        "exo": args.exo,
        "procs": args.procs,
        "cache": args.cache,
        "generations": args.generations,
        "reps": args.reps,
        "co_args": {e: co_args(e) for e in args.exo},
        "thread_env": {v: os.environ.get(v) for v in common.THREAD_VARS},
    }
    common.write_results("co", settings, summary, runs, args.out)
    for key, s in summary.items():
        print(
            f"  {key:<34} rollouts/s {s['rollouts_per_s']['mean']:8.2f} ± {s['rollouts_per_s']['sd']}  "
            f"stepping {s['stepping_physics_steps_per_s']['mean']}  build {s['build_s_mean']['mean']:.3f}s"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--msk", nargs="+", default=CO_MSKS)
    ap.add_argument("--exo", nargs="+", choices=["on", "off"], default=["on"], help="--ExoOn 1 / 0; the exo stays in the model")
    ap.add_argument("--procs", type=int, nargs="+", default=[0, 8, 32, 60], help="0 = serial, in-process")
    ap.add_argument("--cache", nargs="+", choices=["off", "hit"], default=["off", "hit"])
    ap.add_argument("--generations", type=int, default=8, help="CMA-ES generations per run; the first is not timed")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=1800, help="seconds before a run counts as hung")
    ap.add_argument("--out", default=None)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--rep", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        assert len(args.msk) == 1 and len(args.exo) == 1 and len(args.procs) == 1 and len(args.cache) == 1
        args.msk, args.exo, args.procs, args.cache = args.msk[0], args.exo[0], args.procs[0], args.cache[0]
        worker(args)
    else:
        driver(args)


if __name__ == "__main__":
    main()
