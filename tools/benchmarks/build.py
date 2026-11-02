"""Model build time per MSK: compose (MSK + device -> one MJCF) and MuJoCo compile, with and without the cache.

Every env build runs two stages:

  * compose: `compose_env_model` merges the MSK, the device and the ground into one MJCF
    string. `MYOASSIST_CACHE_DIR` turns on a cache of that string;
  * compile: `mujoco.MjModel.from_xml_string`, which runs on every build, cached or not.

Both are timed for the build each pipeline uses: RL keeps the freejoint root, CO sets
`planar_root=True` (22/26/80 only; CO does not support the full-body model). Each repeat is a
fresh process pinned to one physical core. Its first build is reported as cold (it also pays for
lazy imports and registry loading) and the next ones as warm. The warm figure is what a
long-lived worker pays per build, for example a CO pool process composing once per CMA-ES
candidate.

Cache modes:
  * off: `MYOASSIST_CACHE_DIR` unset, so every build composes from scratch;
  * hit: the variable points at a cache the driver filled beforehand, so compose is a file read.

`--device none` times the bare MSK (common.compose_bare_model: the same pipeline with the device
step skipped). The compose cache does not cover that path, so it runs uncached, RL build only.

Usage (repo root, project venv):
    python tools/benchmarks/build.py                                   # Tutorial_L1, 10 reps
    python tools/benchmarks/build.py --device Tutorial_L1 none
    python tools/benchmarks/build.py --msk myolegs22 --reps 2 --core -1
"""

import os
import sys

import common

# One thread per process: the workers are pinned to single cores. Set before numpy is imported.
if __name__ == "__main__":
    common.set_threads(1)

import argparse  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

CO_MSKS = {"myolegs22", "myolegs26", "myolegs"}  # ctrl_optim/optim/config/environment.py msk_to_musc
CACHE_ENV = "MYOASSIST_CACHE_DIR"


def worker(args):
    common.pin(args.core if args.core >= 0 else None)
    t0 = time.perf_counter()
    import mujoco
    import numpy as np
    from myoassist_utils.compose import compose_env_model

    import_s = time.perf_counter() - t0
    assert (os.environ.get(CACHE_ENV) is not None) == (args.cache == "hit"), "cache env does not match the case"
    builds = []
    for _ in range(1 + args.warm):
        t0 = time.perf_counter()
        if args.device == "none":
            xml = common.compose_bare_model(args.msk)
        else:
            xml = compose_env_model(args.msk, args.device, planar_root=args.build == "co")
        t1 = time.perf_counter()
        m = mujoco.MjModel.from_xml_string(xml)
        t2 = time.perf_counter()
        builds.append({"compose_s": t1 - t0, "compile_s": t2 - t1})
    common.emit(
        {
            "msk": args.msk,
            "device": args.device,
            "build": args.build,
            "cache": args.cache,
            "rep": args.rep,
            "core": args.core,
            "nu": m.nu,
            "nv": m.nv,
            "ngeom": m.ngeom,
            "xml_kb": round(len(xml) / 1024, 1),
            "import_s": round(import_s, 4),
            "cold_compose_s": round(builds[0]["compose_s"], 4),
            "cold_compile_s": round(builds[0]["compile_s"], 4),
            "warm_compose_s": round(float(np.mean([b["compose_s"] for b in builds[1:]])), 4),
            "warm_compile_s": round(float(np.mean([b["compile_s"] for b in builds[1:]])), 4),
        }
    )


def run_case(args, msk, device, build, cache, rep, cache_dir):
    env = {k: v for k, v in os.environ.items() if k != CACHE_ENV}
    if cache == "hit":
        env[CACHE_ENV] = str(cache_dir)
    cmd = [sys.executable, __file__, "--worker", "--msk", msk, "--device", device, "--build", build, "--cache", cache]
    cmd += ["--rep", str(rep), "--core", str(args.core), "--warm", str(args.warm)]
    return common.run_worker(cmd, env=env)


def driver(args):
    cache_dir = common.RESULTS / "compile_cache"
    shutil.rmtree(cache_dir, ignore_errors=True)
    cases = []  # (msk, device, build, cache)
    for msk in args.msk:
        for device in args.device:
            builds = ["rl"] if device == "none" or msk not in CO_MSKS else ["rl", "co"]
            caches = ["off"] if device == "none" else ["off", "hit"]
            cases += [(msk, device, b, c) for b in builds for c in caches]
    for msk, device, b, cache in cases:  # fill the cache once per build, so every "hit" reads an entry
        if cache == "hit":
            run_case(args, msk, device, b, cache, -1, cache_dir)

    runs = []
    for rep in range(args.reps):
        order = cases[rep % len(cases) :] + cases[: rep % len(cases)]  # rotate to spread drift
        for msk, device, b, cache in order:
            r = run_case(args, msk, device, b, cache, rep, cache_dir)
            runs.append(r)
            print(
                f"rep {rep:>2} {msk:<12} {device:<12} {b} cache={cache:<3} cold {r['cold_compose_s'] + r['cold_compile_s']:7.3f}s "
                f"| warm compose {r['warm_compose_s']:7.3f}s compile {r['warm_compile_s']:6.3f}s",
                flush=True,
            )
    shutil.rmtree(cache_dir, ignore_errors=True)

    summary = {}
    for msk, device, b, cache in cases:
        rs = [r for r in runs if (r["msk"], r["device"], r["build"], r["cache"]) == (msk, device, b, cache)]
        summary[f"{msk}@{device}@{b}@{cache}"] = {
            "msk": msk,
            "device": device,
            "build": b,
            "cache": cache,
            "nu": rs[0]["nu"],
            "ngeom": rs[0]["ngeom"],
            "xml_kb": rs[0]["xml_kb"],
            "cold_total_s": common.stats([r["cold_compose_s"] + r["cold_compile_s"] for r in rs]),
            "cold_compose_s": common.stats([r["cold_compose_s"] for r in rs]),
            "cold_compile_s": common.stats([r["cold_compile_s"] for r in rs]),
            "warm_total_s": common.stats([r["warm_compose_s"] + r["warm_compile_s"] for r in rs]),
            "warm_compose_s": common.stats([r["warm_compose_s"] for r in rs]),
            "warm_compile_s": common.stats([r["warm_compile_s"] for r in rs]),
            "import_s": common.stats([r["import_s"] for r in rs]),
        }
    settings = {"msk": args.msk, "device": args.device, "reps": args.reps, "warm": args.warm, "core": args.core}
    common.write_results("build", settings, summary, runs, args.out)
    for key, s in summary.items():
        print(f"  {key:<36} cold {s['cold_total_s']['mean']:7.3f}s  warm {s['warm_total_s']['mean']:7.3f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--msk", nargs="+", default=common.MSKS)
    ap.add_argument("--device", nargs="+", default=[common.DEVICE], help='"none" times the bare MSK')
    ap.add_argument("--reps", type=int, default=10, help="fresh processes per case")
    ap.add_argument("--warm", type=int, default=5, help="builds after the first, per process")
    ap.add_argument("--core", type=int, default=2, help="logical CPU to pin each worker to; -1 leaves it unpinned (macOS)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--build", choices=["rl", "co"], help=argparse.SUPPRESS)
    ap.add_argument("--cache", choices=["off", "hit"], help=argparse.SUPPRESS)
    ap.add_argument("--rep", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        assert len(args.msk) == 1 and len(args.device) == 1
        args.msk, args.device = args.msk[0], args.device[0]
        worker(args)
    else:
        driver(args)


if __name__ == "__main__":
    main()
