"""Shared helpers for the model cost benchmarks: paths, worker processes, statistics, system info.

Kept free of numpy/mujoco imports at module level, so each benchmark can set its thread
environment before anything that reads it is imported.
"""

import json
import os
import pathlib
import platform
import subprocess
import sys
from datetime import datetime
from importlib.metadata import version

REPO = pathlib.Path(__file__).resolve().parents[2]  # tools/benchmarks/<this file>
RESULTS = REPO / "tools" / "benchmarks" / "results"
MSKS = ["myolegs22", "myolegs26", "myolegs", "myofullbody"]
MUSCLES = {"myolegs22": 22, "myolegs26": 26, "myolegs": 80, "myofullbody": 416}
DEVICE = "Tutorial_L1"
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
PACKAGES = ("myo-sim", "assist-sim", "myoassist-terrains", "myosuite", "stable-baselines3", "torch", "gymnasium", "cma")

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def set_threads(n):
    """Thread count for OpenMP/BLAS in this process and its children. Call before importing numpy."""
    for var in THREAD_VARS:
        os.environ[var] = str(n)


def pin(core):
    """Pin this process to one logical CPU (no-op for None; macOS has no affinity API)."""
    if core is not None:
        import psutil

        psutil.Process().cpu_affinity([core])


def emit(result):
    """A worker's one line of output, read back by `collect`."""
    print("RESULT " + json.dumps(result), flush=True)


def _parse(stdout, stderr, returncode, what):
    assert returncode == 0, f"worker failed ({what}):\n{stderr[-3000:]}"
    lines = [ln for ln in stdout.splitlines() if ln.startswith("RESULT ")]
    assert len(lines) == 1, f"worker printed no result ({what}):\n{stdout[-2000:]}\n{stderr[-2000:]}"
    return json.loads(lines[0][len("RESULT ") :])


def run_worker(cmd, env=None, timeout=None):
    """Run one worker process to completion and return its RESULT."""
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env, timeout=timeout)
    return _parse(p.stdout, p.stderr, p.returncode, " ".join(cmd[2:8]))


def spawn_worker(cmd, env=None):
    """Start one worker process; `collect` waits for it."""
    return subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)


def collect(p):
    out, err = p.communicate()
    return _parse(out, err, p.returncode, " ".join(p.args[2:8]))


def stats(values):
    import numpy as np

    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "sd": None, "n": 0}
    return {
        "mean": round(float(np.mean(values)), 4),
        "sd": round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else None,
        "n": len(values),
    }


def cpu_name():
    if os.name == "nt":
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    if sys.platform == "darwin":
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def package_checkouts():
    """`{module: {"path", "commit", "modified"}}` for the packages imported from a git checkout."""
    import importlib.util

    out = {}
    for module in ("assist_sim", "myo_sim", "myoassist_terrains"):
        root = pathlib.Path(importlib.util.find_spec(module).origin).resolve().parents[1]
        if not (root / ".git").exists():
            continue
        git = ["git", "-C", str(root)]
        out[module] = {
            "path": str(root),
            "commit": subprocess.check_output([*git, "rev-parse", "--short", "HEAD"]).decode().strip(),
            "modified": subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=no"]).decode().splitlines(),
        }
    return out


def system_info():
    import mujoco
    import numpy as np
    import psutil

    freq = None if sys.platform == "darwin" else psutil.cpu_freq()  # unsupported on Apple silicon
    return {
        "cpu": cpu_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(logical=True),
        "cpu_freq_mhz_max": freq.max if freq else None,
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "os": platform.platform(),
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "packages": {p: version(p) for p in PACKAGES},
        # A package installed from a git checkout (editable) reports its release version
        # even when the checkout carries unreleased commits, so record the commit too.
        "package_checkouts": package_checkouts(),
        # Under Windows "Balanced", 416-muscle single-core runs split into two clusters ~25% apart.
        "power_scheme": subprocess.check_output(["powercfg", "/getactivescheme"]).decode().strip() if os.name == "nt" else None,
        "git_commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO).decode().strip(),
        # Tracked files that differ from the commit: the commit alone does not reproduce the
        # numbers when this is non-empty. Untracked files are left out; nothing here imports them.
        "git_modified": subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO)
        .decode()
        .splitlines(),
    }


def write_results(benchmark, settings, summary, runs, out=None, **extra):
    """Write one benchmark's results as JSON (default: results/<benchmark>_<time>.json) and return the path."""
    path = pathlib.Path(out) if out else RESULTS / f"{benchmark}_{datetime.now():%Y%m%d-%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "benchmark": benchmark,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "settings": settings,
        "system": system_info(),
        **extra,
        "summary": summary,
        "runs": runs,
    }
    path.write_text(json.dumps(doc, indent=2))
    print(f"\nwrote {path}", flush=True)
    return path


def compose_bare_model(msk):
    """The bare MSK through the compose pipeline: `compose_env_model` with the device step skipped.

    `_compose_env_model` gets the MSK + device from `assist_sim.load_combined`; this swaps in
    `assist_sim.load_msk`, the device-less loader, for one call, so the ground, render defaults
    and seating are the shipped ones. Freejoint root only (`load_msk` has no planar root).
    """
    import myoassist_utils.compose as compose_mod
    from assist_sim import load_msk

    def msk_only(msk_key, device_key, export_xml=None, planar_root=False):
        assert device_key is None and not planar_root
        return load_msk(msk_key, export_xml=export_xml)

    shipped = compose_mod.load_combined
    compose_mod.load_combined = msk_only
    try:
        return compose_mod._compose_env_model(msk, None)
    finally:
        compose_mod.load_combined = shipped
