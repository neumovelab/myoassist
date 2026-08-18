"""Joint kinematics and exo torque, both legs, each in its own gait cycle, against the reference.

This is the "what did the policy actually produce" figure: hip, knee and ankle angle for each
leg next to the mocap reference, with that leg's exo torque underneath, so a change in
assistance can be read against any change in the gait that produced it.

Two alignment details, both verified against the data rather than assumed:

  * `reference_data/segmented.npz` is segmented from a *single* event. Checking the circular
    shift that best aligns `q_hip_flexion_l` onto `q_hip_flexion_r` gives 50 %, so the left
    reference is the right one phase-shifted and has to be rolled by half a cycle before it
    can be compared to the left leg in the left leg's own cycle. Skipping this makes a correct
    left leg look half a cycle wrong -- the same trap that hid the exo asymmetry for a while.
  * reference angles are radians; the plots are degrees.

Sign convention is whatever the model uses, applied identically to simulation and reference,
so the comparison is valid even where a joint's positive direction is not the clinical one.

    .venv/bin/python tools/plot_kinematics_exo.py RUN_DIR [RUN_DIR ...] -o out.png
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from score_exo_policy import ASSIST_WINDOW, MIN_FLIGHT_S, N_PHASE, _cycles, _rising_edges, _series, _stance  # noqa: E402

# Joint label -> (observation key stem, reference key stem)
JOINTS = [
    ("hip", "hip_flexion", "q_hip_flexion"),
    ("knee", "knee_angle", "q_knee_angle"),
    ("ankle", "ankle_angle", "q_ankle_angle"),
]
SIDES = (("r", "tab:blue", "-"), ("l", "tab:red", "--"))


def _load(run_dir: pathlib.Path):
    import json

    candidates = sorted(run_dir.glob("gait_evaluated_data*.json"))
    assert candidates, f"no gait_evaluated_data*.json in {run_dir}"
    data = json.loads(candidates[0].read_text())
    cfg_path = run_dir / "session_config.json"
    if not cfg_path.exists():
        cfg_path = run_dir.parent / "session_config.json"
    return data["series_data"], json.loads(cfg_path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", type=pathlib.Path)
    ap.add_argument("--mass", type=float, default=90.96)
    ap.add_argument("--reference", type=pathlib.Path, default=pathlib.Path("rl_train/reference_data/segmented.npz"))
    ap.add_argument("-o", "--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    ref = np.load(args.reference, allow_pickle=True)
    lo, hi = ASSIST_WINDOW["ankle"]
    phase = np.arange(N_PHASE)

    n = len(args.run_dirs)
    fig, axes = plt.subplots(4, n, figsize=(4.0 * n, 9.6), squeeze=False, sharex=True)
    print(f"{'run':22} {'joint':6} {'RMSE vs reference (deg)':>26}")

    for col, run_dir in enumerate(args.run_dirs):
        series, cfg = _load(run_dir)
        fps = cfg["env_params"]["control_framerate"]
        stance = {
            s: _stance(
                _series(series["sensor_data"][f"{s}_foot"]),
                _series(series["sensor_data"][f"{s}_toes"]),
                max(2, round(MIN_FLIGHT_S * fps)),
            )
            for s in "rl"
        }
        strikes = {s: _rising_edges(stance[s]) for s in "rl"}

        for row, (label, obs_stem, ref_stem) in enumerate(JOINTS):
            ax = axes[row][col]
            for s, colour, style in SIDES:
                sim = np.degrees(_cycles(_series(series["joint_data"][f"{obs_stem}_{s}"]["qpos"]), strikes[s]).mean(axis=0))
                # Left reference is the right one shifted half a cycle (single-event segmentation).
                reference = np.degrees(np.asarray(ref[f"{ref_stem}_{s}"]))
                if s == "l":
                    reference = np.roll(reference, N_PHASE // 2)
                ax.plot(phase, sim, style, color=colour, lw=1.6, label=f"{s.upper()} sim")
                ax.plot(phase, reference, ":", color=colour, lw=1.2, alpha=0.75, label=f"{s.upper()} ref")
                print(f"{run_dir.name[:22]:22} {label + '_' + s:6} {np.sqrt(((sim - reference) ** 2).mean()):26.1f}")
            ax.set_ylabel(f"{label} (deg)", fontsize=9)
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(run_dir.name, fontsize=9)
                if col == 0:
                    ax.legend(fontsize=6, ncol=2, loc="lower left")

        ax = axes[3][col]
        ax.axvspan(lo * 100, hi * 100, color="0.9", zorder=0)
        for s, colour, style in SIDES:
            torque = (
                _cycles(np.abs(_series(series["actuator_data"][f"Exo_{s.upper()}"]["force"])), strikes[s]).mean(axis=0)
                / args.mass
            )
            ax.plot(phase, torque, style, color=colour, lw=1.8, label=f"Exo_{s.upper()}")
        ax.set_ylabel("|exo| (N*m/kg)", fontsize=9)
        ax.set_xlabel("gait cycle (%, own leg)", fontsize=9)
        ax.set_xlim(0, N_PHASE)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Kinematics against mocap reference (dotted) and exo torque, each leg in its own cycle", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
