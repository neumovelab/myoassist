"""One figure comparing exo torque across runs, in each leg's own gait cycle.

The comparison that matters for this search cannot be read off the per-run composite: it is
the left and right exo profiles of several configs side by side, each plotted against *its
own* leg's phase. Segmenting both legs by right heel strike is what hid the problem for a
while -- the left profile then lands wherever the inter-leg phase offset puts it, so a policy
that assists push-off on one leg and early stance on the other looks merely noisy.

Reads the same rollout the analyzer already writes and reuses `score_exo_policy`, so the
figure and the numbers underneath it cannot disagree.

    .venv/bin/python tools/plot_exo_comparison.py rl_train/results/tutorial_30M_* -o out.png
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Importable from the repo root, which is where every documented command starts.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from score_exo_policy import ASSIST_WINDOW, N_PHASE, score_run  # noqa: E402

SUBSCORES = ("stability", "symmetry", "plausibility")


def _label(result: dict) -> str:
    """Config identity in the two axes that have been varied, plus the run name."""
    name = result["run"].replace("tutorial_30M_", "").replace("imitation_22_", "")
    mirror = result["mirror_coef"]
    return f"{name}\nactpen {result['muscle_activation_penalty']:g}" + (f", mirror {mirror:g}" if mirror else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", type=pathlib.Path)
    ap.add_argument("--mass", type=float, default=90.96)
    ap.add_argument("--joint", choices=sorted(ASSIST_WINDOW), default="ankle")
    ap.add_argument("-o", "--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    results = sorted((score_run(d, args.mass, args.joint) for d in args.run_dirs), key=lambda r: -r["total"])
    lo, hi = ASSIST_WINDOW[args.joint]
    phase = np.arange(N_PHASE)

    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(3.1 * n, 6.2), squeeze=False)
    y_max = max(max(r["mean_cycles"][s]) for r in results for s in ("r", "l")) / args.mass * 1.15

    for col, r in enumerate(results):
        ax = axes[0][col]
        ax.axvspan(lo * 100, hi * 100, color="0.88", zorder=0, label="assist window")
        for side, colour, style in (("r", "tab:blue", "-"), ("l", "tab:red", "--")):
            profile = np.asarray(r["mean_cycles"][side]) / args.mass
            ax.plot(phase, profile, style, color=colour, lw=1.8, label=f"Exo_{side.upper()}")
            ax.plot(profile.argmax(), profile.max(), "o", color=colour, ms=5)
        # Toe-off taken from this run's own contact data rather than a textbook 60 %: an
        # ankle assist that is still pushing after the foot has left the ground is doing
        # something non-physiological, and that is only visible against the measured value.
        toe_off = r["stability"]["stance_frac_mean"] * 100
        ax.axvline(toe_off, color="k", lw=0.9, ls=":", zorder=1)
        ax.text(toe_off + 1, y_max * 0.02, "toe-off", fontsize=6, rotation=90, va="bottom")
        ax.set_xlim(0, N_PHASE)
        ax.set_ylim(0, y_max)
        ax.set_title(_label(r), fontsize=8)
        ax.set_xlabel("gait cycle (%, own leg)", fontsize=8)
        if col == 0:
            ax.set_ylabel("|exo torque|  (N*m/kg)", fontsize=9)
            ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(labelsize=7)
        ax.text(
            0.98,
            0.95,
            f"peak {r['plausibility']['peak_phase_r'] * 100:.0f}%R / {r['plausibility']['peak_phase_l'] * 100:.0f}%L\n"
            f"in-window {r['plausibility']['in_window_frac_r'] * 100:.0f}% / {r['plausibility']['in_window_frac_l'] * 100:.0f}%",
            transform=ax.transAxes,
            fontsize=6.5,
            va="top",
            ha="right",
        )

        bx = axes[1][col]
        values = [r[k]["score"] for k in SUBSCORES] + [r["total"]]
        colours = ["tab:green", "tab:orange", "tab:purple", "0.3"]
        bx.bar(range(4), values, color=colours)
        for i, v in enumerate(values):
            bx.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.5)
        bx.set_xticks(range(4))
        bx.set_xticklabels(["stab", "symm", "plaus", "TOTAL"], fontsize=7)
        bx.set_ylim(0, 1.12)
        bx.tick_params(labelsize=7)
        if col == 0:
            bx.set_ylabel("score", fontsize=9)

    fig.suptitle(
        f"Exo torque per leg, each in its own gait cycle (shaded: {lo * 100:.0f}-{hi * 100:.0f}% {args.joint} assist window)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    for r in results:
        print(f"  {r['run']:52} total {r['total']:.3f}  symm {r['symmetry']['score']:.3f}")


if __name__ == "__main__":
    main()
