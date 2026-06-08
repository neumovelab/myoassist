"""Unified evaluation visualization for RL and Control Optimization pipelines.

Both pipelines emit `GaitData` JSON in the RL schema (see rl_train/analyzer/gait_data.py)
and a single skeleton-render frame. This module composes a single B&W summary figure:

    Row 1 (full width): skeleton snapshot
    Row 2: Return / CMA fitness panel | segmented R-leg joint angles vs reference
    Row 3: muscle activation grid (3 x 5; 12 muscles + Exo)
    Row 4: timeseries kinematics + foot sensor traces

Public entry point: `build_composite(...)`.
"""

from __future__ import annotations

import glob
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from rl_train.analyzer.gait_data import GaitData
from rl_train.analyzer.gait_analyze import GaitAnalyzer


# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------

JOINT_LIMIT = {"HIP": (-30, 35), "KNEE": (-70, 5), "ANKLE": (-30, 25)}

LINE_BLACK = "#000000"
LINE_GREY = "#666666"
FILL_GREY = "#bbbbbb"
REF_STYLE = dict(color=LINE_BLACK, linestyle="--", linewidth=1.0)
TOE_OFF_STYLE = dict(color=LINE_BLACK, linestyle="--", linewidth=0.8, alpha=0.5)


LINE_W_DATA = 1.6   # primary data traces (kinematics, muscles, CMA best)
LINE_W_REF = 1.2    # reference / median / worst overlays


def _ensure_gui_backend() -> None:
    """If the current matplotlib backend is non-interactive (e.g. Agg, picked up
    from upstream MyoSuite init), switch to TkAgg so plt.show() pops a window."""
    backend = matplotlib.get_backend().lower()
    if "agg" in backend and "tkagg" not in backend and "qtagg" not in backend:
        for candidate in ("TkAgg", "QtAgg"):
            try:
                plt.switch_backend(candidate)
                return
            except Exception:
                continue


def apply_style() -> None:
    """Apply a minimal B&W matplotlib style. Idempotent."""
    # Disable the navigation toolbar — its callbacks misfire on Tk backends after
    # MuJoCo/MyoSuite init opens and tears down hidden figures during env setup.
    matplotlib.rcParams["toolbar"] = "None"
    matplotlib.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": LINE_W_DATA,
        "figure.dpi": 110,
        "savefig.dpi": 200,
    })


# -----------------------------------------------------------------------------
# Helpers (segmentation, interpolation)
# -----------------------------------------------------------------------------

def _analyzer(gait_data: GaitData, ref_data: Optional[dict] = None) -> GaitAnalyzer:
    return GaitAnalyzer(gait_data, ref_data, show_plot=False)


def _segments_r(gait_data: GaitData):
    return _analyzer(gait_data).get_gait_segment_index(is_right_foot_based=True)


def _toe_off_pct(gait_data: GaitData, right: bool = True) -> float:
    return _analyzer(gait_data).get_toe_off_average(is_right_foot_based=right) * 100.0


def _segment_and_average(values: list, segments, length: int = 101):
    """Resample each gait segment to `length` samples on [0,100]; return (x, mean, std, all)."""
    x = np.linspace(0, 100, length)
    stacks = []
    for start, _toe, end in segments:
        seg = np.asarray([v[0] for v in values[start:end]])
        if len(seg) < 2:
            continue
        stacks.append(np.interp(x, np.linspace(0, 100, len(seg)), seg))
    if not stacks:
        return x, None, None, []
    arr = np.vstack(stacks)
    return x, arr.mean(axis=0), arr.std(axis=0), arr


# -----------------------------------------------------------------------------
# Panel: skeleton snapshot
# -----------------------------------------------------------------------------

SKEL_MAX_HEIGHT_IN = 3.5     # cap snapshot row height
SKEL_TARGET_ASPECT = 1920 / 960  # 2:1 — sized to fill a half-width snapshot column


def _normalize_skeleton(frame: np.ndarray, target_aspect: float = SKEL_TARGET_ASPECT) -> np.ndarray:
    """Defensive aspect normalization. Both CO and RL now render natively at
    1920x540 (RL forces buffer re-allocation in `render_snapshot`), so this is a
    no-op in practice. If a frame ever arrives at a different aspect we
    center-crop to target."""
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return frame
    src_aspect = w / h
    if abs(src_aspect - target_aspect) < 0.01:
        return frame
    if src_aspect > target_aspect:
        new_w = int(round(h * target_aspect))
        x0 = (w - new_w) // 2
        return frame[:, x0:x0 + new_w]
    new_h = int(round(w / target_aspect))
    y0 = (h - new_h) // 2
    return frame[y0:y0 + new_h]


def draw_skeleton(ax, frame: Optional[np.ndarray], title: Optional[str] = None) -> None:
    ax.axis("off")
    if frame is None:
        ax.text(0.5, 0.5, "No skeleton frame", ha="center", va="center",
                transform=ax.transAxes, color=LINE_GREY)
        return
    ax.imshow(frame, aspect="equal")
    if title:
        ax.set_title(title)


# -----------------------------------------------------------------------------
# Panel: return curve (RL) / CMA fitness curve (CO)
# -----------------------------------------------------------------------------

def draw_return_curve(ax, timesteps, returns) -> None:
    ax.plot(timesteps, returns, color=LINE_BLACK)
    ax.set_title("Return")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Return")
    ax.margins(x=0)


def draw_cma_fitness(ax, fbest, fmedian=None, fworst=None) -> None:
    """Draw f-best / f-median / f-worst per iteration. Any of median/worst may be None.
    The best curve is emphasized; median/worst are de-emphasized references."""
    it = np.arange(1, len(fbest) + 1)
    if fworst is not None and len(fworst) == len(fbest):
        ax.plot(it, fworst, color=LINE_GREY, linestyle=":", linewidth=0.8, label="worst")
    if fmedian is not None and len(fmedian) == len(fbest):
        ax.plot(it, fmedian, color=LINE_GREY, linestyle="-.", linewidth=0.8, label="median")
    ax.plot(it, fbest, color=LINE_BLACK, linestyle="-", linewidth=2.0, label="best")
    ax.set_title("CMA-ES Fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cost")
    ax.set_yscale("log")
    ax.margins(x=0)
    ax.legend(loc="upper right", frameon=False)


# -----------------------------------------------------------------------------
# Panel: segmented joint angles (R leg) vs reference
# -----------------------------------------------------------------------------

JOINT_KEYS = [("hip_flexion_r", "HIP", "hip"),
              ("knee_angle_r", "KNEE", "knee"),
              ("ankle_angle_r", "ANKLE", "ankle")]
REF_KEYS = {"hip_flexion_r": "q_hip_flexion_r",
            "knee_angle_r": "q_knee_angle_r",
            "ankle_angle_r": "q_ankle_angle_r"}


def draw_segmented_kinematics(axes, gait_data: GaitData, ref_data: Optional[dict]) -> None:
    """axes: iterable of 3 Axes (hip/knee/ankle, top to bottom). Independent autoscale."""
    segments = _segments_r(gait_data)
    toe_off = _toe_off_pct(gait_data, right=True)
    joint_data = gait_data.series_data["joint_data"]

    for ax, (jkey, _lim_key, label) in zip(axes, JOINT_KEYS):
        x, mean, std, _ = _segment_and_average(joint_data[jkey]["qpos"], segments)
        if mean is not None:
            mean_deg = np.rad2deg(mean)
            std_deg = np.rad2deg(std)
            ax.plot(x, mean_deg, color=LINE_BLACK, linestyle="-", label="Simulation")
            ax.fill_between(x, mean_deg - std_deg, mean_deg + std_deg,
                            color=FILL_GREY, alpha=0.5, linewidth=0)
        if ref_data is not None and REF_KEYS[jkey] in ref_data:
            ref = np.rad2deg(np.asarray(ref_data[REF_KEYS[jkey]]))
            xref = np.linspace(0, 100, len(ref))
            ax.plot(xref, ref, label="Reference", **REF_STYLE)
        ax.axvline(toe_off, **TOE_OFF_STYLE)
        ax.set_xlim(0, 100)
        ax.set_ylabel(label)
    # Only the bottom axis shows tick labels and x-label
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("Gait cycle (%)")
    # Only show legend when there's a reference overlay to disambiguate
    if ref_data is not None:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, loc="upper right", frameon=False, fontsize=6)


# -----------------------------------------------------------------------------
# Panel: joint kinetics (active actuator-driven joint moment, R leg)
# -----------------------------------------------------------------------------

def draw_kinetics(axes, gait_data: GaitData) -> None:
    """Draw active joint moments (Nm) per gait cycle for hip/knee/ankle (R leg).
    Reads `qfrc_actuator` populated by `GaitData.add_data`. Gracefully shows a
    placeholder for older gait JSONs that predate the moment capture."""
    segments = _segments_r(gait_data)
    toe_off = _toe_off_pct(gait_data, right=True)
    joint_data = gait_data.series_data["joint_data"]

    for ax, (jkey, _lim_key, label) in zip(axes, JOINT_KEYS):
        jdict = joint_data.get(jkey, {})
        qfrc = jdict.get("qfrc_actuator")
        if not qfrc:
            ax.text(0.5, 0.5, "no moment data\n(regenerate gait JSON)",
                    ha="center", va="center", transform=ax.transAxes,
                    color=LINE_GREY, fontsize=7)
            ax.set_xlim(0, 100)
            ax.set_ylabel(label)
            ax.set_yticks([])
            continue
        x, mean, std, _ = _segment_and_average(qfrc, segments)
        if mean is not None:
            ax.plot(x, mean, color=LINE_BLACK, linestyle="-", label="Simulation")
            ax.fill_between(x, mean - std, mean + std,
                            color=FILL_GREY, alpha=0.5, linewidth=0)
        ax.axvline(toe_off, **TOE_OFF_STYLE)
        ax.set_xlim(0, 100)
        ax.set_ylabel(label)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("Gait cycle (%)")


# -----------------------------------------------------------------------------
# Panel: muscle activation grid (R leg) + Exo torque
# -----------------------------------------------------------------------------

# Preferred muscle ordering (matches typical RL output); any unmatched ones are appended.
MUSCLE_ORDER = ["bifemsh", "edl", "fdl", "gastroc", "glutmax", "hamstrings",
                "iliopsoas", "rectfem", "soleus", "tibant", "vasti"]


def _collect_right_actuators(gait_data: GaitData) -> list[str]:
    names = [n for n in gait_data.series_data["actuator_data"].keys()
             if n.endswith("_r") or n.endswith("_R")]
    # Sort: known muscles first in order, then unknown muscles alphabetical, exo last
    def rank(n):
        base = n[:-2].lower()
        if "exo" in base:
            return (3, base)
        if base in MUSCLE_ORDER:
            return (1, MUSCLE_ORDER.index(base))
        return (2, base)
    return sorted(names, key=rank)


def draw_muscle_grid(axes_flat, gait_data: GaitData, n_cols: int = 4) -> None:
    """axes_flat: flat iterable of Axes (>= muscles + exo). Empty axes hidden.
    Activations on [0,1] scale; Exo on autoscaled torque (Nm)."""
    segments = _segments_r(gait_data)
    toe_off = _toe_off_pct(gait_data, right=True)
    actuator_data = gait_data.series_data["actuator_data"]
    names = _collect_right_actuators(gait_data)
    n_total = len(axes_flat)

    for i, ax in enumerate(axes_flat):
        if i >= len(names):
            ax.set_visible(False)
            continue
        name = names[i]
        ad = actuator_data[name]
        is_exo = "exo" in name.lower()
        if is_exo:
            x, mean, std, _ = _segment_and_average(ad["force"], segments)
            unit_label = name[:-2] + " (Nm)"
            if mean is not None:
                ax.plot(x, -mean, color=LINE_BLACK)
                ax.fill_between(x, -mean - std, -mean + std,
                                color=FILL_GREY, alpha=0.5, linewidth=0)
        else:
            x, mean, std, _ = _segment_and_average(ad["ctrl"], segments)
            unit_label = name[:-2]
            if mean is not None:
                mean_a = np.abs(mean)
                ax.plot(x, mean_a, color=LINE_BLACK)
                ax.fill_between(x, mean_a - std, mean_a + std,
                                color=FILL_GREY, alpha=0.5, linewidth=0)
                ax.set_ylim(0, 1)
        ax.axvline(toe_off, **TOE_OFF_STYLE)
        ax.set_xlim(0, 100)
        ax.set_title(unit_label, fontsize=8, pad=2)
        # Hide x-tick labels except on the bottom-most populated cell in each column
        row = i // n_cols
        col = i % n_cols
        last_row_in_col = (len(names) - 1 - col) // n_cols
        if row < last_row_in_col:
            ax.tick_params(labelbottom=False)


# -----------------------------------------------------------------------------
# Panel: timeseries kinematics + foot sensors
# -----------------------------------------------------------------------------

def draw_timeseries(axes, gait_data: GaitData) -> None:
    """axes: 5 Axes top->bot: right hip, right knee, right ankle, pelvis y, foot sensors."""
    joint_data = gait_data.series_data["joint_data"]
    sensor_data = gait_data.series_data["sensor_data"]

    def _ts(key):
        return np.rad2deg([v[0] for v in joint_data[key]["qpos"]])

    axes[0].plot(_ts("hip_flexion_r"), color=LINE_BLACK)
    axes[0].set_ylabel("hip (deg)")
    axes[1].plot(_ts("knee_angle_r"), color=LINE_BLACK)
    axes[1].set_ylabel("knee (deg)")
    axes[2].plot(_ts("ankle_angle_r"), color=LINE_BLACK)
    axes[2].set_ylabel("ankle (deg)")

    if "pelvis_ty" in joint_data:
        axes[3].plot([v[0] for v in joint_data["pelvis_ty"]["qpos"]], color=LINE_BLACK)
    axes[3].set_ylabel("pelvis y (m)")

    # Foot sensors: right (solid black), left (dashed grey)
    for key, style in (("r_foot", dict(color=LINE_BLACK, linestyle="-")),
                       ("r_toes", dict(color=LINE_BLACK, linestyle=":")),
                       ("l_foot", dict(color=LINE_GREY, linestyle="--")),
                       ("l_toes", dict(color=LINE_GREY, linestyle=":"))):
        if key in sensor_data:
            axes[4].plot([v[0] for v in sensor_data[key]["data"]], label=key, **style)
    axes[4].set_ylabel("sensors (N)")
    axes[4].legend(loc="upper right", frameon=False, ncol=4, fontsize=6)

    for ax in axes:
        ax.margins(x=0)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("timestep")


# -----------------------------------------------------------------------------
# Composite builder
# -----------------------------------------------------------------------------

@dataclass
class CompositeInputs:
    gait_data: GaitData
    skeleton_frame: Optional[np.ndarray] = None
    ref_data: Optional[dict] = None
    # Provide either return_curve (RL) OR cma_fitness (CO). The other panel slot is skipped.
    return_curve: Optional[tuple] = None       # (timesteps, returns)
    cma_fitness: Optional[tuple] = None        # (fbest, fmedian, fworst)
    title: Optional[str] = None
    metadata: Optional[dict] = None            # rendered as two-column text block under title


def build_composite(inputs: CompositeInputs,
                    *,
                    save_path: Optional[str] = None,
                    show: bool = True,
                    muscle_grid_shape: tuple = (3, 4)) -> plt.Figure:
    apply_style()
    if show:
        _ensure_gui_backend()
    plt.close("all")

    n_mrows, n_mcols = muscle_grid_shape
    n_timeseries = 5
    fig_width = 14.0

    # Snapshot row: normalize to a uniform wide aspect so RL and CO render
    # the snapshot at identical dimensions regardless of source resolution.
    skel_frame = inputs.skeleton_frame
    if skel_frame is not None:
        skel_frame = _normalize_skeleton(skel_frame)
        h, w = skel_frame.shape[:2]
        natural_h = fig_width * h / w
        skel_height_in = min(natural_h, SKEL_MAX_HEIGHT_IN)
    else:
        skel_height_in = 2.5

    # Row heights — give timeseries more vertical room
    kin_h = 3.0
    mus_h = 1.0 * n_mrows + 0.4
    ts_h = 5.0
    other_h = kin_h + mus_h + ts_h
    fig_height = skel_height_in + other_h + 2.4  # +pad for hspace + headers

    fig = plt.figure(figsize=(fig_width, fig_height))
    outer = fig.add_gridspec(
        nrows=4, ncols=1,
        height_ratios=[skel_height_in, kin_h, mus_h, ts_h],
        hspace=0.40,
    )

    # Section header helper — places bold text just above each row's gridspec cell
    def _section_header(spec, label: str, offset: float = 0.012) -> None:
        bbox = spec.get_position(fig)
        fig.text(0.5, bbox.y1 + offset, label,
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    if inputs.title:
        fig.suptitle(inputs.title, fontsize=13, fontweight="bold", y=0.995)

    # Metadata block — fills the space under the title with key:value pairs in two columns
    if inputs.metadata:
        items = list(inputs.metadata.items())
        mid = (len(items) + 1) // 2
        cols = (items[:mid], items[mid:])
        col_x = (0.32, 0.62)
        meta_y = 0.965
        for col_items, x in zip(cols, col_x):
            lines = [f"{k}:  {v}" for k, v in col_items]
            fig.text(x, meta_y, "\n".join(lines),
                     ha="left", va="top", fontsize=8, family="monospace",
                     color="#333333")

    # Row 1: skeleton snapshot (50%) + return/CMA panel (50%)
    row1 = outer[0].subgridspec(nrows=1, ncols=2, width_ratios=[1.0, 1.0], wspace=0.18)
    ax_skel = fig.add_subplot(row1[0, 0])
    draw_skeleton(ax_skel, skel_frame, title=None)
    _section_header(row1[0, 0], "Environment Snapshot")
    ax_perf = fig.add_subplot(row1[0, 1])
    if inputs.return_curve is not None:
        ts, rs = inputs.return_curve
        draw_return_curve(ax_perf, ts, rs)
    elif inputs.cma_fitness is not None:
        fb, fm, fw = inputs.cma_fitness
        draw_cma_fitness(ax_perf, fb, fm, fw)
    else:
        ax_perf.axis("off")

    # Row 2: kinematics (left) + kinetics (right), equal width
    row2 = outer[1].subgridspec(nrows=1, ncols=2, width_ratios=[1.0, 1.0], wspace=0.22)
    left2 = row2[0, 0].subgridspec(nrows=3, ncols=1, hspace=0.12)
    right2 = row2[0, 1].subgridspec(nrows=3, ncols=1, hspace=0.12)
    kin_axes = [fig.add_subplot(left2[i, 0]) for i in range(3)]
    kinetics_axes = [fig.add_subplot(right2[i, 0]) for i in range(3)]
    draw_segmented_kinematics(kin_axes, inputs.gait_data, inputs.ref_data)
    draw_kinetics(kinetics_axes, inputs.gait_data)
    # Per-block sub-headers so the two columns are clearly labeled
    kin_axes[0].set_title("Kinematics (deg)", fontsize=9, pad=4)
    kinetics_axes[0].set_title("Kinetics (Nm)", fontsize=9, pad=4)
    _section_header(outer[1], "Kinematics & Kinetics")

    # Row 3: muscle grid (3 rows x 4 cols by default)
    row3 = outer[2].subgridspec(nrows=n_mrows, ncols=n_mcols, hspace=0.65, wspace=0.30)
    mus_axes = [fig.add_subplot(row3[r, c]) for r in range(n_mrows) for c in range(n_mcols)]
    draw_muscle_grid(mus_axes, inputs.gait_data, n_cols=n_mcols)
    _section_header(outer[2], "Activation")

    # Row 4: timeseries (shared x via sharex pattern; only bottom shows tick labels)
    row4 = outer[3].subgridspec(nrows=n_timeseries, ncols=1, hspace=0.12)
    ts_axes = [fig.add_subplot(row4[i, 0]) for i in range(n_timeseries)]
    draw_timeseries(ts_axes, inputs.gait_data)
    _section_header(outer[3], "Timeseries Kinematics & Sensor Data")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Composite saved to {save_path}")
    if show:
        plt.show()
    return fig


# -----------------------------------------------------------------------------
# CMA loaders
# -----------------------------------------------------------------------------

def load_cma_fitness(results_dir: str, pkl_path: Optional[str] = None):
    """Return (fbest, fmedian, fworst) for the CMA-ES fitness panel.

    Prefers pycma's on-disk `outcmaes/*_fit.dat` (columns: iter, evals, sigma,
    axis_ratio, fbest, fmedian, fworst). Falls back to `es.fit.hist` from the pkl
    (best only) if outcmaes is missing — common for older runs.
    """
    fit_files = glob.glob(os.path.join(results_dir, "outcmaes", "*_fit.dat"))
    if fit_files:
        # Use the most recent if more than one prefix shows up in the dir.
        fit_path = max(fit_files, key=os.path.getmtime)
        try:
            arr = np.loadtxt(fit_path, comments="%")
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] >= 7:
                return arr[:, 4], arr[:, 5], arr[:, 6]
            return arr[:, 4], None, None
        except Exception as e:
            print(f"Warning: could not parse {fit_path}: {e}")

    # Fallback: read what we can from the pkl
    if pkl_path and os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            es = pickle.load(f)
        fbest = np.asarray(list(reversed(getattr(es.fit, "hist", []))))
        if fbest.size == 0:
            fbest = np.asarray([es.result.fbest])
        return fbest, None, None

    return None, None, None


# Back-compat alias for any external callers
load_cma_fitness_from_pkl = load_cma_fitness
