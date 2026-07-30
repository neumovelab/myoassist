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
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

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
TRAJ_LW = 1.0       # global scale for speed-coloured trajectory line widths


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


def draw_segmented_kinematics(axes, gait_data: GaitData, ref_data: Optional[dict],
                              norm=None, cmap=None) -> None:
    """axes: iterable of 3 Axes (hip/knee/ankle, top to bottom).

    Constant speed (norm is None): mean +- SD black trace.
    Varying speed (norm given):     one teal-coloured trace per stride, shaded by
                                    the stride's mean speed (see SPEED_CMAP)."""
    segments = _segments_r(gait_data)
    toe_off = _toe_off_pct(gait_data, right=True)
    joint_data = gait_data.series_data["joint_data"]
    vx = np.asarray([v[0] for v in joint_data["pelvis_tx"]["qvel"]], float)
    x = np.linspace(0, 100, 101)
    cmap = cmap or SPEED_CMAP

    for ax, (jkey, _lim_key, label) in zip(axes, JOINT_KEYS):
        if norm is None:
            _x, mean, std, _ = _segment_and_average(joint_data[jkey]["qpos"], segments)
            if mean is not None:
                mean_deg = np.rad2deg(mean)
                std_deg = np.rad2deg(std)
                ax.plot(_x, mean_deg, color=LINE_BLACK, linestyle="-", label="Simulation")
                ax.fill_between(_x, mean_deg - std_deg, mean_deg + std_deg,
                                color=FILL_GREY, alpha=0.5, linewidth=0)
        else:
            for (start, _toe, end) in segments:
                c = _resample_stride(joint_data[jkey]["qpos"], start, end)
                if c is None:
                    continue
                ax.plot(x, np.rad2deg(c), color=cmap(norm(vx[start:end].mean())),
                        lw=TRAJ_LW, alpha=0.85)
        if ref_data is not None and REF_KEYS[jkey] in ref_data:
            ref = np.rad2deg(np.asarray(ref_data[REF_KEYS[jkey]]))
            xref = np.linspace(0, 100, len(ref))
            ax.plot(xref, ref, label="Reference", **REF_STYLE)
        ax.axvline(toe_off, **TOE_OFF_STYLE)
        ax.set_xlim(0, 100)
        ax.set_ylabel(label)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("Gait cycle (%)")
    if ref_data is not None and norm is None:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, loc="upper right", frameon=False, fontsize=6)


# -----------------------------------------------------------------------------
# Panel: joint kinetics (active actuator-driven joint moment, R leg)
# -----------------------------------------------------------------------------

def draw_kinetics(axes, gait_data: GaitData, norm=None, cmap=None) -> None:
    """Draw active joint moments (Nm) per gait cycle for hip/knee/ankle (R leg).
    Reads `qfrc_actuator` populated by `GaitData.add_data`. Gracefully shows a
    placeholder for older gait JSONs that predate the moment capture.

    Constant speed (norm is None): mean +- SD. Varying (norm given): per-stride
    teal traces shaded by the stride's mean speed."""
    segments = _segments_r(gait_data)
    toe_off = _toe_off_pct(gait_data, right=True)
    joint_data = gait_data.series_data["joint_data"]
    vx = np.asarray([v[0] for v in joint_data["pelvis_tx"]["qvel"]], float)
    x = np.linspace(0, 100, 101)
    cmap = cmap or SPEED_CMAP

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
        if norm is None:
            _x, mean, std, _ = _segment_and_average(qfrc, segments)
            if mean is not None:
                ax.plot(_x, mean, color=LINE_BLACK, linestyle="-", label="Simulation")
                ax.fill_between(_x, mean - std, mean + std,
                                color=FILL_GREY, alpha=0.5, linewidth=0)
        else:
            for (start, _toe, end) in segments:
                c = _resample_stride(qfrc, start, end)
                if c is None:
                    continue
                ax.plot(x, c, color=cmap(norm(vx[start:end].mean())), lw=TRAJ_LW, alpha=0.85)
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


def draw_muscle_grid(axes_flat, gait_data: GaitData, n_cols: int = 4, norm=None, cmap=None) -> None:
    """axes_flat: flat iterable of Axes (>= muscles + exo). Empty axes hidden.
    Activations on [0,1] scale; Exo on autoscaled torque (Nm).

    Constant speed (norm is None): mean +- SD (black). Varying (norm given): one
    trace per stride, colour-mapped by the stride's mean speed with the same
    colormap as the kinematics (unified)."""
    segments = _segments_r(gait_data)
    toe_off = _toe_off_pct(gait_data, right=True)
    actuator_data = gait_data.series_data["actuator_data"]
    joint_data = gait_data.series_data["joint_data"]
    vx = np.asarray([v[0] for v in joint_data["pelvis_tx"]["qvel"]], float)
    xg = np.linspace(0, 100, 101)
    cmap = cmap or SPEED_CMAP
    names = _collect_right_actuators(gait_data)

    def _per_stride(values, transform):
        for (start, _toe, end) in segments:
            c = _resample_stride(values, start, end)
            if c is None:
                continue
            yield transform(c), cmap(norm(vx[start:end].mean()))

    for i, ax in enumerate(axes_flat):
        if i >= len(names):
            ax.set_visible(False)
            continue
        name = names[i]
        ad = actuator_data[name]
        is_exo = "exo" in name.lower()
        if is_exo:
            unit_label = name[:-2] + " (Nm)"
            if norm is None:
                x, mean, std, _ = _segment_and_average(ad["force"], segments)
                if mean is not None:
                    ax.plot(x, -mean, color=LINE_BLACK)
                    ax.fill_between(x, -mean - std, -mean + std,
                                    color=FILL_GREY, alpha=0.5, linewidth=0)
            else:
                for y, col in _per_stride(ad["force"], lambda c: -c):
                    ax.plot(xg, y, color=col, lw=0.8 * TRAJ_LW, alpha=0.85)
        else:
            unit_label = name[:-2]
            if norm is None:
                x, mean, std, _ = _segment_and_average(ad["ctrl"], segments)
                if mean is not None:
                    mean_a = np.abs(mean)
                    ax.plot(x, mean_a, color=LINE_BLACK)
                    ax.fill_between(x, mean_a - std, mean_a + std,
                                    color=FILL_GREY, alpha=0.5, linewidth=0)
            else:
                for y, col in _per_stride(ad["ctrl"], np.abs):
                    ax.plot(xg, y, color=col, lw=0.8 * TRAJ_LW, alpha=0.85)
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

def draw_timeseries(axes, gait_data: GaitData, norm=None, fs: float = 30.0, cmap=None) -> None:
    """axes: 5 Axes top->bot: right hip, right knee, right ankle, pelvis y, foot sensors.

    Constant speed (norm is None): solid black. Varying (norm given): the four
    continuous rows are colour-mapped point-wise by the low-pass filtered pelvis
    speed (same signal as the 'simulated (filtered)' speed trace), so the colour
    tracks the commanded speed instead of the raw per-step jitter."""
    joint_data = gait_data.series_data["joint_data"]
    sensor_data = gait_data.series_data["sensor_data"]
    vx = np.asarray([v[0] for v in joint_data["pelvis_tx"]["qvel"]], float)
    vx_c = _lowpass(vx, fs=fs) if norm is not None else vx
    cmap = cmap or SPEED_CMAP

    def _ts(key):
        return np.rad2deg([v[0] for v in joint_data[key]["qpos"]])

    kin = [("hip_flexion_r", "hip (deg)"), ("knee_angle_r", "knee (deg)"),
           ("ankle_angle_r", "ankle (deg)")]
    for ax, (key, ylabel) in zip(axes[:3], kin):
        if norm is None:
            ax.plot(_ts(key), color=LINE_BLACK)
        else:
            _colored_line(ax, _ts(key), vx_c, norm, cmap=cmap, lw=0.9)
        ax.set_ylabel(ylabel)

    if "pelvis_ty" in joint_data:
        py = [v[0] for v in joint_data["pelvis_ty"]["qpos"]]
        if norm is None:
            axes[3].plot(py, color=LINE_BLACK)
        else:
            _colored_line(axes[3], py, vx_c, norm, cmap=cmap, lw=0.9)
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
                    muscle_grid_shape: tuple = (3, 4),
                    speed_varying: bool = False,
                    fs: float = 30.0,
                    cmap=None,
                    mono_timeseries: bool = False) -> plt.Figure:
    """Unified B&W composite (fixed 4-row layout).

    Constant speed (speed_varying=False): the classic panels —
        Kinematics | Kinetics (mean +- SD), B&W activation and timeseries.
    Varying speed (speed_varying=True): same layout, no rows/cols added, but the
    two Row-2 slots are repurposed —
        left  = Speed tracking (commanded vs simulated) + step length + cadence
        right = Kinematics (R leg), one teal trace per stride shaded by stride speed
    and the Activation grid + Timeseries are velocity-colour-mapped (teal)."""
    apply_style()
    if show:
        _ensure_gui_backend()
    plt.close("all")
    cmap = cmap or SPEED_CMAP
    if speed_varying and SPEED_CMAP_CLIP != (0.0, 1.0):
        cmap = _truncate_cmap(cmap, *SPEED_CMAP_CLIP)

    n_mrows, n_mcols = muscle_grid_shape
    n_timeseries = 5
    fig_width = 14.0

    # Varying-speed prep: per-stride speeds -> colour normalization
    norm = None
    if speed_varying:
        _spd, *_ = stride_metrics(inputs.gait_data, _segments_r(inputs.gait_data), fs)
        if len(_spd):
            norm = Normalize(vmin=float(_spd.min()), vmax=float(_spd.max()))

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

    kin_h = 3.0
    mus_h = 1.0 * n_mrows + 0.4
    ts_h = 5.0
    heights = [skel_height_in, kin_h, mus_h, ts_h]
    fig_height = sum(heights) + 2.4

    fig = plt.figure(figsize=(fig_width, fig_height))
    outer = fig.add_gridspec(nrows=4, ncols=1, height_ratios=heights, hspace=0.40)

    def _section_header(spec, label: str, offset: float = 0.012) -> None:
        bbox = spec.get_position(fig)
        fig.text(0.5, bbox.y1 + offset, label,
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    if inputs.title:
        fig.suptitle(inputs.title, fontsize=13, fontweight="bold", y=0.995)

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

    # Row 2: two equal columns, repurposed for varying speed.
    row2 = outer[1].subgridspec(nrows=1, ncols=2, width_ratios=[1.0, 1.0], wspace=0.22)
    left2 = row2[0, 0].subgridspec(nrows=3, ncols=1, hspace=0.12)
    right2 = row2[0, 1].subgridspec(nrows=3, ncols=1, hspace=0.12)
    left_axes = [fig.add_subplot(left2[i, 0]) for i in range(3)]
    right_axes = [fig.add_subplot(right2[i, 0]) for i in range(3)]
    if norm is not None:
        # Keep kinematics in the LEFT slot (same position/size as the constant
        # composite); speed & gait metrics on the right.
        draw_segmented_kinematics(left_axes, inputs.gait_data, inputs.ref_data, norm=norm, cmap=cmap)
        draw_gait_metrics(right_axes, inputs.gait_data, fs, norm=norm, cmap=cmap)
        left_axes[0].set_title("Kinematics (deg)", fontsize=9, pad=4)
        right_axes[0].set_title("Speed & Gait Metrics", fontsize=9, pad=4)
        # Shared speed colorbar in a fixed axes at the figure's right margin, so
        # it does NOT shrink any panel (kinematics keeps its original width/aspect).
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        p_top = left_axes[0].get_position(); p_bot = left_axes[-1].get_position()
        p_right = right_axes[0].get_position()
        cax = fig.add_axes([p_right.x1 + 0.012, p_bot.y0, 0.009, p_top.y1 - p_bot.y0])
        fig.colorbar(sm, cax=cax).set_label("stride mean speed (m/s)", fontsize=7)
        _section_header(outer[1], "Kinematics & Speed / Gait Metrics")
    else:
        draw_segmented_kinematics(left_axes, inputs.gait_data, inputs.ref_data)
        draw_kinetics(right_axes, inputs.gait_data)
        left_axes[0].set_title("Kinematics (deg)", fontsize=9, pad=4)
        right_axes[0].set_title("Kinetics (Nm)", fontsize=9, pad=4)
        _section_header(outer[1], "Kinematics & Kinetics")

    # Row 3: muscle grid (velocity-colour-mapped when varying)
    row3 = outer[2].subgridspec(nrows=n_mrows, ncols=n_mcols, hspace=0.65, wspace=0.30)
    mus_axes = [fig.add_subplot(row3[r, c]) for r in range(n_mrows) for c in range(n_mcols)]
    draw_muscle_grid(mus_axes, inputs.gait_data, n_cols=n_mcols, norm=norm, cmap=cmap)
    _section_header(outer[2], "Activation")

    # Row 4: timeseries (velocity-colour-mapped when varying)
    row4 = outer[3].subgridspec(nrows=n_timeseries, ncols=1, hspace=0.12)
    ts_axes = [fig.add_subplot(row4[i, 0]) for i in range(n_timeseries)]
    draw_timeseries(ts_axes, inputs.gait_data,
                    norm=(None if mono_timeseries else norm), fs=fs, cmap=cmap)
    _section_header(outer[3], "Timeseries Kinematics & Sensor Data")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        fig.savefig(os.path.splitext(save_path)[0] + ".svg", bbox_inches="tight")
        print(f"  Composite saved to {save_path} (+ .svg)")
    if show:
        plt.show()
    return fig


# -----------------------------------------------------------------------------
# Varying-speed helpers (used by build_composite when speed_varying=True)
# -----------------------------------------------------------------------------
# For a single rollout with a time-varying commanded speed (SINUSOIDAL / STEP),
# every stride is coloured by its own mean speed (MyoSuite teal) instead of being
# collapsed to a mean +- SD. build_composite repurposes the Row-2 slots for a
# speed-tracking + step-length + cadence column alongside the coloured kinematics.

# Colour policy for varying-speed composites: ONLY two things carry hue —
#   (1) the speed colour map (applied to velocity-encoded "trajectories":
#       kinematics by-cycle, kinematics timeseries, and the speed trace), and
#   (2) the MyoSuite brand colour (green/teal), used as the single accent for
#       every other data line (step length, cadence, raw speed, muscle traces).
# Neutrals (black/grey) are used for references/axes only.
MYOSUITE_LIGHT = (0.314, 0.663, 0.667)
MYOSUITE_DARK = (0.129, 0.325, 0.341)
MYOSUITE_ACCENT = MYOSUITE_DARK           # single non-colormap accent colour
_TEAL_LIGHT_EXT = (0.690, 0.848, 0.850)
_TEAL_DARK_EXT = (0.058, 0.146, 0.153)
# teal option (slow=dark -> fast=light)
TEAL_CMAP = LinearSegmentedColormap.from_list(
    "myosuite_teal_wide",
    [_TEAL_DARK_EXT, MYOSUITE_DARK, MYOSUITE_LIGHT, _TEAL_LIGHT_EXT])
# blue -> light grey -> red (slow -> fast). A diverging map with a light neutral
# midpoint: mid speeds read as grey (clearly distinct from blue/red) and the map
# stays light overall instead of the dark look a purple midpoint gave.
BR_CMAP = LinearSegmentedColormap.from_list(
    "blue_grey_red",
    [(0.0, 0.0, 1.0), (0.78, 0.78, 0.78), (1.0, 0.0, 0.0)])
# fraction of the colormap used for the speed range (1.0 slice -> use full map)
SPEED_CMAP_CLIP = (0.0, 1.0)


def _truncate_cmap(cmap, lo=0.0, hi=1.0, n=256):
    """Return a colormap using only the [lo, hi] slice of `cmap`."""
    return LinearSegmentedColormap.from_list(
        f"{cmap.name}_clip", cmap(np.linspace(lo, hi, n)))
CMAPS = {"teal": TEAL_CMAP, "bluered": BR_CMAP, "rainbow": plt.get_cmap("rainbow")}
SPEED_CMAP = CMAPS["rainbow"]   # default speed colormap
VEL_CUTOFF_HZ = 2.0   # low-pass cutoff for the displayed pelvis velocity


def _lowpass(x, cutoff=VEL_CUTOFF_HZ, fs=30.0, order=4):
    from scipy.signal import butter, filtfilt
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    return filtfilt(b, a, np.asarray(x, float))


def _resample_stride(values, start, end, length=101):
    seg = np.asarray([v[0] for v in values[start:end]], float)
    if len(seg) < 3:
        return None
    return np.interp(np.linspace(0, 100, length), np.linspace(0, 100, len(seg)), seg)


def stride_metrics(gait_data: GaitData, segments, fs: float):
    """Per-stride mean speed (m/s), mid-time (s), cadence (steps/min), step length (m)."""
    jd = gait_data.series_data["joint_data"]
    vx = np.asarray([v[0] for v in jd["pelvis_tx"]["qvel"]], float)
    tx = np.asarray([v[0] for v in jd["pelvis_tx"]["qpos"]], float)
    speed, tmid, cadence, steplen = [], [], [], []
    for start, _toe, end in segments:
        if end - start < 2:
            continue
        speed.append(vx[start:end].mean())
        tmid.append(((start + end) / 2) / fs)
        T = (end - start) / fs
        cadence.append(120.0 / T)                # 2 steps per stride
        steplen.append((tx[end] - tx[start]) / 2.0)
    return (np.array(speed), np.array(tmid), np.array(cadence), np.array(steplen))


def draw_speed_tracking(ax, gait_data: GaitData, fs: float, norm=None, cmap=None) -> None:
    """Commanded (black dashed) vs simulated pelvis vx. Raw = faint MyoSuite accent;
    filtered = colour-mapped by speed (same scale as the kinematics) when `norm` is
    given, else MyoSuite accent."""
    cmap = cmap or SPEED_CMAP
    jd = gait_data.series_data["joint_data"]
    vx = np.asarray([v[0] for v in jd["pelvis_tx"]["qvel"]], float)
    tv = np.asarray([v[0] for v in gait_data.series_data["target_data"]["target_velocity"]], float)
    t = np.arange(len(vx)) / fs
    vf = _lowpass(vx, fs=fs)
    ax.plot(t, vx, color=MYOSUITE_ACCENT, lw=0.6, alpha=0.25, label="simulated (raw)")
    if norm is not None:
        _colored_line(ax, vf, vf, norm, cmap=cmap, lw=1.7, x=t)
    else:
        ax.plot(t, vf, color=MYOSUITE_ACCENT, lw=1.7, label="simulated (filtered)")
    ax.plot(t, tv, "--", color=LINE_BLACK, lw=1.8, label="commanded")
    ymin = float(min(vx.min(), tv.min())); ymax = float(max(vx.max(), tv.max()))
    pad = 0.05 * (ymax - ymin + 1e-9)
    ax.set_xlim(0, t[-1]); ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_ylabel("speed\n(m/s)"); ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=6)


def draw_gait_metrics(axes, gait_data: GaitData, fs: float, norm=None, cmap=None) -> None:
    """axes: 3 Axes top->bot: speed tracking (commanded vs simulated), step length,
    cadence. Shared time (s) x-axis; only the bottom shows tick labels.
    Step length / cadence use the MyoSuite accent colour."""
    segments = _segments_r(gait_data)
    _spd, tmid, cadence, steplen = stride_metrics(gait_data, segments, fs)
    xmax = gait_data.metadata["data_length"] / fs

    draw_speed_tracking(axes[0], gait_data, fs, norm=norm, cmap=cmap)
    axes[1].plot(tmid, steplen, "-o", color=LINE_BLACK, ms=3, lw=1)
    axes[1].set_ylabel("step length\n(m)")
    axes[2].plot(tmid, cadence, "-o", color=LINE_BLACK, ms=3, lw=1)
    axes[2].set_ylabel("cadence\n(steps/min)")
    for ax in axes:
        ax.set_xlim(0, xmax)
    for ax in axes[:-1]:
        ax.set_xlabel(""); ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("time (s)")


def _colored_line(ax, y, cvals, norm, cmap=None, lw=None, x=None):
    """Plot y as a poly-line coloured point-wise by cvals. x defaults to index."""
    cmap = cmap or SPEED_CMAP
    lw = 0.8 * TRAJ_LW if lw is None else lw
    y = np.asarray(y, float)
    x = np.arange(len(y)) if x is None else np.asarray(x, float)
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=norm)
    lc.set_array(np.asarray(cvals, float)[:-1])
    lc.set_linewidth(lw)
    ax.add_collection(lc)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(np.nanmin(y), np.nanmax(y))


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
