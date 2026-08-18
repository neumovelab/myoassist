"""Score a trained exo policy on stability, left/right symmetry and biological plausibility.

Motivation: the config search for "good exo torque" was being decided by looking at plots,
which is fine for spotting a disaster and useless for ranking two runs that both walk. This
turns each of the three words into numbers read off the evaluation rollout that
`train_analyzer` already writes (`gait_evaluated_data_00.json`), so configs can be ordered.

Every gait-cycle quantity is computed in the *side's own* cycle. Segmenting both exo signals
by right heel strike is what previously made a 4:1 left/right torque split look symmetric:
the left profile got plotted against the right leg's phase, so its push-off landed wherever
the phase offset put it.

Reference values (anchors for the plausibility scores) are literature values for walking at
a self-selected speed, and each is named where it is used:

  * ankle plantarflexion moment peaks at ~50 % of the gait cycle, just before toe-off
    (Winter, *Biomechanics and Motor Control of Human Movement*, 4th ed.).
  * powered ankle exoskeletons deliver a peak of roughly 0.15-0.80 N*m/kg; below that the
    device is doing nothing useful, above it the assistance stops resembling augmentation.
    The upper bound is the human-in-the-loop optimum of Zhang et al. 2017, 0.77 +/- 0.08
    N*m/kg (see PEAK_BAND_NM_PER_KG, which this must stay consistent with).
  * stance occupies ~60 % of the cycle at walking speeds, and both feet leave the ground in
    running, so a stance fraction far below that is a different gait, not a better one.

The hip windows are provisional -- a hip flexion assist has two physiological lobes rather
than one -- and only the ankle path has been checked against composed models, so scoring a
hip device prints a warning rather than pretending the window is established.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

# Assist window as a fraction of the gait cycle, per assisted joint.
#   ankle: from the earliest reported torque onset to toe-off. Zhang et al. 2017 report
#          optimised onsets between 20 % and 40 % of the stride, so the window opens at 20 %.
#   hip:   provisional -- extension assist in early stance, flexion assist around toe-off.
ASSIST_WINDOW = {
    "ankle": (0.20, 0.65),
    "hip": (0.00, 0.20),
}
# Peak torque a powered ankle exo plausibly delivers, N*m/kg of body mass. The upper bound is
# the human-in-the-loop optimum of 0.77 +- 0.08 N*m/kg measured by Zhang et al. 2017 (Science
# 356:1280), whose optimised peak timing of 50.3 +- 1.4 % of the cycle is also what the peak
# phase is scored against; the lower bound is around the autonomous-device magnitudes reported
# by Mooney & Herr 2016, below which a device is not doing useful work.
#
# This started at (0.15, 0.50) from recollection rather than a source, which put the
# best-established value in the field outside the "plausible" band and charged a policy for
# reaching it. Corrected after a run was penalised for a 0.67 N*m/kg peak -- so the correction
# favours that run, and the effect on the ranking is reported rather than absorbed.
PEAK_BAND_NM_PER_KG = (0.15, 0.80)
# myolegs22 + a drop-in device; the device shells are light enough not to move this much.
DEFAULT_BODY_MASS_KG = 90.96
# Cycles resampled onto this many phase points, so 1 point == 1 % of the cycle.
N_PHASE = 100
# Unloaded stretches shorter than this are contact-sensor dropouts, not a flight phase.
MIN_FLIGHT_S = 0.15


# Devices whose actuator drives a tendon rather than a joint. For a joint transmission MuJoCo's
# `actuator_force` *is* the joint torque -- verified on the composed models, where commanding
# Tutorial_L1 to -1 gives actuator_force -100.0 against -99.25 N*m at the ankle, and Hippo_L1
# +1 gives 100.0 against 114.18 N*m at the hip. For a tendon transmission it is cable tension
# and bears no fixed relation to the joint torque: STRIDE_L2 at 400 delivers 0.80 N*m to the
# ankle, HMEDI_L1 at -100 delivers -6.13 N*m to the hip. Dividing that by body mass and calling
# it N*m/kg is meaningless, so the peak-magnitude band is not applied to these and the number is
# reported as cable tension per kilogram instead.
TENDON_DRIVEN_DEVICES = {"STRIDE_L2", "HMEDI_L1", "UTAnkleExo_L2"}


def _series(node) -> np.ndarray:
    """Flatten one series to shape (T,).

    The dump is not uniform: sensors are stored as `{"data": [[v], ...]}` while actuator and
    joint channels are the bare `[[v], ...]`, so accept either rather than making the caller
    remember which block it is reading from.
    """
    arr = np.asarray(node["data"] if isinstance(node, dict) else node, dtype=float)
    assert arr.ndim == 2 and arr.shape[1] == 1, f"unexpected series shape {arr.shape}"
    return arr[:, 0]


# The 22 muscles of myolegs22. Anything else in `actuator_data` belongs to the device.
_MUSCLE_STEMS = (
    "hamstrings",
    "bifemsh",
    "edl",
    "fdl",
    "glutmax",
    "iliopsoas",
    "rectfem",
    "vasti",
    "gastroc",
    "soleus",
    "tibant",
)
_SIDE_MARKERS = (("_r", "_l"), ("_R", "_L"), ("_dx", "_sx"), ("_DX", "_SX"))


def _device_actuators(actuator_data: dict) -> dict:
    """Map {"r": name, "l": name} for the device's two actuators, found rather than assumed.

    The eight sweep devices use four different naming conventions -- `Exo_R`, `HMEDI_L1_Exo_R`,
    `STRIDE_L2_cable_r`, `UTAnkleExo_L2_part2part3act_dx` -- so hardcoding `Exo_R`/`Exo_L` scored
    only some of them and raised KeyError on the rest.
    """
    muscles = {f"{stem}_{side}" for stem in _MUSCLE_STEMS for side in ("r", "l")}
    device = [k for k in actuator_data if k not in muscles]
    assert device, f"no device actuator among {sorted(actuator_data)[:6]}..."
    for right_mark, left_mark in _SIDE_MARKERS:
        right = [k for k in device if k.endswith(right_mark)]
        left = [k for k in device if k.endswith(left_mark)]
        if len(right) == 1 and len(left) == 1:
            return {"r": right[0], "l": left[0]}
    raise AssertionError(f"cannot tell the sides apart for device actuators {device}")


def _hip_flexion_strikes(series: dict, side: str, shift_fraction: float = 0.07) -> np.ndarray:
    """Foot-strike indices from peak hip flexion, for devices whose contact sensors are unusable.

    Not every composed model reports foot contact. A device that puts a sole under the foot takes
    over the ground contact, and the `*_foot` / `*_toes` sensors then read exactly zero for the
    whole rollout: `DephyExoBoot_L1` and `Humotech_L1` do this, so their gait cannot be segmented
    from contact at all. `OpenExo_L1` is worse than useless -- it reports 12 strikes for 6 strides,
    double-triggering where device and human geoms both touch.

    Peak hip flexion happens in terminal swing, a fixed fraction of a cycle before foot strike.
    Measured against contact-based strikes on the three devices whose sensors do work, it leads
    them by 6.1 %, 6.9 % and 9.2 % of the cycle with a standard deviation of 0.0-1.3 %, so shifting
    the peaks by 7 % recovers foot strike to within a few percent of a cycle. That is well inside
    the 20-65 % assist window this is used to interrogate.
    """
    from scipy.signal import find_peaks

    hip = _series(series["joint_data"][f"hip_flexion_{side}"]["qpos"])
    peaks, _ = find_peaks(hip, distance=15, prominence=0.15)
    if len(peaks) < 2:
        return peaks
    stride = float(np.mean(np.diff(peaks)))
    shifted = np.round(peaks + shift_fraction * stride).astype(int)
    return shifted[(shifted >= 0) & (shifted < len(hip))]


def _contact_segmentation_is_usable(strikes: np.ndarray, hip_strikes: np.ndarray) -> bool:
    """Whether contact-derived strikes can be trusted for this rollout.

    Checked against the hip-flexion count rather than in isolation, because both failure modes
    seen in the sweep are count errors: none at all when a sole carries the load, and roughly
    double when device and human geoms both contact.
    """
    if len(strikes) < 3 or len(hip_strikes) < 3:
        return len(strikes) >= 3
    return abs(len(strikes) - len(hip_strikes)) <= 1


def _stance(foot: np.ndarray, toes: np.ndarray, min_flight_frames: int) -> np.ndarray:
    """Boolean stance mask for one leg, from the heel and toe contact forces.

    Two things this has to get right, both learned from the raw dump:

      * stance is heel *or* toe contact. The `*_foot` sensor unloads at heel-off while the
        `*_toes` sensor carries the rest of stance, so reading `*_foot` alone reports a
        stance fraction near 41 %, which looks like a bouncing gait when the union gives the
        physiological ~59 %.
      * contact forces drop to exactly zero for isolated frames in mid-stance. Every such
        dropout is otherwise a spurious heel strike, and a two-frame "cycle" between a pair
        of them cannot be resampled at all.
    """
    loaded = (foot > 0.0) | (toes > 0.0)
    # Close unloaded runs shorter than a plausible flight phase.
    padded = np.concatenate([[True], loaded, [True]])
    edges = np.flatnonzero(padded[:-1] != padded[1:])
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start < min_flight_frames:
            loaded[start:end] = True
    return loaded


def _rising_edges(stance: np.ndarray) -> np.ndarray:
    """Indices where a leg enters stance (foot strike)."""
    return np.flatnonzero(~stance[:-1] & stance[1:]) + 1


def _cycles(signal: np.ndarray, strikes: np.ndarray) -> np.ndarray:
    """Resample `signal` between consecutive heel strikes onto N_PHASE points, (n_cycles, 100)."""
    phase = np.linspace(0.0, 1.0, N_PHASE, endpoint=False)
    out = []
    for start, end in zip(strikes[:-1], strikes[1:]):
        seg = signal[start:end]
        assert len(seg) >= 4, f"cycle of {len(seg)} frames is too short to resample"
        out.append(np.interp(phase, np.linspace(0.0, 1.0, len(seg), endpoint=False), seg))
    assert out, "no complete gait cycle in this rollout"
    return np.asarray(out)


def _band(x: float, lo: float, hi: float, width: float) -> float:
    """1.0 inside [lo, hi], falling linearly to 0 once `width` past either edge."""
    if lo <= x <= hi:
        return 1.0
    over = (lo - x) if x < lo else (x - hi)
    return float(max(0.0, 1.0 - over / width))


def _lower_better(x: float, good: float, bad: float) -> float:
    """1.0 at or below `good`, 0.0 at or above `bad`."""
    return float(np.clip((bad - x) / (bad - good), 0.0, 1.0))


def _n_prominent_peaks(profile: np.ndarray) -> int:
    """Local maxima of a cyclic profile rising at least 25 % of the range above their valleys."""
    from scipy.signal import find_peaks

    span = profile.max() - profile.min()
    if span <= 0.0:
        return 0
    # Wrapped so a peak sitting on the cycle boundary is not split in two.
    wrapped = np.concatenate([profile, profile, profile])
    peaks, _ = find_peaks(wrapped, prominence=0.25 * span)
    in_middle = peaks[(peaks >= N_PHASE) & (peaks < 2 * N_PHASE)]
    return int(len(in_middle))


def _side_metrics(torque: np.ndarray, stance_mask: np.ndarray, strikes: np.ndarray, dt: float, mass: float) -> dict:
    """Per-leg exo torque description, in that leg's own gait cycle."""
    mag = np.abs(torque)
    cyc = _cycles(mag, strikes)
    mean_cycle = cyc.mean(axis=0)
    stance = _cycles(stance_mask.astype(float), strikes).mean(axis=0) > 0.5

    peak = float(mean_cycle.max())
    total = float(mean_cycle.sum())
    lo, hi = ASSIST_WINDOW["ankle"]  # replaced by the caller for hip devices
    return {
        "peak_nm": peak,
        "peak_nm_per_kg": peak / mass,
        "peak_phase": float(mean_cycle.argmax()) / N_PHASE,
        "impulse_nms": float(mag[strikes[0] : strikes[-1]].sum() * dt / (len(strikes) - 1)),
        "swing_impulse_frac": float(mean_cycle[~stance].sum() / total) if total > 0 else 0.0,
        "stance_frac": float(stance.mean()),
        "n_peaks": _n_prominent_peaks(mean_cycle),
        # Spread across cycles at the same phase, relative to the peak: how repeatable the
        # profile is from stride to stride.
        "cycle_scatter": float(cyc.std(axis=0).mean() / peak) if peak > 0 else 0.0,
        # Mean absolute slew over one cycle, scaled to the peak: chatter shows up here even
        # when the mean profile looks smooth.
        "slew_per_cycle": float(np.abs(np.diff(mean_cycle)).sum() / peak) if peak > 0 else 0.0,
        "_mean_cycle": mean_cycle,
        "_window": (lo, hi),
    }


def score_run(run_dir: pathlib.Path, mass: float, joint: str) -> dict:
    # Two writers, two names: the training analyzer indexes by evaluate_param_list entry
    # ("..._00.json") while run_policy_eval writes a single unindexed file. Accepting both is
    # what lets the same scorer read a 200-step checkpoint evaluation and a long
    # `run_policy_eval --steps 1000` rollout of the finished policy.
    candidates = sorted(run_dir.glob("gait_evaluated_data*.json"))
    assert candidates, f"no gait_evaluated_data*.json in {run_dir}"
    data = json.loads(candidates[0].read_text())
    # Training writes one session_config.json per session and one analyze_results_<steps>_00
    # directory per periodic evaluation beneath it, so pointing this at a checkpoint's
    # evaluation -- which is how a score-versus-steps trajectory gets built -- finds the
    # config one level up rather than beside the rollout.
    config_path = run_dir / "session_config.json"
    if not config_path.exists():
        config_path = run_dir.parent / "session_config.json"
    assert config_path.exists(), f"no session_config.json in {run_dir} or its parent"
    config = json.loads(config_path.read_text())
    series = data["series_data"]
    dt = 1.0 / config["env_params"]["control_framerate"]
    target_v = config["evaluate_param_list"][0]["max_target_velocity"]

    if joint == "hip":
        print(f"  warning: hip assist window {ASSIST_WINDOW['hip']} is provisional (single lobe assumed)")
    lo, hi = ASSIST_WINDOW[joint]
    window = (np.arange(N_PHASE) / N_PHASE >= lo) & (np.arange(N_PHASE) / N_PHASE < hi)

    min_flight = max(2, round(MIN_FLIGHT_S * config["env_params"]["control_framerate"]))
    stance = {
        s: _stance(
            _series(series["sensor_data"][f"{s}_foot"]),
            _series(series["sensor_data"][f"{s}_toes"]),
            min_flight,
        )
        for s in ("r", "l")
    }
    strikes = {s: _rising_edges(stance[s]) for s in ("r", "l")}
    segmentation = "contact"
    hip_strikes = {s: _hip_flexion_strikes(series, s) for s in ("r", "l")}
    if not all(_contact_segmentation_is_usable(strikes[s], hip_strikes[s]) for s in ("r", "l")):
        # Fall back rather than fail: the contact signal is a property of the device's geometry,
        # not of the policy, and a device whose sole carries the load reports nothing at all.
        strikes = hip_strikes
        segmentation = "hip-flexion"
        # Stance is still needed for the swing-phase share; without contact, take the
        # literature ~60 % of the cycle rather than pretending to measure it.
        stance = {s: np.zeros(len(_series(series["joint_data"]["pelvis_tx"]["qpos"])), dtype=bool) for s in ("r", "l")}
        for s in ("r", "l"):
            for a, b in zip(strikes[s][:-1], strikes[s][1:]):
                stance[s][a : a + int(0.6 * (b - a))] = True
    for s in ("r", "l"):
        assert len(strikes[s]) >= 3, (
            f"{s}: only {len(strikes[s])} strikes from {segmentation} segmentation -- rollout too short to score"
        )

    exo_names = _device_actuators(series["actuator_data"])
    sides = {}
    for s, act in (("r", exo_names["r"]), ("l", exo_names["l"])):
        m = _side_metrics(_series(series["actuator_data"][act]["force"]), stance[s], strikes[s], dt, mass)
        m["in_window_impulse_frac"] = float(m["_mean_cycle"][window].sum() / m["_mean_cycle"].sum())
        sides[s] = m

    # --- stability -------------------------------------------------------------------
    pelvis_x = _series(series["joint_data"]["pelvis_tx"]["qpos"])
    pelvis_y = _series(series["joint_data"]["pelvis_ty"]["qpos"])
    stride_times = np.concatenate([np.diff(strikes[s]) * dt for s in ("r", "l")])
    distance = float(pelvis_x[-1] - pelvis_x[0])
    mean_speed = distance / (len(pelvis_x) * dt)
    stability = {
        "n_strides_r": len(strikes["r"]) - 1,
        "n_strides_l": len(strikes["l"]) - 1,
        "distance_m": distance,
        "mean_speed_mps": mean_speed,
        "speed_error_mps": abs(mean_speed - target_v),
        "pelvis_height_std_m": float(pelvis_y.std()),
        "stride_time_cv": float(stride_times.std() / stride_times.mean()),
        "stance_frac_mean": float(np.mean([sides[s]["stance_frac"] for s in ("r", "l")])),
    }
    stability["score"] = float(
        np.mean(
            [
                _lower_better(stability["speed_error_mps"], 0.05, 0.40),
                _lower_better(stability["pelvis_height_std_m"], 0.010, 0.060),
                _lower_better(stability["stride_time_cv"], 0.03, 0.25),
                # Winter: ~0.60 stance at walking speed. Below ~0.50 the model is bouncing.
                _band(stability["stance_frac_mean"], 0.55, 0.65, 0.15),
            ]
        )
    )

    # --- symmetry --------------------------------------------------------------------
    def ratio(key: str) -> float:
        a, b = sides["r"][key], sides["l"][key]
        return float(min(a, b) / max(a, b)) if max(a, b) > 0 else 0.0

    phase_diff = abs(sides["r"]["peak_phase"] - sides["l"]["peak_phase"])
    phase_diff = min(phase_diff, 1.0 - phase_diff)  # cyclic
    profile_corr = float(np.corrcoef(sides["r"]["_mean_cycle"], sides["l"]["_mean_cycle"])[0, 1])
    symmetry = {
        "peak_ratio": ratio("peak_nm"),
        "impulse_ratio": ratio("impulse_nms"),
        "peak_phase_diff": phase_diff,
        "profile_corr": profile_corr,
    }
    symmetry["score"] = float(
        np.mean(
            [
                symmetry["peak_ratio"],
                symmetry["impulse_ratio"],
                _lower_better(phase_diff, 0.03, 0.20),
                float(np.clip(profile_corr, 0.0, 1.0)),
            ]
        )
    )

    # --- biological plausibility ------------------------------------------------------
    tendon_driven = config["env_params"].get("device_key") in TENDON_DRIVEN_DEVICES
    per_side = []
    for s in ("r", "l"):
        m = sides[s]
        terms = [
            _band(m["peak_phase"], lo, hi, 0.15),
            m["in_window_impulse_frac"],
            _lower_better(m["swing_impulse_frac"], 0.05, 0.40),
        ]
        if not tendon_driven:
            terms.append(_band(m["peak_nm_per_kg"], *PEAK_BAND_NM_PER_KG, 0.25))
        per_side.append(
            np.mean(
                terms
                + [
                    1.0 if m["n_peaks"] == 1 else 0.0,
                    _lower_better(m["cycle_scatter"], 0.10, 0.60),
                    # One rise and one fall over a cycle is a slew of ~2 peak-units; more
                    # than that is oscillation on top of the profile.
                    _lower_better(m["slew_per_cycle"], 2.5, 8.0),
                ]
            )
        )
    plausibility = {
        "peak_phase_r": sides["r"]["peak_phase"],
        "peak_phase_l": sides["l"]["peak_phase"],
        "peak_nm_per_kg_r": sides["r"]["peak_nm_per_kg"],
        "peak_nm_per_kg_l": sides["l"]["peak_nm_per_kg"],
        "in_window_frac_r": sides["r"]["in_window_impulse_frac"],
        "in_window_frac_l": sides["l"]["in_window_impulse_frac"],
        "swing_frac_r": sides["r"]["swing_impulse_frac"],
        "swing_frac_l": sides["l"]["swing_impulse_frac"],
        # The decision-relevant magnitude for a bilateral device, and the only seed-invariant
        # way to state it. A penalty-trained policy abandons one leg, but *which* leg is set
        # by the initialisation draw: at one seed the left peak fell to 0.10 N*m/kg, at
        # another the right fell to 0.06. Naming a side would have reported the seed.
        "weaker_leg_peak_nm_per_kg": min(sides["r"]["peak_nm_per_kg"], sides["l"]["peak_nm_per_kg"]),
        "n_peaks_r": sides["r"]["n_peaks"],
        "n_peaks_l": sides["l"]["n_peaks"],
        "cycle_scatter_r": sides["r"]["cycle_scatter"],
        "cycle_scatter_l": sides["l"]["cycle_scatter"],
        "slew_r": sides["r"]["slew_per_cycle"],
        "slew_l": sides["l"]["slew_per_cycle"],
        "score": float(np.mean(per_side)),
    }

    # --- absolute-time diagnostic (reported, not scored) -------------------------------
    # Independent of the strike detection, and it names the failure mode rather than just
    # measuring it: two symmetric exos must fire half a stride apart, so a zero-lag
    # correlation near +1 means a single un-lateralised command is driving both actuators.
    # Kept out of the score because the peak-phase gap already counts this evidence once.
    stride_frames = float(np.mean(np.concatenate([np.diff(strikes[s]) for s in ("r", "l")])))
    abs_time = {"stride_frames": stride_frames}
    r_sig, l_sig = (np.abs(_series(series["actuator_data"][exo_names[k]]["force"])) for k in ("r", "l"))
    if r_sig.std() > 0 and l_sig.std() > 0:
        rz, lz = ((v - v.mean()) / v.std() for v in (r_sig, l_sig))
        xcorr = np.correlate(rz, lz, "full") / len(rz)
        lags = np.arange(-len(rz) + 1, len(rz))
        near = np.abs(lags) <= round(stride_frames)
        abs_time["zero_lag_corr"] = float(xcorr[lags == 0][0])
        abs_time["best_lag_frames"] = int(lags[near][xcorr[near].argmax()])
        abs_time["best_lag_corr"] = float(xcorr[near].max())
        abs_time["best_lag_stride_frac"] = abs_time["best_lag_frames"] / stride_frames
    else:
        abs_time["one_side_silent"] = True

    # Exclude the device by the actuator names already resolved above, not by an "Exo" prefix:
    # only some devices use it, so STRIDE_L2_cable_r was being averaged into the muscle RMS.
    muscle_rms = float(
        np.sqrt(np.mean([_series(v["ctrl"]) ** 2 for k, v in series["actuator_data"].items() if k not in exo_names.values()]))
    )
    return {
        # Session and evaluation directory, because the analyze directory name is the step count
        # and collides across sessions: scoring seven devices' 29245440 checkpoints produced seven
        # identically labelled rows.
        "run": f"{run_dir.parent.name}/{run_dir.name}" if run_dir.parent.name.startswith("train_session") else run_dir.name,
        "device": config["env_params"].get("device_key"),
        "segmentation": segmentation,
        "tendon_driven": tendon_driven,
        "muscle_activation_penalty": config["env_params"]["reward_keys_and_weights"]["muscle_activation_penalty"],
        "mirror_coef": config["ppo_params"].get("mirror_coef", 0.0),
        "muscle_ctrl_rms": muscle_rms,
        "stability": stability,
        "symmetry": symmetry,
        "plausibility": plausibility,
        "abs_time": abs_time,
        "total": float(np.mean([stability["score"], symmetry["score"], plausibility["score"]])),
        "mean_cycles": {s: sides[s]["_mean_cycle"].tolist() for s in ("r", "l")},
    }


def _print_report(r: dict) -> None:
    seg = r.get("segmentation", "contact")
    note = "" if seg == "contact" else f", segmented by {seg}"
    print(f"\n=== {r['run']}  ({r['device']}, actpen={r['muscle_activation_penalty']}, mirror={r['mirror_coef']}{note})")
    print(
        f"  TOTAL {r['total']:.3f}   stability {r['stability']['score']:.3f}"
        f"   symmetry {r['symmetry']['score']:.3f}   plausibility {r['plausibility']['score']:.3f}"
    )
    s, y, p = r["stability"], r["symmetry"], r["plausibility"]
    print(
        f"  stability     strides {s['n_strides_r']}R/{s['n_strides_l']}L  dist {s['distance_m']:.2f} m"
        f"  speed {s['mean_speed_mps']:.2f} (err {s['speed_error_mps']:.2f})"
        f"  pelvis-h sd {s['pelvis_height_std_m'] * 100:.1f} cm"
        f"  stride CV {s['stride_time_cv'] * 100:.0f}%  stance {s['stance_frac_mean'] * 100:.0f}%"
    )
    print(
        f"  symmetry      peak R/L {y['peak_ratio']:.2f}  impulse R/L {y['impulse_ratio']:.2f}"
        f"  peak-phase gap {y['peak_phase_diff'] * 100:.0f}%  profile corr {y['profile_corr']:+.2f}"
    )
    if r.get("tendon_driven"):
        # Cable tension, not joint torque: the floor and band do not apply.
        print(
            f"  weaker leg    peak {p['weaker_leg_peak_nm_per_kg']:.2f} N/kg cable tension (tendon-driven; torque band not applicable)"
        )
    else:
        print(
            f"  weaker leg    peak {p['weaker_leg_peak_nm_per_kg']:.2f} Nm/kg"
            f"  ({'above' if p['weaker_leg_peak_nm_per_kg'] >= PEAK_BAND_NM_PER_KG[0] else 'BELOW'}"
            f" the {PEAK_BAND_NM_PER_KG[0]:.2f} useful-work floor)"
        )
    print(
        f"  plausibility  peak phase {p['peak_phase_r'] * 100:.0f}%R / {p['peak_phase_l'] * 100:.0f}%L"
        f"  peak {p['peak_nm_per_kg_r']:.2f}/{p['peak_nm_per_kg_l']:.2f} Nm/kg"
        f"  in-window {p['in_window_frac_r'] * 100:.0f}%/{p['in_window_frac_l'] * 100:.0f}%"
        f"  swing {p['swing_frac_r'] * 100:.0f}%/{p['swing_frac_l'] * 100:.0f}%"
    )
    a = r["abs_time"]
    if "one_side_silent" in a:
        print(f"  abs-time      one exo is silent (stride {a['stride_frames']:.0f} frames)")
    else:
        print(
            f"  abs-time      zero-lag corr {a['zero_lag_corr']:+.2f}"
            f"  best lag {a['best_lag_frames']:+d} frames = {a['best_lag_stride_frac']:+.2f} stride"
            f" (corr {a['best_lag_corr']:+.2f})   [symmetric target: |lag| = 0.50 stride]"
        )
    print(
        f"                peaks/cycle {p['n_peaks_r']}/{p['n_peaks_l']}"
        f"  cycle scatter {p['cycle_scatter_r']:.2f}/{p['cycle_scatter_l']:.2f}"
        f"  slew {p['slew_r']:.1f}/{p['slew_l']:.1f}"
        f"  |  muscle ctrl rms {r['muscle_ctrl_rms']:.4f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", type=pathlib.Path, help="Directories holding gait_evaluated_data_00.json.")
    ap.add_argument("--mass", type=float, default=DEFAULT_BODY_MASS_KG, help="Body mass for N*m/kg normalisation.")
    ap.add_argument("--joint", choices=sorted(ASSIST_WINDOW), default="ankle", help="Joint the device assists.")
    ap.add_argument("--json-out", type=pathlib.Path, default=None, help="Write all scores to this JSON file.")
    ap.add_argument(
        "--skip-unscorable",
        action="store_true",
        help="Report and continue past rollouts that cannot be scored instead of stopping. Off "
        "by default so a single run fails loudly. Turn it on to score a whole training "
        "trajectory, where the early checkpoints genuinely cannot walk and so contain too few "
        "foot strikes to segment -- that is a fact about the checkpoint, not a bad rollout.",
    )
    ap.add_argument(
        "--by-name",
        action="store_true",
        help="Order the report by directory name rather than by score, which is what a "
        "score-versus-steps trajectory wants (analyze_results_<steps>_00 sorts by step).",
    )
    args = ap.parse_args()

    results, skipped = [], []
    for run_dir in args.run_dirs:
        if args.skip_unscorable:
            try:
                results.append(score_run(run_dir, args.mass, args.joint))
            except AssertionError as exc:
                skipped.append((run_dir.name, str(exc)))
        else:
            results.append(score_run(run_dir, args.mass, args.joint))

    order = (lambda r: r["run"]) if args.by_name else (lambda r: -r["total"])
    for r in sorted(results, key=order):
        _print_report(r)

    for name, reason in skipped:
        print(f"\n=== {name}\n  NOT SCORED: {reason}")

    print("\n--- ranked ---")
    print(f"{'device':16} {'run':58} {'total':>6} {'stab':>6} {'symm':>6} {'plaus':>6}")
    for r in sorted(results, key=order):
        print(
            f"{r['device'] or '-':16} {r['run']:58} {r['total']:6.3f} {r['stability']['score']:6.3f}"
            f" {r['symmetry']['score']:6.3f} {r['plausibility']['score']:6.3f}"
        )

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
