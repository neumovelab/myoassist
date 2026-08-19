# Author(s): Chun Kwang Tan <cktan.neumove@gmail.com>, Calder Robbins <robbins.cal@northeastern.edu>
"""Parameter bounds for optimization.

Bounds are *generated* from the live reflex controller's key lists rather than
hand-written per muscle model.  There is one controller
(:class:`~ctrl_optim.ctrl.reflex.reflex_ctrl.MyoLocoCtrl`), driven by 13 muscle
groups, so the bound vector depends only on the control mode (``"2D"`` / ``"3D"``)
and the optional device tails -- not on the muscle model (22 / 26 / 80 all map
onto the same groups).  :class:`~optim_utils.param_layout.ParamLayout` owns the
block sizes; this module owns the per-key sign rule.

This replaces three hand-written tables that had drifted out of sync with the
controller (the 22/26 2D list was missing ``1_FDL_FG`` and mislabelled module 5,
so its last two reflex gains landed on pose bounds and could go negative; the
22/26 3D branch raised ``NameError``; and the 80-muscle list was length 79 vs the
controller's 77, so ``musc_model=80`` failed the ``reset()`` length check).
"""

from typing import List, Optional

import numpy as np

from ctrl_optim.ctrl.param_layout import (
    MUSC_MODELS_2D_ONLY,
    POSE_ACT_N,
    POSE_JOINT_N,
    VALID_MUSC_MODELS,
    ParamLayout,
)

# Global set by train.py (``bounds_mod.input_args = input_args``); supplies the
# exo / bilateral / stiffness configuration used to size the device tails.
input_args = None

# --- per-parameter bound primitives -----------------------------------------
_FREE = [-np.inf, np.inf]
_NONNEG = [0.0, np.inf]
_ACT_BOUND = [1.0, 50.0]  # initial-activation scale
_VEL_BOUND = [-29.0, 6.0]  # vel_pelvis_tx
_DEVICE_BOUND = [0.0, 1.0]  # normalized exo-spline / stiffness params

# The only reflex keys allowed to go negative; every other reflex gain/threshold
# is constrained non-negative.
_REFLEX_FREE_KEYS = {"knee_tgt", "knee_sw_tgt", "knee_off_st", "ankle_tgt", "mtp_tgt"}


def _reflex_bound(key: str) -> List[float]:
    return list(_FREE) if key in _REFLEX_FREE_KEYS else list(_NONNEG)


def _pose_joint_bound(key: str) -> List[float]:
    # Joint-angle pose params are free; vel_pelvis_tx is the one bounded entry.
    return list(_VEL_BOUND) if key == "vel_pelvis_tx" else list(_FREE)


def _spline_count() -> int:
    """Exo spline parameter count implied by the module-global ``input_args``."""
    if input_args is None or not getattr(input_args, "ExoOn", 0):
        return 0
    if getattr(input_args, "use_4param_spline", False):
        return 4
    return int(input_args.n_points) * 2


def _build_bound_vect(layout: ParamLayout) -> List[List[float]]:
    """Assemble the ordered [reflex (x legs) | pose | spline | stiffness] bounds."""
    # Lazy import so this module stays importable without the (heavy) env stack,
    # and so it always reads the controller's *current* key lists (single source
    # of truth -- no duplicated tables to drift).
    from ctrl_optim.ctrl.reflex.reflex_ctrl import MyoLocoCtrl
    from ctrl_optim.ctrl.reflex.reflex_interface import myoLeg_reflex

    mode = layout.control_mode
    n_reflex = layout.reflex_per_leg
    n_pose_joint = POSE_JOINT_N[mode]
    n_act = POSE_ACT_N[mode]

    if len(MyoLocoCtrl.cp_keys) < n_reflex:
        raise ValueError(f"MyoLocoCtrl.cp_keys has {len(MyoLocoCtrl.cp_keys)} entries, need >= {n_reflex} for {mode}")
    if len(myoLeg_reflex.pose_key) < n_pose_joint:
        raise ValueError(f"pose_key has {len(myoLeg_reflex.pose_key)} entries, need >= {n_pose_joint} for {mode}")
    if len(myoLeg_reflex.init_act_key) < n_act:
        raise ValueError(f"init_act_key has {len(myoLeg_reflex.init_act_key)} entries, need >= {n_act} for {mode}")

    reflex_keys = MyoLocoCtrl.cp_keys[:n_reflex]
    pose_joint_keys = myoLeg_reflex.pose_key[:n_pose_joint]

    bound_vect: List[List[float]] = []
    for _ in range(2 if layout.bilateral else 1):
        bound_vect.extend(_reflex_bound(k) for k in reflex_keys)
    bound_vect.extend(_pose_joint_bound(k) for k in pose_joint_keys)
    bound_vect.extend(list(_ACT_BOUND) for _ in range(n_act))
    bound_vect.extend(list(_DEVICE_BOUND) for _ in range(layout.spline + layout.stiffness))

    if len(bound_vect) != layout.total:
        raise AssertionError(f"generated {len(bound_vect)} bounds but layout.total is {layout.total}")
    return bound_vect


def get_bounds(
    musc_model: str,
    control_mode: str,
    *,
    exo_spline: Optional[int] = None,
    bilateral: Optional[bool] = None,
    stiffness: Optional[int] = None,
) -> List[List[float]]:
    """Return ``[bound_start, bound_end]`` for the given model and control mode.

    The device tails (exo spline, bilateral doubling, stiffness) are taken from
    the module-global ``input_args`` unless overridden explicitly (the overrides
    exist mainly for tests).
    """
    if musc_model not in VALID_MUSC_MODELS:
        raise ValueError(f"No bounds defined for muscle model: {musc_model!r}")
    if control_mode == "3D" and musc_model in MUSC_MODELS_2D_ONLY:
        raise ValueError(f"3D control needs hip ab/adductors; muscle model {musc_model!r} is 2D-only")

    spline = _spline_count() if exo_spline is None else exo_spline
    bil = (getattr(input_args, "reflex_mode", None) in ("bilat", "amp")) if bilateral is None else bilateral
    stiff = (2 if getattr(input_args, "optimize_stiffness", False) else 0) if stiffness is None else stiffness

    layout = ParamLayout(control_mode, bilateral=bil, spline=spline, stiffness=stiff)
    bound_vect = _build_bound_vect(layout)
    bound_start = [b[0] for b in bound_vect]
    bound_end = [b[1] for b in bound_vect]
    return [bound_start, bound_end]
