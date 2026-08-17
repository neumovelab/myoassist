"""Passive prosthetic-ankle stiffness: denormalize + write into the live model.

A powered/passive prosthetic foot (e.g. ``KFoot``) models its ankle as two
one-directional spring joints on the same axis: ``pf_ankle_angle_r``
(plantarflexion, negative range) and ``df_ankle_angle_r`` (dorsiflexion, positive
range), each carrying a ``stiffness``.  This module turns two normalized CMA-ES
parameters ``[p_pf, p_df]`` in ``[0, 1]`` into absolute stiffnesses (Nm/rad) and
writes them straight into ``model.jnt_stiffness`` at ``reset`` -- no temporary
XML, no recompile.

The joints are resolved by name *suffix*, because assist_sim prefixes device
elements (the composed names are ``<Device>_pf_ankle_angle_r`` etc.).
"""

import numpy as np

# Absolute denormalization ranges (Nm/rad).  These are the ME5374 / ANAT working
# ranges; replace with published quasi-stiffness data for the specific foot and
# record the citation here (the normalized value is meaningless without them).
PF_RANGE = (30.0, 305.0)
DF_RANGE = (100.0, 1050.0)

N_STIFFNESS_PARAMS = 2
PF_JOINT_SUFFIX = "pf_ankle_angle_r"
DF_JOINT_SUFFIX = "df_ankle_angle_r"


def denormalize(p: float, lower: float, upper: float) -> float:
    """Map a normalized ``p`` in ``[0, 1]`` (clipped) to ``[lower, upper]``."""
    return lower + float(np.clip(p, 0.0, 1.0)) * (upper - lower)


def _resolve_joint(model, suffix: str) -> str:
    """Return the joint whose name is ``suffix`` or ends with ``_<suffix>`` (the
    assist_sim device-prefixed form).  Raises ``KeyError`` if none/ambiguous."""
    matches = [model.joint(i).name for i in range(model.njnt) if model.joint(i).name.endswith(suffix)]
    exact = [n for n in matches if n == suffix or n.endswith("_" + suffix)]
    if not exact:
        raise KeyError(f"no joint ending with {suffix!r}; is this a prosthetic (df/pf ankle) model?")
    return exact[0]


def apply_stiffness(sim, p_pf: float, p_df: float, pf_range=PF_RANGE, df_range=DF_RANGE):
    """Write the denormalized pf/df stiffness into ``sim.model.jnt_stiffness``.

    Returns ``(K_pf, K_df)`` in Nm/rad -- the absolute values applied, so the
    caller can log them (the normalized parameters alone are not self-describing).
    """
    model = sim.model
    pf_name = _resolve_joint(model, PF_JOINT_SUFFIX)
    df_name = _resolve_joint(model, DF_JOINT_SUFFIX)
    k_pf = denormalize(p_pf, *pf_range)
    k_df = denormalize(p_df, *df_range)
    model.jnt_stiffness[model.joint(pf_name).id] = k_pf
    model.jnt_stiffness[model.joint(df_name).id] = k_df
    return k_pf, k_df


def describe_ranges() -> dict:
    """Ranges + joint suffixes, for serializing next to the results (self-describing)."""
    return {
        "pf_joint": PF_JOINT_SUFFIX,
        "df_joint": DF_JOINT_SUFFIX,
        "pf_range_nm_per_rad": list(PF_RANGE),
        "df_range_nm_per_rad": list(DF_RANGE),
    }
