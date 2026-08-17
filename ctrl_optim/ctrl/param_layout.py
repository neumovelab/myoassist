"""Single source of truth for the CMA-ES parameter-vector layout.

The reflex controller (:class:`ctrl.reflex.reflex_ctrl.MyoLocoCtrl`) is a single
class driven by 13 muscle *groups*, so the parameter layout depends only on the
control mode (``"2D"`` / ``"3D"``) and the optional tails (bilateral doubling of
the reflex block, an exo spline block, a device-stiffness block).  It does NOT
depend on the muscle model: 22 / 26 / 80-muscle models all map their muscles onto
the same groups, and the muscle model only decides which real muscles exist (and
whether 3D is possible -- the 22-muscle model has no hip ab/adductors, so it is
2D only).

Everything that needs to know where the reflex / pose / spline / stiffness
blocks sit derives it from here, so the counts are defined once instead of being
re-derived in ``train.py``, ``reflex_interface.py`` and ``bounds.py`` (the old
D6 defect, where a mismatch silently became ``np.ones(N)``).

Block sizes per control mode (base = reflex + pose, before any device tail):

======  ======  ==============================  ==========
mode    reflex  pose (joints+vel / group act)   base total
======  ======  ==============================  ==========
2D      51      8 + 18 = 26                     77
3D      63      12 + 22 = 34                    97
======  ======  ==============================  ==========

The full vector is laid out as::

    [ reflex (x2 if bilateral) | pose | exo spline | device stiffness ]

The reflex block is per-leg; under bilateral control it is ``[r_leg | l_leg]``.
The pose block and the device tails are never doubled.
"""

from __future__ import annotations

from dataclasses import dataclass

# Length of MyoLocoCtrl.cp_keys actually consumed in each mode (the 2D keys are
# the first 51; the 3D frontal-plane keys extend that to 63).
REFLEX_N = {"2D": 51, "3D": 63}

# pose_key entries consumed per mode (includes vel_pelvis_tx at index 7); the 3D
# additions are hip_adduction_r/l and hip_rotation_r/l.
POSE_JOINT_N = {"2D": 8, "3D": 12}

# init_act_key entries consumed per mode; 3D adds the four HAB/HAD activations.
POSE_ACT_N = {"2D": 18, "3D": 22}

# Muscle models that map onto the group controller.  3D needs hip ab/adductors,
# which the 22-muscle model lacks.
VALID_MUSC_MODELS = {"22", "26", "80", "leg_11", "leg_80"}
MUSC_MODELS_2D_ONLY = {"22", "leg_11"}


def pose_n(control_mode: str) -> int:
    """Total pose-block length (joints/vel + group activations) for a mode."""
    return POSE_JOINT_N[control_mode] + POSE_ACT_N[control_mode]


@dataclass(frozen=True)
class ParamLayout:
    """Where each block sits in the flat CMA-ES parameter vector.

    Parameters
    ----------
    control_mode : str
        ``"2D"`` or ``"3D"``.
    bilateral : bool
        If True the reflex block is doubled as ``[r_leg | l_leg]``.
    spline : int
        Exo spline parameter count (0 when the exo is off, 4 for the legacy
        4-param spline, or ``2 * n_points`` for the n-point spline).
    stiffness : int
        Device-stiffness parameter count (0, or 2 for the pf/df ankle pair).
    """

    control_mode: str
    bilateral: bool = False
    spline: int = 0
    stiffness: int = 0

    def __post_init__(self) -> None:
        if self.control_mode not in REFLEX_N:
            raise ValueError(f"control_mode must be one of {sorted(REFLEX_N)}; got {self.control_mode!r}")

    # -- block sizes ----------------------------------------------------------
    @property
    def reflex_per_leg(self) -> int:
        return REFLEX_N[self.control_mode]

    @property
    def reflex(self) -> int:
        return self.reflex_per_leg * (2 if self.bilateral else 1)

    @property
    def pose(self) -> int:
        return pose_n(self.control_mode)

    @property
    def base(self) -> int:
        """Reflex + pose: the model-side core, before any device tail."""
        return self.reflex + self.pose

    @property
    def total(self) -> int:
        return self.base + self.spline + self.stiffness

    # -- slices into a full parameter vector ----------------------------------
    def slice_reflex(self) -> slice:
        return slice(0, self.reflex)

    def slice_reflex_leg(self, leg: str) -> slice:
        """Reflex sub-block for ``"r_leg"`` / ``"l_leg"``.

        For a non-bilateral layout both legs return the same single block (the
        controller mirrors it), matching the symmetric feed.
        """
        n = self.reflex_per_leg
        if not self.bilateral or leg == "r_leg":
            return slice(0, n)
        if leg == "l_leg":
            return slice(n, 2 * n)
        raise ValueError(f"leg must be 'r_leg' or 'l_leg'; got {leg!r}")

    def slice_pose(self) -> slice:
        return slice(self.reflex, self.reflex + self.pose)

    def slice_spline(self) -> slice:
        start = self.base
        return slice(start, start + self.spline)

    def slice_stiffness(self) -> slice:
        start = self.base + self.spline
        return slice(start, start + self.stiffness)
