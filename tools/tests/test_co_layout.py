"""CO parameter-layout smoke tests (the sim-free half of co-smoke).

These assert that the single source of truth (:class:`ParamLayout`) and the bounds
generator agree on the flat CMA-ES vector across every control mode and device
tail. The point mirrors the RL observation-layout assertion: configs and bounds
address the vector by absolute block offset, so a count drift silently repoints
the reflex / pose / tail blocks -- the old "D6" defect, where a length mismatch
became ``np.ones(N)`` instead of an error. Pure counts, milliseconds, no sim.
"""

from __future__ import annotations

import pytest

from ctrl_optim.ctrl.param_layout import ParamLayout
from ctrl_optim.optim.optim_utils.bounds import get_bounds


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (dict(control_mode="2D"), 77),
        (dict(control_mode="3D"), 97),
        (dict(control_mode="2D", bilateral=True), 128),
        (dict(control_mode="3D", bilateral=True), 160),
        (dict(control_mode="2D", bilateral=True, stiffness=2), 130),  # amp + K-Foot stiffness
        (dict(control_mode="2D", spline=4), 81),  # exo 4-param spline
        (dict(control_mode="3D", spline=12), 109),  # exo 6-point spline (2n)
    ],
)
def test_paramlayout_total(kwargs, expected):
    assert ParamLayout(**kwargs).total == expected


@pytest.mark.parametrize(
    "musc,mode,kw,expected",
    [
        ("22", "2D", dict(bilateral=False, exo_spline=0, stiffness=0), 77),
        ("26", "3D", dict(bilateral=False, exo_spline=0, stiffness=0), 97),
        ("80", "3D", dict(bilateral=False, exo_spline=0, stiffness=0), 97),
        ("22", "2D", dict(bilateral=True, exo_spline=0, stiffness=0), 128),
        ("26", "3D", dict(bilateral=True, exo_spline=0, stiffness=0), 160),
        ("22", "2D", dict(bilateral=True, exo_spline=0, stiffness=2), 130),  # amp + K-Foot stiffness
        ("22", "2D", dict(bilateral=False, exo_spline=4, stiffness=0), 81),  # exo 4-param
    ],
)
def test_get_bounds_length_matches_layout(musc, mode, kw, expected):
    """The bounds generator asserts len == layout.total internally; this also pins the numbers."""
    start, end = get_bounds(musc, mode, **kw)
    assert len(start) == len(end) == expected


def test_get_bounds_rejects_2d_only_model_in_3d():
    """The 22-muscle model has no hip ab/adductors, so 3D control must be refused."""
    with pytest.raises(ValueError):
        get_bounds("22", "3D", bilateral=False, exo_spline=0, stiffness=0)


def test_slices_partition_the_vector():
    """reflex | pose | spline | stiffness must tile [0, total) with no gap or overlap."""
    lay = ParamLayout("3D", bilateral=True, spline=4, stiffness=2)
    covered = []
    for s in (lay.slice_reflex(), lay.slice_pose(), lay.slice_spline(), lay.slice_stiffness()):
        covered.extend(range(s.start, s.stop))
    assert covered == list(range(lay.total))
