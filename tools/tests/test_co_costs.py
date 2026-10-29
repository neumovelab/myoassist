"""CO cost terms on synthetic data: kinematic and trunk tracking, and joint-limit reliance.

Fast checks of the cost definitions, with no simulation: every angle is constant, so each
expected value is known in closed form (100 gait-cycle samples, averaged over strides).
"""

from __future__ import annotations

import numpy as np
import pytest

from ctrl_optim.optim.cost_functions import effort_costs
from ctrl_optim.optim.cost_functions.kinematic_costs import calculate_kinematic_costs

N_STRIDES = 3
STRIDE = 50  # samples per stride
TRUNK = 0.1  # rad
JOINT = -0.3  # rad, every hip, knee, and ankle


@pytest.fixture
def stride_data():
    """Constant angles [timesteps, 7] (trunk, then 6 joints) and the heel-strike index_vector."""
    angles = np.tile([TRUNK] + [JOINT] * 6, ((N_STRIDES + 1) * STRIDE, 1))
    index_vector = np.arange(0, 2 * N_STRIDES + 1) * (STRIDE // 2)
    return angles, index_vector


def test_kinematic_cost_is_absolute_error(stride_data):
    angles, iv = stride_data
    kin, _ = calculate_kinematic_costs(angles, np.zeros((100, 7)), N_STRIDES, iv, "ref_diff")
    assert kin == pytest.approx(100 * 6 * abs(JOINT))  # squared error would give 100 * 6 * 0.09


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("ref_diff", 100 * TRUNK),
        ("zero_diff", 100 * TRUNK**2),
        ("tgt_diff", 100 * TRUNK**2),  # target 0 by default
        ("no_trunk", 0.0),
        (None, 100 * TRUNK),  # unset = ref_diff, the behavior before the option was connected
    ],
)
def test_trunk_err_types(stride_data, mode, expected):
    angles, iv = stride_data
    _, trunk = calculate_kinematic_costs(angles, np.zeros((100, 7)), N_STRIDES, iv, mode)
    assert trunk == pytest.approx(expected)


def test_trunk_vel_square(stride_data):
    angles, iv = stride_data
    vel = np.full(angles.shape[0], 0.5)
    _, trunk = calculate_kinematic_costs(angles, np.zeros((100, 7)), N_STRIDES, iv, "vel_square", trunk_vel=vel)
    assert trunk == pytest.approx(100 * 0.5**2)


def test_unknown_trunk_err_type_raises(stride_data):
    angles, iv = stride_data
    with pytest.raises(ValueError):
        calculate_kinematic_costs(angles, np.zeros((100, 7)), N_STRIDES, iv, "bogus")


def test_joint_limit_cost_layout():
    """Columns are hip, knee, ankle, each left then right, as evaluateCost fills them."""
    torques = np.zeros((11, 6))
    torques[:, 0:2] = 1.0  # hip L/R
    torques[:, 2:4] = -2.0  # knee L/R: the magnitude counts
    torques[:, 4:6] = 3.0  # ankle L/R
    costs = effort_costs.calculate_joint_limit_cost(torques, np.array([0, 10]))
    assert costs["hip"] == pytest.approx(2 * 1.0)
    assert costs["knee"] == pytest.approx(2 * 2.0)
    assert costs["ankle"] == pytest.approx(2 * 3.0)
    assert costs["total"] == pytest.approx(12.0)
