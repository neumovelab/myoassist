"""CO cost terms on synthetic data: kinematic and trunk tracking, effort, EMG tracking,
joint-limit reliance, GRF limit, velocity, and the final-stage weights.

Fast checks of the cost definitions, with no simulation: every signal is constant, so each
expected value is known in closed form.
"""

from __future__ import annotations

import numpy as np
import pytest

from ctrl_optim.optim.cost_functions import effort_costs
from ctrl_optim.optim.cost_functions.evaluate_cost import (
    calculate_grf_cost,
    calculate_velocity_cost,
    resolve_cost_weights,
)
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


def test_kinematic_cost_is_mean_absolute_error(stride_data):
    angles, iv = stride_data
    kin, _ = calculate_kinematic_costs(angles, np.zeros((100, 7)), N_STRIDES, iv, "ref_diff")
    assert kin == pytest.approx(abs(JOINT))  # squared error would give 0.09


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("ref_diff", TRUNK),
        ("zero_diff", TRUNK**2),
        ("tgt_diff", TRUNK**2),  # target 0 by default
        ("no_trunk", 0.0),
        (None, TRUNK),  # unset = ref_diff, the behavior before the option was connected
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
    assert trunk == pytest.approx(0.5**2)


def test_unknown_trunk_err_type_raises(stride_data):
    angles, iv = stride_data
    with pytest.raises(ValueError):
        calculate_kinematic_costs(angles, np.zeros((100, 7)), N_STRIDES, iv, "bogus")


def test_effort_is_integral_per_muscle_and_distance():
    """Constant activation a over T seconds and D meters: a^2 T / D, whatever dt and muscle count."""
    a, T, D = 0.2, 4.0, 5.0
    for dt, n_muscles in ((0.02, 22), (0.01, 80), (0.001, 22)):
        n = int(round(T / dt))
        act = np.full((n + 10, n_muscles), a)
        cost = effort_costs.calculate_effort_cost(act, np.array([0, n]), D, dt)
        assert cost == pytest.approx(a**2 * T / D)


def test_effort_full_activation_at_1_m_s_is_one():
    act = np.ones((101, 10))
    assert effort_costs.calculate_effort_cost(act, np.array([0, 100]), 2.0, 0.02) == pytest.approx(1.0)


def test_emg_cost_is_mean_squared_error(stride_data):
    _, iv = stride_data
    muscles_dict = {"l_leg": {"SOL": [0], "TA": [1]}, "r_leg": {"SOL": [2], "TA": [3]}}
    act = np.full(((N_STRIDES + 1) * STRIDE, 4), 0.3)
    ref = np.zeros((100, 11))
    ref[:, 10] = 0.1  # TA reference; SOL reference stays 0
    cost = effort_costs.calculate_emg_profile_cost(act, ref, muscles_dict, "2D", iv)
    assert cost == pytest.approx((0.3**2 + 0.2**2) / 2)  # mean of the SOL and TA groups


def test_joint_limit_cost_layout():
    """Columns are hip, knee, ankle, each left then right, as evaluateCost fills them."""
    dt, D, mass, L = 0.02, 2.0, 80.0, 0.8
    torques = np.zeros((11, 6))
    torques[:, 0:2] = 1.0  # hip L/R
    torques[:, 2:4] = -2.0  # knee L/R: the magnitude counts
    torques[:, 4:6] = 3.0  # ankle L/R
    costs = effort_costs.calculate_joint_limit_cost(torques, np.array([0, 10]), dt, D, mass, L)
    scale = 10 * dt / (D * mass * effort_costs.GRAVITY * L)  # time integral of 1 N m, per distance and m g L
    assert costs["hip"] == pytest.approx(1.0 * scale)
    assert costs["knee"] == pytest.approx(2.0 * scale)
    assert costs["ankle"] == pytest.approx(3.0 * scale)
    assert costs["total"] == pytest.approx(2.0 * scale)  # mean over all joints


def test_grf_cost_integrates_excess():
    grf = np.full((51, 2), 0.8)  # 1.6 body weights in total
    grf[25:, :] = 0.5  # 1.0 body weights: no excess
    cost = calculate_grf_cost(grf, np.array([0, 50]), 1.5, 0.02, 2.0)
    assert cost == pytest.approx(25 * 0.02 * 0.1 / 2.0)


def test_velocity_cost_is_relative():
    pos = np.zeros((11, 3))
    pos[:, 0] = np.linspace(0, 1.1, 11)  # 1.1 m in 1 s
    t = np.zeros((11, 6))
    t[:, 2] = np.linspace(0, 1.0, 11)
    cost, dist, dur = calculate_velocity_cost(pos, t, np.array([0, 10]), 0.0, 1.0)
    assert cost == pytest.approx(0.1)
    cost2, _, _ = calculate_velocity_cost(pos, t, np.array([0, 10]), 0.0, 1.25)
    assert cost2 == pytest.approx(0.15 / 1.25)


def test_cost_weights_defaults_and_validation():
    w = resolve_cost_weights({"kinematic": 2})
    assert w["kinematic"] == 2.0 and w["effort"] == 1.0
    with pytest.raises(ValueError):
        resolve_cost_weights({"bogus": 1})
