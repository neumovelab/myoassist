"""Gait strides come from the foot sensors when the sensors and the hip-flexion fallback agree.

`GaitAnalyzer.get_gait_segment_index` keeps the sensor strides only when their count is within one
of the hip-flexion strides. The sensor path drops its first (start-up) stride and the fallback used
not to, so the counts differed by two on a normal walk: the report threw away valid heel strikes,
cut every stride about 9 % of the cycle early, and put toe-off at the fallback's fixed 60 %.
"""

from __future__ import annotations

import numpy as np

from rl_train.analyzer.gait_analyze import GaitAnalyzer
from rl_train.analyzer.gait_data import GaitData

STRIDE = 50  # samples
TOE_OFF = 32  # samples after heel strike (64 %), so a measured toe-off differs from the fallback's 60 %
STRIKES = range(50, 501, STRIDE)  # the last stance runs past the end, so it has no toe-off
LENGTH = 520


def _walk() -> GaitData:
    """Heel and toe loads with a strike every STRIDE samples, and a hip-flexion peak 6 samples before each."""
    heel, toes = np.zeros(LENGTH), np.zeros(LENGTH)
    for s in STRIKES:
        heel[s : s + 20] = 500.0
        toes[s + 5 : s + TOE_OFF] = 500.0
    t = np.arange(LENGTH)
    hip = 0.5 * np.cos(2 * np.pi * (t - (STRIKES[0] - 6)) / STRIDE)
    gait = GaitData()
    gait.series_data["sensor_data"]["r_foot"] = {"data": [[v] for v in heel]}
    gait.series_data["sensor_data"]["r_toes"] = {"data": [[v] for v in toes]}
    gait.series_data["joint_data"]["hip_flexion_r"] = {"qpos": [[v] for v in hip]}
    return gait


def test_sensor_strides_are_kept_when_they_agree_with_hip_flexion():
    analyzer = GaitAnalyzer(_walk(), None, show_plot=False)
    segments = analyzer.get_gait_segment_index(is_right_foot_based=True)
    # Strike-to-toe-off pairs exist for STRIKES[:-1]; the sensor path makes strides of consecutive
    # pairs and drops the first, start-up stride.
    expected = [(s, s + TOE_OFF, s + STRIDE) for s in list(STRIKES)[1:-2]]
    assert segments == expected, f"got the hip-flexion strides {segments[:2]}..., expected the sensor strides {expected[:2]}..."
