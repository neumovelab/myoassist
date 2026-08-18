"""Left/right mirror maps for the sagittal leg model.

Mirror symmetry loss (Yu et al. 2018, *Learning Symmetric and Low-energy Locomotion*)
needs two permutations: one that swaps the left and right halves of an observation, and
one that does the same to an action. This model is planar, so both are pure index
permutations -- there is no abduction or rotation DOF whose sign would have to flip.

Both are derived from the config's own key lists and the compiled model's actuator names
rather than hardcoded, so a config that changes its observation keys cannot silently get a
stale map.
"""

from __future__ import annotations

import numpy as np

_OTHER_SIDE = {"l": "r", "r": "l", "L": "R", "R": "L"}
# Device models do not all use l/r. UTAnkleExo_L2 names its two actuators `..._dx` and `..._sx`
# (destra/sinistra), which the l/r rule maps to themselves -- and a self-map passes the
# involution check, so the mirror penalty silently stops constraining the one pair it exists to
# constrain. Any further convention has to be added here *and* caught by the no-partner check in
# EnvironmentHandler._mirror_permutations, which is what makes a missing convention loud.
_OTHER_SIDE_WORD = {"dx": "sx", "sx": "dx", "DX": "SX", "SX": "DX"}


def _swap_name(name: str) -> str:
    """`ankle_angle_l` -> `ankle_angle_r`; `Exo_R` -> `Exo_L`; `r_foot` -> `l_foot`.

    Case is preserved, which matters: the muscles are `_r`/`_l` but the device actuators
    are `Exo_R`/`Exo_L`, and matching only lowercase silently made the exo actuators map
    to themselves -- the one pair the mirror loss exists to constrain.

    Names with no side marker (`pelvis_tilt`, `target_velocity`) are their own mirror.
    """
    if len(name) > 3 and name[-3] == "_" and name[-2:] in _OTHER_SIDE_WORD:
        return name[:-2] + _OTHER_SIDE_WORD[name[-2:]]
    if len(name) > 2 and name[-2] == "_" and name[-1] in _OTHER_SIDE:
        return name[:-1] + _OTHER_SIDE[name[-1]]
    if len(name) > 2 and name[1] == "_" and name[0] in _OTHER_SIDE:
        return _OTHER_SIDE[name[0]] + name[1:]
    return name


def _permutation(names: list[str], what: str) -> np.ndarray:
    """Index array `p` with `mirrored[i] = original[p[i]]`."""
    index = {n: i for i, n in enumerate(names)}
    perm = np.empty(len(names), dtype=np.int64)
    for i, n in enumerate(names):
        partner = _swap_name(n)
        if partner not in index:
            raise KeyError(f"{what}: '{n}' mirrors to '{partner}', which is not present in {names}")
        perm[i] = index[partner]
    if not np.array_equal(perm[perm], np.arange(len(names))):
        raise ValueError(f"{what}: mirror map is not an involution")
    return perm


def observation_permutation(env_params, n_muscle_act: int, muscle_names: list[str]) -> np.ndarray:
    """Mirror permutation over `qpos | qvel | act | sensor | target_velocity`."""
    blocks = [
        list(env_params.observation_joint_pos_keys),
        list(env_params.observation_joint_vel_keys),
        list(muscle_names[:n_muscle_act]),
        list(env_params.observation_joint_sensor_keys),
        ["target_velocity"],
    ]
    perm, offset = [], 0
    for names in blocks:
        perm.append(_permutation(names, "observation") + offset)
        offset += len(names)
    return np.concatenate(perm)


def action_permutation(actuator_names: list[str]) -> np.ndarray:
    """Mirror permutation over the full actuator vector (muscles then device actuators)."""
    return _permutation(list(actuator_names), "action")
