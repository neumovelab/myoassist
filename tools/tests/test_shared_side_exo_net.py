"""Check that the weight-shared per-side exo network is exactly mirror-symmetric.

The mirror penalty asks the policy for symmetry and reward can outbid it. The per-side
network is supposed to make symmetry structural instead, and the difference only matters if
it actually holds to floating-point precision, which is what this asserts:

    Exo_L(s) == Exo_R(mirror(s))   and   Exo_R(s) == Exo_L(mirror(s))

for random observations, against an untrained network. Untrained is the point -- structural
symmetry must hold at initialisation and for every weight setting, so it cannot be reached by
training and cannot be lost by it either.

The single-`exo_actor` control in the same file is what makes the assertion meaningful: it
shows the property is a real consequence of the architecture rather than something the test
would pass either way.

Run:  .venv/bin/python -m tools.tests.test_shared_side_exo_net
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch as th
from gymnasium import spaces

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rl_train.train.policies.rl_agent_exo import CustomNetworkHumanExo  # noqa: E402
from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig  # noqa: E402
from rl_train.utils.data_types import DictionableDataclass  # noqa: E402

N_OBS, N_ACT = 44, 24
EXO_SLOTS = (22, 23)


def _observation_permutation(env_params: dict, muscle_names: list[str]) -> np.ndarray:
    """Mirror permutation over qpos | qvel | act | sensor | target_velocity."""
    from rl_train.train.policies.mirror import _permutation

    blocks = [
        env_params["observation_joint_pos_keys"],
        env_params["observation_joint_vel_keys"],
        muscle_names,
        env_params["observation_joint_sensor_keys"],
        ["target_velocity"],
    ]
    perm, offset = [], 0
    for names in blocks:
        perm.append(_permutation(list(names), "observation") + offset)
        offset += len(names)
    return np.concatenate(perm)


def _muscle_names() -> list[str]:
    """The 22 muscle actuators of myolegs22, right side first, as the env orders them."""
    base = [
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
    ]
    return [f"{m}_r" for m in base] + [f"{m}_l" for m in base]


def _build(config_path: pathlib.Path) -> CustomNetworkHumanExo:
    cfg = json.loads(config_path.read_text())
    params = DictionableDataclass.create(
        ExoImitationTrainSessionConfig.PolicyParams.CustomPolicyParams,
        cfg["policy_params"]["custom_policy_params"],
    )
    obs_space = spaces.Box(-np.inf, np.inf, (N_OBS,), dtype=np.float32)
    act_space = spaces.Box(-1.0, 1.0, (N_ACT,), dtype=np.float32)
    return CustomNetworkHumanExo(obs_space, act_space, params), cfg


def _exo_mirror_gap(net: CustomNetworkHumanExo, obs_perm: np.ndarray) -> float:
    """max |Exo_R(s) - Exo_L(mirror(s))| over a random batch, plus the symmetric partner."""
    th.manual_seed(0)
    obs = th.randn(512, N_OBS)
    with th.no_grad():
        action = net.forward_actor(obs)
        mirrored = net.forward_actor(obs[:, obs_perm])
    right, left = EXO_SLOTS
    return float(
        max(
            (action[:, right] - mirrored[:, left]).abs().max(),
            (action[:, left] - mirrored[:, right]).abs().max(),
        )
    )


def main() -> int:
    gen = REPO / "tools" / "make_device_sweep_configs.py"
    # A temporary directory, never the shipped one: this test generates two throwaway variants and
    # must not leave them in the set that is meant to be deployed.
    tmp = tempfile.mkdtemp(prefix="sidenet_test_")
    out_dir = pathlib.Path(tmp)
    common = [
        "--devices",
        "Tutorial_L1",
        "--human-net",
        "128",
        "--exo-net",
        "32",
        "--muscle-activation-penalty",
        "10",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run([sys.executable, str(gen), *common, "--exo-shared-side-net"], check=True, capture_output=True, cwd=REPO)
    subprocess.run([sys.executable, str(gen), *common, "--exo-contact"], check=True, capture_output=True, cwd=REPO)

    shared_net, cfg = _build(out_dir / "imitation_22_Tutorial_L1_h128_e32_sidenet_actpen10.json")
    single_net, _ = _build(out_dir / "imitation_22_Tutorial_L1_h128_e32_contact_actpen10.json")
    obs_perm = _observation_permutation(cfg["env_params"], _muscle_names())

    # Sanity: the permutation must be an involution over the full 44-vector.
    assert np.array_equal(obs_perm[obs_perm], np.arange(N_OBS)), "observation mirror map is not an involution"

    shared_gap = _exo_mirror_gap(shared_net, obs_perm)
    single_gap = _exo_mirror_gap(single_net, obs_perm)

    exo_params_shared = sum(p.numel() for p in shared_net.exo_policy_net_right.parameters())
    assert shared_net.exo_policy_net_right is shared_net.exo_policy_net_left, "per-side nets are not the same object"
    # Both names point at one module, so the optimizer must still see it exactly once.
    ids = [id(p) for p in shared_net.parameters()]
    assert len(ids) == len(set(ids)), "shared exo network is registered twice in parameters()"

    print(f"shared per-side exo net: exo mirror gap {shared_gap:.3e}  ({exo_params_shared} exo params)")
    print(f"single exo_actor        : exo mirror gap {single_gap:.3e}")

    ok = True
    if shared_gap > 1e-6:
        print(f"FAIL: shared per-side net is not mirror-symmetric (gap {shared_gap:.3e})")
        ok = False
    if single_gap < 1e-3:
        print(f"FAIL: control is already symmetric (gap {single_gap:.3e}), so the test proves nothing")
        ok = False
    shutil.rmtree(tmp, ignore_errors=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


def test_shared_side_exo_net_is_mirror_symmetric():
    """pytest entry point. `main` prints the gaps and returns a status, which the CLI form uses."""
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
