"""PPO with a left/right mirror-symmetry penalty.

Follows the idea in Yu et al. 2018, *Learning Symmetric and Low-energy Locomotion*:
penalise the policy for disagreeing with its own mirror image,

    L_mirror = mean( ( pi(s) - M_a[ pi(M_s s) ] )^2 )

where ``M_s`` mirrors an observation left-to-right and ``M_a`` does the same to an action.
This constrains behaviour without touching the reward, which is what we want here: the
reward is a fixed part of the study, but the exo sub-policy settles on driving one leg in
one gait phase and the other leg in a different phase even though the model, the reference
and the resulting gait are all symmetric.

Deviation from the paper, deliberate: the penalty is applied as its own optimiser step
after the PPO update rather than as a term inside PPO's joint loss. SB3 builds that loss
in the middle of a 118-line ``PPO.train``, with no hook between the loss and
``backward()``, so folding the term in means vendoring the whole method and re-vendoring it
on every SB3 upgrade. The gradient direction is the same; what differs is that Adam sees
the mirror gradient in a separate step rather than summed with the policy gradient. If the
penalty turns out to matter and the difference is suspected to matter too, the vendored
version is the fallback.
"""

from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3 import PPO


class MirrorPPO(PPO):
    """PPO plus a mirror-symmetry penalty on the policy mean action.

    ``mirror_coef <= 0`` disables the penalty, making this exactly PPO.
    """

    def __init__(
        self, *args, mirror_coef: float = 0.0, obs_perm=None, act_perm=None, n_muscle_actuators: int | None = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mirror_coef = float(mirror_coef)
        if self.mirror_coef > 0:
            if obs_perm is None or act_perm is None:
                raise ValueError("mirror_coef > 0 requires both obs_perm and act_perm")
            self._obs_perm = th.as_tensor(np.asarray(obs_perm), dtype=th.long, device=self.device)
            self._act_perm = th.as_tensor(np.asarray(act_perm), dtype=th.long, device=self.device)
        self._n_muscle_actuators = n_muscle_actuators

    def _mean_action(self, obs: th.Tensor) -> th.Tensor:
        """The policy's mean action -- the mirror penalty constrains behaviour, not noise."""
        return self.policy.policy_network.forward_actor(obs)

    def _mirror_squared_error(self, obs: th.Tensor) -> th.Tensor:
        """Per-action-dimension mean squared mirror discrepancy, shape (n_actions,)."""
        mirrored_obs = obs[:, self._obs_perm]
        action = self._mean_action(obs)
        mirrored_action = self._mean_action(mirrored_obs)[:, self._act_perm]
        return ((action - mirrored_action) ** 2).mean(dim=0)

    def _mirror_loss(self, obs: th.Tensor) -> th.Tensor:
        return self._mirror_squared_error(obs).mean()

    def train(self) -> None:
        super().train()
        if self.mirror_coef <= 0:
            return

        losses, per_dim = [], []
        for rollout_data in self.rollout_buffer.get(self.batch_size):
            squared_error = self._mirror_squared_error(rollout_data.observations)
            loss = self.mirror_coef * squared_error.mean()
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            losses.append(loss.item())
            per_dim.append(squared_error.detach().cpu().numpy())
        if not losses:
            return
        self.logger.record("train/mirror_loss", float(np.mean(losses)) / self.mirror_coef)
        self.logger.record("train/mirror_loss_weighted", float(np.mean(losses)))

        # The penalty is a mean over every action dimension, and on these models 22 of 24 are
        # muscles, so the device dimensions can only ever receive a small share of the
        # gradient it produces. Logging the split makes that share visible instead of leaving
        # it to be inferred: if the device share stays tiny while the device asymmetry does
        # not improve, the penalty is being spent on muscles that were already near-symmetric.
        n_muscle = self._n_muscle_actuators
        if n_muscle is not None:
            dims = np.mean(per_dim, axis=0)
            muscle, device = dims[:n_muscle], dims[n_muscle:]
            self.logger.record("train/mirror_loss_muscle", float(muscle.mean()))
            if device.size:
                self.logger.record("train/mirror_loss_device", float(device.mean()))
                self.logger.record("train/mirror_device_grad_share", float(device.sum() / dims.sum()))
