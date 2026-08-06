# RL pipeline (`rl_train/`) — Review 1 handoff

RL-specific findings needing the **RL author** + a train/eval run to validate (config /
net-architecture decisions; require the myosuite + SB3 stack). Full context in
`myoassist/review.md`. CO (`ctrl_optim/`) findings are in **`CO_PIPELINE_HANDOFF.md`**;
shared compose-pipeline items are deferred (see `REVIEW_1.md`).

Legend: **file:line** · issue · suggested fix · **test needed**.

## RL-1. `imitation.json` action layout doesn't match the composed model
- **`rl_train/train/train_configs/imitation.json:39,110-144`**
- Composes `myolegs22 + Tutorial_L1` → `nu = 24` (22 muscle + 2 `general` exo, `na = 22`), but
  `env_id = myoAssistLegImitation-v0` selects the **muscle-only** `HumanActorCriticPolicy` and
  the net-indexing declares **26** muscle activations (`observation range [17,43]`,
  `action range_action [0,26]`) — a count matching neither 22 nor 24, and no exo actor for the
  2 exo actuators. Nothing trains this config today (no `.bat`/test), so not a live blocker.
- **Suggested fix (author's call):** either (a) retire `imitation.json` (the
  `imitation_tutorial_22_separated_net_*` configs supersede it — `exo_off` is the maintained
  22-muscle + 2-exo template, `env_id = myoAssistLegImitationExo-v0`), or (b) repoint it to
  `myoAssistLegImitationExo-v0` with the 22-muscle (`[0,11]`+`[11,22]`) + 2-exo (`[22,24]`)
  layout mirroring `exo_off`. If (a): repoint the `test_setup.py:336` existence check to a
  maintained config.
- **Test needed:** build the env, confirm SB3 action space width == model `nu` and the
  net-indexing slices are in range; run a short train/eval step without a shape error.

## RL-2. Add an action-dim guard in `create_environment`
- **`rl_train/envs/environment_handler.py:10-57`**
- After composing, nothing checks the built model's actuator/muscle count against the config's
  action layout, so RL-1-style mismatches surface deep in a run.
- **Suggested fix:** after `compose_env_model(...)`, compile the returned XML and validate — e.g.
  `n_motor = model.nu - model.na`; if a muscle-only policy (`env_id` not `…Exo-v0`) and
  `n_motor > 0`, raise a clear error naming env + `msk_key/device_key`; also flag when the config's
  action-net width != `model.nu`.
- **Test needed:** guard raises clearly on `imitation.json`, passes the `separated_net_*` configs.

## RL-3. `create_environment` silently falls back to an absent `model_path`
- **`rl_train/envs/environment_handler.py:20-27`** — composes only when **both** `msk_key` and
  `device_key` are set; otherwise uses `config.env_params.model_path` (`None` in every migrated
  config) → opaque `gym.make` failure. **Fix:** raise a clear error when exactly one key is set.

## Open questions — need a run with the RL stack
- **`imitation.json` obs keys:** defines no `observation_joint_pos_keys/vel/sensor` (unlike the
  `separated_net` configs), so the base `EnvParams` empty-list defaults may yield empty qpos/qvel
  obs — inconsistent with the net's `[0,8]`/`[8,17]` slices. Confirm whether defaults are injected;
  if not, `imitation.json` was already unusable before this branch (strengthens RL-1's "retire it").
- **Keyframe-DOF extension (RL side):** `myoassist_leg_base.py:170` indexes `key_qpos[0]`, assuming
  assist_sim extends the human keyframes to cover the device DOFs when composing. Confirm the merged
  `key_qpos` width == `nq`. (CO side of the same assumption is in `CO_PIPELINE_HANDOFF.md`.)
