# RL pipeline (`rl_train/`) — Review 1 handoff

RL-specific findings needing the **RL author** + a train/eval run to validate (config /
net-architecture decisions; require the myosuite + SB3 stack). Full context in
`myoassist/review.md`. CO (`ctrl_optim/`) findings are in **`CO_PIPELINE_HANDOFF.md`**;
shared compose-pipeline items are deferred (see `REVIEW_1.md`).

Legend: **file:line** · issue · suggested fix · **test needed**.

## RL-1. `imitation.json` action layout doesn't match the composed model — **DONE (retired)**
- **`rl_train/train/train_configs/imitation.json:39,110-144`** (file deleted)
- Composed `myolegs22 + Tutorial_L1` → `nu = 24` (22 muscle + 2 `general` exo, `na = 22`), but
  `env_id = myoAssistLegImitation-v0` selects the **muscle-only** `HumanActorCriticPolicy` and
  the net-indexing declared **26** muscle activations (`observation range [17,43]`,
  `action range_action [0,26]`) — a count matching neither 22 nor 24, and no exo actor for the
  2 exo actuators. Nothing trained this config (no `.bat`/test), so it was never a live blocker.
- **Resolved as (a) retire.** Two findings made (b) "repoint it" pointless:
  - **The config could never have run, on any branch, at any point in this repo's history.** Its
    pre-migration `model_path` (on `dev`) was
    `models/22muscle_2D/gait14dof22musc_cvt3_Right_Toeless_2D.xml` — a path that was **never
    tracked in this repo** (`git log --all --` on it is empty; no `gait14dof*` file has any
    history, not even a deletion). So the stale `26` never matched a model here, and the
    "26-muscle net" framing in `REPO_ALIGNMENT.md`'s wave-7 flag was inexact — the referenced
    file was a 22-muscle 2D name that simply did not exist.
  - **No 26-muscle model has ever been trained successfully** (author, 2026-08-07); only the
    22-muscle line has. So the config encodes no working result worth preserving.
  Repairing it would have produced a near-duplicate of
  `imitation_tutorial_22_separated_net_exo_off.json` (the maintained 22-muscle + 2-exo template,
  `env_id = myoAssistLegImitationExo-v0`) and added a second file to maintain.
- **Deliberately NOT removed** — all of these are live, not dead:
  - `MyoAssistLegImitation` — parent class of `MyoAssistLegImitationExo`.
  - `ImitationTrainSessionConfig` — parent of `ExoImitationTrainSessionConfig`, and used directly
    by `run_policy_eval.py:74` and `train_analyzer.py:36`.
  - the `myoAssistLegImitation-v0` registration — `train_analyzer.py:34,100` branch on it.
  - `HumanActorCriticPolicy` — `environment_handler.py:168`'s `else` branch; muscle-only training
    is a legitimate capability.
  **Accepted side effect:** no config now exercises the muscle-only path
  (`myoAssistLegImitation-v0` + `HumanActorCriticPolicy`). The code stays reachable but untested
  by any shipped config. If a muscle-only baseline is wanted later, write a fresh config from
  `exo_off` minus the exo actor rather than reviving this one.
- **Also done:** `test_setup.py::test_config_files` existence check repointed off the deleted file
  onto the 5 maintained configs (was checking `imitation.json` alone).
- **Left for RL-2:** the class of bug is unfixed — nothing still checks a config's action layout
  against the composed model's actuator count.

## RL-2. Add an action-dim guard in `create_environment` — **DONE**
- **`rl_train/envs/environment_handler.py`** — `EnvironmentHandler._validate_action_layout`,
  called right after `EnvSpec(...).validate().compose()`.
- After composing, nothing checked the built model's actuator/muscle count against the config's
  action layout, so RL-1-style mismatches surfaced deep in a run as a tensor-shape error naming
  neither the config nor the keys that caused it.
- **Implemented as two checks**, both naming `env_id` + `msk_key` + `device_key` in the message:
  1. **Unaddressed motor actuators** — a device contributing motor actuators paired with an
     `env_id` outside `_EXO_ENV_IDS` (which selects the muscle-only `HumanActorCriticPolicy`).
  2. **Declared action width vs `nu`** — the **maximum** `range_action` end index across all nets
     must equal `model.nu`.
- **Two deviations from the suggested fix, both deliberate:**
  - Muscle count comes from `actuator_dyntype == mjDYN_MUSCLE`, not `nu - na`. A device's
    `general` actuator may declare its own activation dynamics, which would inflate `na` and make
    `nu - na` undercount the motors.
  - The declared width is the **max** `range_action` end, not the **sum** of the widths. Summing
    double-counts a `constant` override of a range a `range_mapping` already covers — `exo_off`
    does exactly this on `[22,24]`, so the sum is 26 against `nu = 24`. Max also mirrors what
    `NetworkIndexHandler.map_network_to_action` actually does (`result[:, start:end]` into an
    `nu`-wide tensor).
- **Test status:** the config-side arithmetic is verified against every shipped config — all 5
  give `declared_action_width = 24`, and the retired `imitation.json` (fetched from
  `origin/refactor`) trips **both** checks (`env_id = myoAssistLegImitation-v0` with 2 motor
  actuators; width 26 != 24). This is the test this item asked for, minus the MuJoCo half: it
  assumes the composed `myolegs22 + Tutorial_L1` gives `nu = 24` / 22 muscle rather than reading
  it from a compiled model. **Still needs a stacked run** to confirm the `mjDYN_MUSCLE` count and
  that the 5 configs build as before.

## RL-3. `create_environment` silently falls back to an absent `model_path` — **DONE**
- **`rl_train/envs/environment_handler.py`** — composed only when **both** `msk_key` and
  `device_key` were set; otherwise used `config.env_params.model_path` (`None` in every migrated
  config) → opaque `gym.make` failure that never named the missing key.
- **Fixed** with two raises around the compose branch, and the model-source decision is now an
  explicit table rather than an `if` with an implicit fall-through:

  | `msk_key` | `device_key` | `model_path` | outcome |
  |---|---|---|---|
  | set | set | any | compose (then RL-2's layout guard) |
  | set | — | any | **raise** — names `device_key` as missing |
  | — | set | any | **raise** — names `msk_key` as missing |
  | — | — | set | escape hatch: load the literal MJCF |
  | — | — | — | **raise** — nothing specified at all |

- Verified against the table above plus every shipped config (all 5 → compose) and a
  pre-migration `dev` config carrying a literal `model_path` and no keys (→ escape hatch, so
  old-style configs still load).
- **Intentional strictness:** half a spec raises *even when a usable `model_path` is present*.
  The config is ambiguous about which model was meant, and silently preferring one is what this
  item is about. No shipped config is in that state.
- Also dropped the defensive `getattr(env_params, "msk_key", None)` reads — `msk_key`,
  `device_key` and `terrain` are all declared on the base `EnvParams` (`config.py:65-67`), so a
  missing attribute would be a real defect worth surfacing, not something to paper over.

## Open questions — need a run with the RL stack
- ~~**`imitation.json` obs keys**~~ — **RESOLVED by inspection; moot after RL-1.** The config
  defined no `observation_joint_pos_keys/vel/sensor`, and `config.py:55-57` defaults all three to
  `field(default_factory=list)` — no defaults are injected anywhere. `get_obs_dict`
  (`myoassist_leg_base.py:187-216`) concatenates `qpos | qvel | act | sensor | target_velocity`, so
  the obs vector would have been `0 + 0 + 22 + 0 + 1 = 23` wide while the net-indexing sliced up to
  index 44. Third independent confirmation that the config was unusable well before this branch.
- **Keyframe-DOF extension (RL side):** `myoassist_leg_base.py:170` indexes `key_qpos[0]`, assuming
  assist_sim extends the human keyframes to cover the device DOFs when composing. Confirm the merged
  `key_qpos` width == `nq`. (CO side of the same assumption is in `CO_PIPELINE_HANDOFF.md`.)

## Related design gap (surfaced by RL-1, not fixed by it)
- **`compose_env_model` has no human-only path** — `device_key` is required and always attaches a
  device, so a muscle-only config is forced to carry uncontrolled exo actuators. No current config
  needs it (all 5 are exo configs, `env_id = myoAssistLegImitationExo-v0`), but it is why a clean
  muscle-only config cannot be written today. Belongs with the deferred compose-pipeline items in
  `REVIEW_1.md`, not with RL-2.
