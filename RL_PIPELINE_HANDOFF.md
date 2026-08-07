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
  2. **Claimed actuator indices vs `[0, nu)`** — the union of every `range_action` must claim no
     index at or beyond `nu`, and must leave no actuator unclaimed.
- **Two deviations from the suggested fix, both deliberate:**
  - Muscle count comes from `actuator_dyntype == mjDYN_MUSCLE`, not `nu - na`. A device's
    `general` actuator may declare its own activation dynamics, which would inflate `na` and make
    `nu - na` undercount the motors. (On `myolegs22 + Tutorial_L1` the two agree — both give 2
    motors — so this is defensive rather than currently load-bearing: `Exo_R`/`Exo_L` are
    `dyntype=0`.)
  - Check 2 uses the **covered index set**, not the summed width and not just its maximum.
    Summing double-counts a `constant` override of a range a `range_mapping` already covers —
    `exo_off` does exactly this on `[22,24]`, so the sum is 26 against `nu = 24`. Taking only the
    maximum was the first implementation and **it let an interior gap through**: with
    `human_actor` claiming `[0,11]+[11,20]` and `exo_actor` `[22,24]`, the widest index still
    lands on 24 while actuators 20 and 21 are driven by nothing and sit at zero. The covered set
    catches that. Overlap stays legal, since the `constant` override depends on it.
- **Test status — verified against the real stack** (see the run record at the end of this file):
  - `n_muscle` from `mjDYN_MUSCLE` is **22**, `n_motor` **2** (`Exo_R`, `Exo_L`) on the composed
    `myolegs22 + Tutorial_L1` (`nq=39, nu=24, na=22`).
  - All 5 shipped configs build and step: `obs=(44,)`, `act=(24,)`.
  - Every failure mode raises with a clear message: motor actuators under a muscle-only `env_id`;
    an index claimed past `nu` (`exo [22,26]`); an unclaimed trailing actuator (`exo [22,23]`);
    and the interior gap above.
  - A 64-timestep PPO run on `..._partial_obs.json` completes (21 updates, exit 0), which is the
    "short train step without a shape error" this item asked for.

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
- ~~**Keyframe-DOF extension (RL side)**~~ — **RESOLVED by a stacked run; the assumption holds,
  but a different one does not.** Widths are extended correctly: composed `myolegs22 + Tutorial_L1`
  and `+ DephyExoBoot_L1` both give `key_qpos (5, 39)` / `key_qvel (5, 39)` against `nq=39, nv=39`.
  What is *not* safe is the `[0]`: composed **`myolegs26 + Humotech_L1` has `nkey=0`**
  (`key_qpos (0, 47)`), so `key_qpos[0]` raises `IndexError` from deep inside `_setup`. Keyframes
  come from the MSK model -- myolegs22 ships 5, myolegs26 ships none. Added an `assert model.nkey > 0`
  naming that, since the raw IndexError says nothing about which MSK is at fault. Not live today
  (no config uses myolegs26) and **one more reason the 26-muscle line has never trained**.

## Related design gap (surfaced by RL-1, not fixed by it)
- **`compose_env_model` has no human-only path** — `device_key` is required and always attaches a
  device, so a muscle-only config is forced to carry uncontrolled exo actuators. No current config
  needs it (all 5 are exo configs, `env_id = myoAssistLegImitationExo-v0`), but it is why a clean
  muscle-only config cannot be written today. Belongs with the deferred compose-pipeline items in
  `REVIEW_1.md`, not with RL-2.

---

## Verification run record (2026-08-07, macOS arm64, Python 3.12.13)

The RL pass was verified on a Mac, which required working around three dependency defects that
are **not** RL-pass items -- they block anyone installing from `requirements.txt`. Recorded in
`REPO_ALIGNMENT.md` under A7; summarised here because they gate reproducing this run.

Stack that makes `test_setup.py` pass **15/15**:

| package | version | why not the obvious choice |
|---|---|---|
| `myosuite` | **2.11.6** | 2.12.0 replaced `physics/mj_sim_scene.py`'s path/string dispatch with `env_base._get_spec` calling `MjSpec.from_file` unconditionally, so the XML *string* `compose_env_model` returns is treated as a filename -> `ValueError: could not decode content`. `requirements.txt` leaves `myosuite` unpinned, so a fresh install gets 2.12.x and every composed env fails. |
| `mujoco` | **3.3.4** | myosuite 2.11.6 pins `mujoco==3.3.0`, which the compose path rejects: myo_sim's 26->22 reduction needs `MjSpec.delete`, absent before 3.3.4. The pin has to be deliberately overridden. |
| `dm-control` | **1.0.31** | 1.0.28 (what myosuite 2.11.6 resolves to) reads `MjModel.light_directional`, removed in mujoco 3.3.4 -> `AttributeError` at env init. |
| `myo_sim` | `MyoHub/myo_sim@dev` + shim | Ships `build_spec` at `myo_sim.build.compose.build_spec` but its `__init__.__getattr__` only re-exports `_COMPOSED_MODELS`, so `myo_sim.build_spec(...)` -- what `assist_sim.registry._resolve_msk` calls -- raises `AttributeError`. Bridged locally with `myo_sim.build_spec = myo_sim.build.compose.build_spec`; the real fix is REPO_ALIGNMENT C4 (public accessor), currently **deferred**. |
| `myoassist_terrains` | local clone @ `origin/main` | The git URL in `requirements.txt` is not publicly reachable. |

The XML-string dispatch is **upstream behaviour, not a myoassist local modification** -- the
vendored 2.8.3 copy on `dev` and upstream 2.11.6 are near-identical there (2.11.6 in fact *adds*
an `MjModel` branch). So this is not fallout from A2's un-audited fork removal.

Also found: `assist_sim`'s `_COMPATIBLE_MSK_KEYS` gives `myolegs22`/`myolegs26`
`min_mujoco=(3,3,3)` with an empty `note`, but the 26->22 reduction uses `MjSpec.delete`, which
needs 3.3.4 -- so the declared floor is understated by one patch and the resulting error reads
`requires ; installed mujoco is 3.3.0`.
