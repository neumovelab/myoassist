# Code review — §A "MyoAssist 1.0" refactor (`git diff main...refactor`, 27 commits)

## Summary & scope
This branch slims MyoAssist to a **composed** architecture: it deletes the bundled
`models/` tree, the vendored `myosuite/`, and the Jekyll docs site (the bulk of the
~622k deleted lines), and adds a single shared model-build path,
`myoassist_utils/compose.py::compose_env_model(msk_key, device_key, terrain=None,
export_path=None)`. That util composes a human MSK + assistive device (via
`assist_sim.load_combined`), merges a `myoassist_terrains` scene (flat default when
`terrain=None`), seats the model on the surface using real collision geometry, and
returns a loadable MJCF **string**. Both frameworks are rewired onto it: RL through
`EnvironmentHandler.create_environment` (new `EnvParams.msk_key/device_key/terrain`,
retiring `terrain_type`/`HfieldManager`), and CO through a new reach-free
`ReflexEnvV0` built in `reflex_interface`, plus threaded compose keys through the CO
optim/eval entry points. Packaging drops the vendored trees and pulls the three
sibling repos + `myosuite` as deps. I read `compose.py`, the RL/CO wiring, all six RL
train configs, the CO optim entry points, `setup.py`/`requirements.txt`/`test_setup.py`,
and cross-checked API contracts against the adjacent `assist_sim` and `myoassist.terrains`
clones. I could not execute anything (none of `mujoco`/`assist_sim`/`myoassist_terrains`/
`myosuite` are importable in this interpreter), so runtime-behavior claims are marked
where they rest on reading rather than a run. **Overall the compose util and the tested
wiring paths look correct**; the findings below are one real config bug plus dead-code /
leak / coupling cleanups.

The core build math checks out on inspection: `assist_sim.export_combined_xml` writes
mesh `file=` paths relative to the export dir, so compose's `_absolutize_files(model_root,
model_xml_path.parent)` resolves them to the persistent install location (correct, and it
lets the temp dir be deleted); the flat-default `height=0.0` tile is valid
(`BASELINE_Z=-2.0` in the terrains pkg, so `top_z=0 > BASELINE_Z`) and emits a named geom
so seating can find it; the seating formula `dz = -penetration - min(gaps)` is measured at
the "stand" keyframe pose and baked into the root body's `pos`, which is consistent with
the env's reset-to-keyframe behavior. Device keys in the CO map resolve
(`registry._device_key` keys off `models/<dir>/*config.yaml`, so `DephyExoBoot_L1` etc.
are valid despite the differing yaml `name:` field). "Always merge a flat default ground"
is satisfied (assist_sim strips the scene; compose always adds either the flat default or
a terrain).

## Findings

### Blockers
None confirmed. (The tested/default paths — `imitation_tutorial_22_separated_net_*`,
`test_setup.py`'s compose + RL init with `DephyExoBoot_L1` — are internally consistent.)

### Major

**1. `imitation.json` composes a 24-actuator model but declares a 26-muscle, no-exo
action layout; nothing guards the mismatch.**
`rl_train/train/train_configs/imitation.json:39` sets `device_key: "Tutorial_L1"`, and
Tutorial_L1 adds two `general` actuators (`Exo_R`, `Exo_L`; verified in
`assist_sim/models/Tutorial/L1config.yaml:23-47`). Composed with `myolegs22` (22 muscles),
the model has `nu = 24` (22 muscle + 2 exo), `na = 22`. But this config uses env
`myoAssistLegImitation-v0` → `HumanActorCriticPolicy` (muscle-only, no exo actor;
`environment_handler.py:151-155`) and its net indexing declares "26 muscle activation"
with action mapping `range_action:[0,26]` (`imitation.json:126-144`) — no exo handling and
a muscle count that matches neither 22 nor 24.
*Why it matters:* training/eval on this config fails on a shape mismatch (SB3 action space
`nu=24` vs a 26-wide action net, and the custom net indexing slices a 22-long `act` obs at
`[17,43]`). `EnvironmentHandler.create_environment` composes and builds with no assertion
that policy action-dim == `nu` or that the config's muscle count matches the model, so the
error surfaces deep in a run, not at config load. Root cause is a design gap:
`compose_env_model` has no human-only path (`device_key` is required and always attaches a
device), so a muscle-only config is *forced* to carry 2 uncontrolled exo actuators.
*Note:* the "26 vs 22" half was already stale on `main` (old `model_path` was a 22-muscle
model), but the migration made it strictly worse by adding the exo actuators, and this is a
shipped config for a registered env (referenced as the representative RL config in
`test_setup.py:336`). Not a blocker only because no `.bat`/test actually trains it.
*Fix:* either (a) drop `imitation.json` / repoint it at `myoAssistLegImitationExo-v0` with
a 22+2 layout like the `separated_net` configs, or (b) add a human-only compose path
(allow `device_key=None`) and correct the muscle count to 22; and add a cheap guard in
`create_environment` that checks the built model's `nu`/`na` against the config's action
layout and raises a clear error.

**STATUS 2026-08-07 (RL pass) — config half DONE via (a) drop; two parts still open.**
`imitation.json` is deleted and `test_setup.py`'s existence check repointed onto the 5
maintained configs. The "already stale on `main`" note above understated it: the old
`model_path` (`models/22muscle_2D/gait14dof22musc_cvt3_Right_Toeless_2D.xml`) was **never
tracked in this repo at all**, so the config could not have run at any point in its history;
and no 26-muscle model has ever been trained successfully (author). Nothing to preserve.
Still open, tracked in `RL_PIPELINE_HANDOFF.md`:
- **RL-2** — the guard. The root cause ("nothing asserts policy action-dim == `nu`") is
  untouched by deleting one config.
- **The design gap** flagged above — `compose_env_model` has no human-only path (`device_key`
  is required, so a muscle-only config is forced to carry uncontrolled exo actuators). Not
  needed by any current config (all 5 are exo configs), but it is the reason a muscle-only
  config cannot be written cleanly today. Belongs with the deferred compose-pipeline items.

### Minor

**2. `resolve_model_path` was not actually removed despite commit `5aa5553` ("…drop
resolve_model_path"); it is now dead code pointing at the deleted `models/` tree.**
`ctrl_optim/optim/optim_utils/resolve_path.py:26` still defines it and builds
`os.path.join(project_root, 'models', model_dir, …)` (`:78`) against a directory this
branch deleted, and it is still re-exported from
`ctrl_optim/optim/optim_utils/__init__.py:17,32`. Nothing calls it anymore (the only
former caller, `config_parser.create_testenv_from_bat`, was rewired to compose).
*Why it matters:* misleading commit history, and a broken function left importable — any
future caller gets a `FileNotFoundError`. *Fix:* delete `resolve_model_path` (and the
now-unused `get_available_models`/`validate_model_config` if they were only serving it)
and drop the `__init__` exports, or trim `resolve_path.py` to the still-used helpers
(`resolve_reference_data_path`, `resolve_results_path`).

**3. `myoLeg_reflex.setupTerrain` is now unreachable dead code that manipulates a
heightfield the composed model no longer has.**
`ctrl_optim/ctrl/reflex/reflex_interface.py:270` still defines `setupTerrain`, but its only
call site was removed and `slope_deg != 0` now raises `NotImplementedError` earlier
(`:171-188`), so it can never run. Its body writes `self.env.sim.model.hfield_data` /
`hfield_size` and looks up `geom 'terrain'` — none of which exist in a composed model
(compose emits a box geom named `flat_r0c0_box`, no hfield). *Why it matters:* a loaded
gun for D5; if re-wired it will `KeyError`/index-fault rather than do anything useful.
*Fix:* delete it, or reduce it to a `pass`/docstring stub explicitly marked "D5 — rewrite
against compose terrain".

**4. compose.py leaks the terrain-asset temp dir for the non-flat, non-export path.**
`myoassist_utils/compose.py:216-217` puts hfield/texture assets in a `tempfile.mkdtemp`
dir when `export_path is None`, and the cleanup at `:240-244` calls
`assets_dir.rmdir()`, which fails (non-empty) and is swallowed — so for a real terrain the
temp dir persists forever (intentionally kept alive for the returned XML's absolute
refs, but never reclaimed). Currently latent: the only wired non-flat path
(`terrain=<config>`) has no live caller yet (RL configs use `terrain: null`; CO raises
`NotImplementedError` for slopes). *Fix:* register the dir for cleanup at process exit
(e.g. `tempfile.TemporaryDirectory` retained on a module-level registry / `atexit`), or
document the intentional persistence.

### Nits

**5. compose.py imports a private terrains symbol.**
`compose.py:30` does `from myoassist_terrains.config import _config_from_dict` — an
underscore-prefixed function not in the package's `__all__` (which exports only
`build_terrain`, `register_tile`). Fragile across terrains releases. *Fix:* ask terrains to
expose a public constructor, or build the flat `TerrainConfig` via the public dataclasses.

**6. `_CO_DEVICE_MAP` maps `"custom"` but CO no longer supports custom models.**
`reflex_interface.py:39` maps `"custom" -> "Tutorial_L1"`, but `arg_parser.py:27`'s
`--model` `choices` exclude `"custom"`, and `config_parser.create_testenv_from_bat` no
longer forwards `model_path` at all — so custom-model support via the CO `.bat` path is
gone. The map entry is dead. *Fix:* drop the `"custom"` entry (and/or document that custom
XML entry is intentionally unsupported now).

**7. `create_environment` silently falls back to a (now-absent) `model_path` if only one
compose key is set.** `environment_handler.py:20-27` composes only when *both*
`msk_key` and `device_key` are truthy; otherwise it uses `config.env_params.model_path`,
which is `None` in every migrated config, producing an opaque `gym.make` failure. *Fix:*
raise a clear error when exactly one key is set.

**8. Export "standalone" file uses absolute terrain-asset paths, unlike the docstring's
implied portability.** `compose.py:221` absolutizes terrain `file=` refs against the
`_assets` dir, so an exported model is only loadable on the same machine, though the
docstring frames the sibling `*_assets` dir as making it portable. Doc nuance, not a bug.

## Open questions
- **`imitation.json` obs keys (pre-existing, out of scope but adjacent):** this config
  defines no `observation_joint_pos_keys`/`vel`/`sensor` (unlike the `separated_net`
  configs), so the base `EnvParams` empty-list defaults would yield empty qpos/qvel obs —
  inconsistent with the net's `[0,8]`/`[8,17]` slices. Does `ImitationTrainSessionConfig`
  or the imitation env inject defaults? If not, `imitation.json` was already unusable
  before this branch, which strengthens the "retire or fix it" recommendation in finding 1.
- **Keyframe DOF extension:** RL (`myoassist_leg_base.py:170`) and `ReflexEnvV0`
  (`reflex_env.py:67-69`) both index `key_qpos[0]`. This assumes `assist_sim` extends the
  human "stand"/etc. keyframes to cover the device's added DOFs when composing. Verified the
  Tutorial device declares `keyframe_overrides`, but I could not run compose to confirm the
  merged model's `key_qpos` width matches `nq`. PLAUSIBLE-correct; worth a runtime check.
- **`_seat_dz_by_collision` on non-flat terrain:** the linear `new_gap = min(gaps) + dz`
  seating is exact for a flat ground directly beneath the model but only approximate on
  sloped/structured terrain (the closest pair can change as the model descends). Fine for
  what's wired today (flat only); revisit when D5 lands non-flat terrain.
- **`adjust_model_height` site fallback:** `reflex_interface.py:~896` picks `*_touch`
  sites when `*_btm` are absent, but if a composed model exposes *neither* it will `KeyError`
  in the loop. Assumes myo_sim always provides the touch sites; unverified here.
