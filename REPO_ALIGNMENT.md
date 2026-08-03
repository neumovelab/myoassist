# MyoAssist 1.0 — Repository Alignment & Fix Register

Working register of the bug fixes, packaging corrections, and import rewiring needed to
bring the four MyoAssist repositories into the structure described in the 1.0 paper draft
and to unblock the documentation-site overhaul.

**Status:** draft — captured 2026-07-30, not yet actioned.
**Scope:** `myoassist`, `assist_sim`, `myo_sim`, `myoassist.terrains`.

**Branching / working process (decided 2026-07-30):** the myoassist refactor does **not** land on
`dev` or `main` directly. Create a new **`refactor`** branch **off `dev`**; all §A / §F work commits
there (including removing the site — see below), and it is **PR'd into `main` when ready**. The
sibling-repo items (§B assist_sim, §D terrains) proceed on their own repos' branches. The website is
its own **standalone repository** (§D7, revised 2026-07-30 — no longer a branch): the aggregated hub
+ build + media live there and deploy independently, so myoassist stays code-only.
*(`refactor` branch created off `dev` 2026-07-30; the uncommitted working tree — this file — carried
over. Clean-slate removals executed on `refactor` 2026-07-30: see Work log.)*

Citations are `file:line` as observed on the branches listed below. Items in §A were read
directly in this repository; items in §B–§D were captured by survey of the sibling repos.
Re-verify a citation before acting on it — several of these repos are moving weekly.

| Repo | Path | Branch surveyed |
|---|---|---|
| myoassist | `C:\Users\calde\Work\myoassist` | `dev` |
| assist_sim | `C:\Users\calde\Work\assist_sim` | `main` |
| myo_sim | `C:\Users\calde\Work\myo_sim` | `expose-legs-spec` (fork `cnrobbins`; upstream `MyoHub/myo_sim@dev`) |
| myoassist.terrains | `C:\Users\calde\Work\myoassist.terrains` | `velocity-map` |

---

## Target architecture

The end state assumed throughout this document:

```
myo_sim              human musculoskeletal models         →  build_spec(name) -> MjSpec
assist_sim           assistive devices + composition      →  load_combined(msk, device) -> (MjModel, MjData)
myoassist.terrains   task scenario / scene layer          →  build_terrain(cfg) -> MjSpec
myoassist            RL framework + CO framework + shared build/utils + the docs site
```

`myoassist` **retains**: `rl_train/`, `ctrl_optim/`, `myoassist_utils/`, `docs/`, packaging.
`myoassist` **sheds**: `models/` (→ myo_sim + assist_sim) and the vendored `myosuite/`
(→ upstream `myosuite` as an installed dependency).

Note that shedding the vendored `myosuite/` tree is *not* the same as removing the myosuite
dependency. `rl_train` and `ctrl_optim` both build on myosuite's env base classes, gym
registration helpers, and physics wrappers (§A2). The change is from a vendored fork to an
installed package — which means any local modifications currently living in
`myoassist/myosuite/` must first be identified and upstreamed or relocated.

---

## Work log

Branches (all local, nothing pushed/merged; user handles git):
- **myoassist** `refactor` (off `dev`) — the refactor lands here.
- **assist_sim** `docs-drift-fixes` (off `main`).
- **terrains** `fixes` (off `main`; via worktree at `C:\Users\calde\Work\terrains_fixes`).

**2026-07-30 — wave 1 (doc-drift + trivial fixes):**

- *assist_sim* `docs-drift-fixes`, 10 commits:
  - **B2** done (`72cf8e6`) + sweep `1b4ef07` (export-and-load) — CLI `compile`→`combine`, `-o`/`--cache-dir`.
  - **B4** done (`441473f`) — `MSK_MODELS`→`_COMPATIBLE_MSK_KEYS`; corrected `resolve` `(MjSpec, Path)` signature.
  - **B8** done (`e0c47d6`) + sweep `5f6e211` (troubleshooting) — `@mm_refactor`→`@dev`.
  - **B9** done (`42e45e5`) — removed dangling `myo_sim-leg-integration.md` ref in CLAUDE.md.
  - **B6** done (`a3a7a5b`) — README device list + use-custom-devices count → the 11 shipped keys.
    (Left a pre-existing stale `HMEDI_L1` "n/a (needs torso)" MSK-compat cell — reserved for the MSK agent / B12.)
  - **B3** done (`50663e1`) — quickstart uses `load_combined`; `py_compile` passes. *Full viewer run still needs manual check.*
  - **B1** pyproject+CI done (`55ac9fb`, →0.5.2). **CHANGELOG backfill PAUSED** — no release tags/clean boundaries; version→commit map for a human: 0.4.0=`0bd1eb3`, 0.5.0=`d66c15a`, 0.5.1=`d7a9570`, 0.5.2=`62e105d`; `db1d71b`+`328aa1d`=Unreleased.
  - Sweep `1e6f53c` — stale `mm_refactor` prose/comment → `dev` across 5 files (CHANGELOG:48 left as historical).
  - **E3** — verified **no-op**: assist_sim prose already conforms (all dotted; it never imports terrains).
- *terrains* velocity-map working-tree cleanup (untracked-only, no commit): removed `MOSS.png`; de-ndseg-ified 11 source configs/scripts (renamed + stripped internal refs; JSON re-parses). Left `debug_ensemble_scene.xml` + `tmp*.xml` (regenerated/temp).
- *terrains* `fixes`, 3 commits: **D1** (`43a1fce`, declare Pillow), **D2** (`25b04da`, org→neumovelab), **D6** (`72636b5`, pyramid_stairs inverted docstring).

**2026-07-30 — wave 2 (myoassist clean-slate on `refactor`):** stripped the repo to code-only in
prep for the composed-architecture utils. Content preserved on `dev`/history for later
extraction (device assets → assist_sim; docs → new docs repo; tutorials → A9).
- **A1** partial — `cfe69d9` removed `models/` (131 files). *Rewiring of the remaining refs
  (A3/A8/A10) still pending; RL/CO will not run until the shared util lands — expected mid-refactor.*
- **A2** partial — `e677989` removed vendored `myosuite/` (1836 tracked files + pycache remnants);
  `f77cfd6` trimmed `setup.py` (dropped `myosuite*` from packages + `myosuite_files` payload).
  *Installed-myosuite dependency + device-asset relocation still pending (see A2 body).*
- **D7/A9** — `787ad60` removed `docs/` (97 files), `.github/workflows/pages.yml`, and `CNAME`
  (site → standalone docs repo). Tutorials went with `docs/` — proper relocation is A9 (from
  history). Live site unaffected until `refactor`→`main`.
- Repo now: `ctrl_optim/`, `rl_train/`, `myoassist_utils/`, `setup.py`, `requirements.txt`,
  `test_setup.py`, `README`, `LICENSE`. (`requirements.txt` dependency wiring deferred to A7.)

---

## §A — `myoassist`: slim-down and import rewiring

### A1. Remove the bundled `models/` tree

**Where:** `models/22muscle_2D/`, `models/26muscle_3D/`, `models/80muscle/`, `models/mesh/`,
`models/terrain_config.xml`, `models/terrain_config80.xml`.

**Why:** these are the static per-model×device XMLs that the composition pipeline replaces.
Keeping them guarantees drift against `assist_sim`'s device configs and `myo_sim`'s composed
models, and makes the documentation site describe two incompatible model catalogs.

**Fix:** relocate each family to its new owner, then delete.

- 80-muscle device variants (`myoLeg80_{DEPHY,HMEDI,HUMOTECH,OPENEXO,OSL_KA}/`) — superseded by
  `myo_sim` `myolegs` + the corresponding `assist_sim` device key. Confirm each device's YAML
  reproduces the bundled XML before deleting.
- 26-muscle variants — superseded by `myolegs26` + device keys.
- 22-muscle variants — **blocked**, see A6.
- `models/mesh/` — device meshes already live in `assist_sim/assist_sim/models/*/mesh/`;
  anatomical meshes live in `myo_sim/myo_sim/models/meshes/`. Verify no orphans before deleting.
- `terrain_config*.xml` — superseded by `myoassist.terrains` style/pointer files.

**Blocked by:** A6 (22-muscle), and reconciliation of the device catalog (§E1).

---

### A2. Replace the vendored `myosuite/` with an installed dependency

**Where:** `myoassist/myosuite/` (entire tree, including `myosuite/simhive/myo_sim/` v0.1.0).

**Why:** the vendored copy contains an old flat `myo_sim` v0.1.0 that collides by name with the
packaged `myo-sim` 0.2.0 the new architecture depends on (§E2). It also now carries collaborator
content added directly to the fork — `myosuite/envs/myo/wheelchair/myowc+{arm,leftarm}.xml` and
`myosuite/simhive/exo_sim/ankle/` (UT ankle exo), added in `b0e020f` — which needs a home.

**Actual myosuite dependency surface** (what must keep working after the swap):

| Consumer | Import | Purpose |
|---|---|---|
| `rl_train/envs/myoassist_leg_base.py:5,6` | `myosuite.utils.gym`, `myosuite.envs.env_base` | `MyoAssistLegBase` extends `env_base.MujocoEnv` |
| `rl_train/envs/__init__.py:1,2` | `myosuite.utils.gym`, `myosuite.envs.env_variants.register_env_variant` | env registration for `myoAssistLeg-v0`, `myoAssistLegImitation-v0`, `myoAssistLegImitationExo-v0` |
| `myoassist_utils/hfield_manager.py:1` | `myosuite.physics.sim_scene.SimScene` | heightfield manipulation — but see A4 |
| `ctrl_optim/ctrl/reflex/reflex_interface.py:15` | `myosuite.utils.gym` | gym shim |
| `ctrl_optim/ctrl/reflex/reflex_interface.py:151` | `gym.make('myoLegStandRandom-v0')` | **a myosuite-registered env id** — hard coupling |

**Fork-diff investigation (read-only, 2026-07-30) — findings:**
- **The MyoAssist-specific Python is already extracted.** Commits `5880409` ("Took out
  myoassist_rl from myosuite") and `88e9894` ("Remove myoassist_sim submodule") previously pulled
  `envs/myoassist/` + `rl_train/myoassist/` out of the vendored tree. What remains local is
  **XML/mesh assets** + the old flat `myo_sim` v0.1.0 vendoring — no MyoAssist Python lives inside
  myosuite anymore.
- **`rl_train`/`ctrl_optim` import only *stock* myosuite modules** (`utils.gym`,
  `envs.env_base`, `envs.env_variants`, `physics.sim_scene`) — see the table above. Nothing in
  them imports a local myosuite addition.
- **Local additions are all from two commits by Calder** (`c8bb0d8` "updated simhive",
  `b0e020f` "added collab envs"), and **none are referenced by any Python** (grep of
  `myowc`/`wheelchair`/`exo_sim`/`ut_ankle` across `*.py` = zero) — pure `<include>`-chained assets,
  relocatable without touching imports:
  - `envs/myo/wheelchair/myowc+{arm,leftarm}.xml` + `simhive/myo_sim/wheelchair/` +
    `simhive/myo_sim/wheelchairandhuman/` + wheelchair hardware meshes in `simhive/myo_sim/meshes/`
    → **assist_sim** (raw material for device #15, the wheelchair; not yet an assist_sim key).
  - `simhive/exo_sim/ankle/` (UT ankle exo, "Vikash Kumar 2024") → **assist_sim** — **but check for
    redundancy first:** assist_sim already ships `UTAnkleExo_L2`; this myosuite copy may be an
    earlier variant to **drop** rather than relocate.
  - `simhive/myo_sim/torso/myotorso_exosuit.xml` + exo-belt meshes (`lower/upper_exo_belt.stl`,
    from `c8bb0d8`) → **assist_sim if MyoAssist-specific, else drop** — needs the upstream myo_sim
    diff to decide (reads as assistive).
  - Trivial include-path renames (`9eed52b`) → upstream or moot after A1.
- **`myoLegStandRandom-v0`** is registered in **stock** `myosuite/envs/myo/myobase/__init__.py:294`
  → `walk_v0.py:ReachEnvV0` (untouched since the initial vendored commit → appears upstream, not a
  MyoAssist addition). Its default `model_path` points at `simhive/myo_sim/leg/myolegs.xml`, but
  CO passes its own `model_path`, so the default is overridden. This is the A5 coupling.

**Fix (revised with findings):**
1. Relocate the three asset groups above to `assist_sim` (fix relative `include` paths after
   move); **drop** the myosuite UT-exo copy if redundant with `UTAnkleExo_L2`; decide exosuit via
   the upstream diff.
2. Add `myosuite` to `requirements.txt`; delete the vendored tree.
3. Resolve `myoLegStandRandom-v0` (A5).

**Before deleting the vendored tree, confirm:** (a) an installed upstream myosuite still registers
`myoLegStandRandom-v0` with a compatible `ReachEnvV0` signature (A5); (b) upstream `myo_sim`
exposes the include paths MyoAssist still needs — largely **moot once A1 removes `models/`**, but
verify during transition (the 0.1.0 pin means renamed-path drift, cf. the `myotorso_rigid_chain`
fix in `9eed52b`).

**Blocked by:** A5. (E1 dependency removed — the device catalog is fixed, and the wheelchair/UT
assets map onto it.)

---

### A3. Rewrite `ctrl_optim`'s model resolver onto `assist_sim`

**Where:** `ctrl_optim/optim/optim_utils/resolve_path.py:26-84` (`resolve_model_path`),
`:123-133` (`get_available_models`), `:136-157` (`validate_model_config`).

**Why:** this function is the single point where CO selects a model, and it is written entirely
in terms of the bundled tree — it maps `{baseline, dephy, hmedi, humotech, tutorial}` × `{2D, 3D}`
onto `models/{22muscle_2D,26muscle_3D}/myoLeg{22_2D,26}_{SUFFIX}.xml` and raises
`FileNotFoundError` against a path under the project root (`:78`). It also silently omits
OPENEXO and OSL_A from `model_mapping` (`:63-69`) even though those XMLs ship — so the CO
framework can't currently reach two of the bundled devices.

**Fix:** replace with a call through `assist_sim.load_combined(msk_key, device_key)`, and replace
the `model`/`mode` argument pair with explicit MSK and device keys. Note the semantics change:
`resolve_model_path` returns a path, `load_combined` returns `(MjModel, MjData)`. Callers to
update: `ctrl_optim/ctrl/reflex/reflex_interface.py:27`, `ctrl_optim/optim/cost_functions/walk_cost.py`,
`ctrl_optim/optim/optim_utils/config_parser.py:138-149`, `ctrl_optim/eval/gait_evaluator.py:18,64`.

**Follow-on:** the `mode` ('2D'/'3D') concept disappears with the 22-muscle planar model. Decide
whether CO's sagittal-plane reflex variant is selected by controller config instead (§A6).

---

### A4. Retire `HfieldManager` in favour of `myoassist_terrains`

**Where:** `myoassist_utils/hfield_manager.py`; consumed at
`rl_train/envs/myoassist_leg_base.py:11,183-186`.

**Why:** `HfieldManager` implements four terrain types (`flat`, `random`, `harmonic_sinusoidal`,
`slope`) driven by a `terrain_type` + space-separated `terrain_params` string. `myoassist_terrains`
implements nine tile types on a grid-of-tiles model with connectors, plus the velocity-map
subsystem the paper's Fig. 3(b) depends on. These are different abstractions, not versions of one.

**Fix:** replace the `HfieldManager` call site with terrain composition from
`myoassist_terrains.build_terrain(cfg)`, layered onto the assist_sim model-only output. This
requires a schema change in the RL training config (`terrain_type`/`terrain_params` →
a terrain JSON config reference) and touches every bundled `train_configs/*.json` plus the four
committed tutorial `session_config.json` files.

**Note:** the geom named `terrain` that `myoassist_leg_base.py:180-186` manipulates is also
emitted by the terrains composer as a transparent backstop (`composer.py:608`), specifically so
models carrying `<contact><pair geom1="terrain">` still compile. The contract survives the swap,
but the mechanism differs — verify rather than assume.

**Blocked by:** decision on RL terrain-config schema; see also §D5 (no curriculum abstraction).

---

### A5. Unify CO's env onto the shared pipeline (retire `myoLegStandRandom-v0`)

**Status:** DECIDED — 2026-07-30. **Full unify** (with A10): migrate CO off the myosuite-registered
`myoLegStandRandom-v0` onto a MyoAssist env built via the shared compose+load util, so RL and CO
share one env pathway. (Chosen over the minimal "keep it + verify upstream" option because it
materially cleans up CO.)

**Where:** `ctrl_optim/ctrl/reflex/reflex_interface.py:151` — `gym.make('myoLegStandRandom-v0',
model_path=…)`.

**What it is (verified 2026-07-30):** `myoLegStandRandom-v0` → myosuite's `ReachEnvV0`
(`myosuite/envs/myo/myobase/walk_v0.py:13`), a **reaching** env (obs `tip_pos`/`reach_err`, rewards
`reach`/`bonus`). CO uses **none** of that — it borrows the env purely as a **steppable MuJoCo sim
container**, passing its own `model_path`. Everything CO touches is generic `MujocoEnv` API:
`self.env.sim` (model+data, read for reflex sensors — `reflex_interface.py:333–463`),
`step`/`reset`/`forward`/`dt`/act space, and the keyframe reset-to-stand that `ReachEnvV0._setup`
sets via `init_qpos = key_qpos[0]` (`walk_v0.py:58`). It also drags in reach-task cruft CO doesn't
want (`target_reach_range` required arg, reach obs/reward).

**Fix:** point CO's reflex env at the shared compose+load util (A10) and a MyoAssist `MujocoEnv`
(the same base RL uses); drop the `gym.make('myoLegStandRandom-v0')` borrow. The ~30
`self.env.sim.*` accesses use the same API and read by body/joint/sensor name (same models), so
they carry over mechanically.

**Must preserve (breakage risks):** (1) keyframe reset-to-stand (`init_qpos = key_qpos[0]`);
(2) CO's `timestep=0.001` + `frame_skip` + neural-delay setup (`reflex_interface.py:148–162`);
(3) CO's own direct hfield/slope manipulation (`reflex_interface.py:224–254`) — CO has its **own**
terrain path, separate from RL's `HfieldManager`; unifying terrain is A4/D5.

**Bonus:** full-unify **removes the A2 upstream-registration risk** — once CO no longer calls
`myoLegStandRandom-v0`, dropping the vendored myosuite can't break CO (RL uses only stock
`env_base`/`env_variants`).

**Blocks:** A2 (de-risks it). **Blocked by:** A10 (shared util).

---

### A6. Decide the fate of the 22-muscle 2D models

**Status:** DECIDED — 2026-07-30. Path A (planar model stays), with the 26→22 reduction
implemented in `assist_sim`, not `myo_sim`, this pass. See Resolution below and §B12.

**Where:** `models/22muscle_2D/` (7 XMLs); `assist_sim/assist_sim/registry.py:70-74`
(`myolegs22` registered as `None`, raises `ValueError`, "not implemented yet").

**Why:** this is the highest-impact blocker. The 22-muscle planar model has no equivalent in
`myo_sim`; `myolegs22` is a planned 26→22 mjspec reduction that does not exist. Meanwhile it is
the model every RL tutorial, every CO tutorial, all four `docs/tutorial/*.ipynb`, and all four
committed pretrained sessions under `docs/assets/tutorial_rl_models/` run on. It is also the
default in `rl_train/run_sim_minimal.py:4` and CO's `mode='2D'` path (§A3).

The model is documented in three mutually inconsistent states across the repos: "planned"
(`assist_sim/registry.py:70`), "not built yet" (`assist_sim/docs/available-models.md:16`), and
"seven shipping XMLs" (`myoassist/models/22muscle_2D/`).

**Options:**
- **(a)** Implement the 26→22 reduction in `myo_sim`, register `myolegs22` in `assist_sim`.
  Preserves every tutorial and the CO 2D path. Largest upstream effort.
- **(b)** Re-anchor all tutorials to `myolegs26`. Requires retraining the four bundled RL
  sessions and re-tuning CO's preoptimized parameter sets, which are model-specific.
- **(c)** Keep a legacy 22-muscle track outside the composition pipeline for tutorials only.
  Cheapest, but reintroduces exactly the dual-catalog problem A1 exists to remove.

**Decision required before:** A1, A3, and the entire Tutorials section of the docs site.

Related: three systemic bugs were previously identified in the myoassist 22/26-muscle baselines
and should be fixed at source regardless of which option is taken — `gastroc_l` P2_z sign, the
`knee_translation2` coupler range, and the EDL/FDL `gainprm`/`biasprm` Fmax mismatch
(EDL 818.778 → 553.241, FDL 1081.909 → 332.13).

**Resolution (2026-07-30):** Path A, **modified**. The 22-muscle planar model stays in 1.0 —
both because the paper commits to a sagittal-plane reflex controller and because a fast planar
model is worth keeping for tutorials and CMA-ES — but the 26→22 reduction is implemented **in
`assist_sim`, not `myo_sim`**, this pass. No `myo_sim` edits (§C is deferred). Rationale: keeps
the reduction downstream where we control it, unblocks the docs/1.0 work without upstream
coordination, and the later lift into `myo_sim` as a composed `myolegs22` is a mechanical move —
the util operates on the already-imported `myolegs26` `MjSpec`, so relocating it upstream changes
only where it lives, not what it does. Implementation is tracked as **§B12**. The three baseline
muscle bugs above fold into the util's output rather than being fixed "at source" — the reference
model already carries corrected EDL/FDL Fmax. The myoassist-side load/resolve mechanism (how
RL/CO name and obtain the model) is **now A10** (decided 2026-07-30: a shared compose+load util,
option 3); it does not gate this decision.

Consequently A1, A3, and A8 are no longer blocked on a *scope* decision — they now wait only on
§B12 (a working `myolegs22`) and on the myoassist-side resolution mechanism.

---

### A7. Packaging and verification

**Where:** `setup.py`, `requirements.txt`, `test_setup.py`.

- `setup.py:62` — `find_packages(include=("myosuite*", "myoassist*", "rl_train*", "ctrl_optim*"))`
  must drop `myosuite*`; `package_data` (`:61`) must drop `myosuite_files` (`:35`) and the
  `models/` payload.
- `requirements.txt:11` — `mujoco==3.3.3` conflicts with `assist_sim`, which requires
  `>=3.3.4` for `MjSpec.delete`. Bump to `>=3.3.4`.
- `requirements.txt` — add `myosuite`, `myo-sim`, `assist_sim`, `myoassist-terrains`. Note
  `myo-sim` cannot come from PyPI (§C3) and `assist_sim`/`myoassist-terrains` are not published,
  so these will be git URLs until releases exist.
- `test_setup.py:91-105` — the 13-test suite checks `import myosuite`, `import rl_train`,
  `import ctrl_optim`. Extend to cover `myo_sim`, `assist_sim`, `myoassist_terrains`, and to
  verify one composed model builds end to end. This is the install-verification step the
  Getting Started page will tell users to run, so it should exercise the full stack.
- `MyoAssist.egg-info/` is a tracked build artifact listing the vendored tree; regenerate or remove.

---

### A8. Hardcoded tutorial model path

**Where:** `rl_train/run_sim_minimal.py:4` —
`model_path="models/22muscle_2D/myoLeg22_2D_TUTORIAL.xml"`.

**Why:** the documented first-run command for new users
(`python rl_train/run_sim_minimal.py`) fails the moment `models/` is removed.

**Fix:** route through `assist_sim` using the `Tutorial_L1` device key, which exists precisely
for onboarding. Depends on A6 for the MSK key.

---

### A9. Relocate tutorials off the website

**Status:** PLANNED — from the D7 decision (2026-07-30).

**Where:** `docs/tutorial/*.ipynb` (4 notebooks); `docs/assets/tutorial_rl_models/` (4 pretrained
sessions incl. ~250 MB of model `.zip`s); tutorial commands embedded in
`docs/reinforcement-learning/index.md` (e.g. the `run_policy_eval.py docs/assets/tutorial_rl_models/...`
examples).

**Why (user, 2026-07-30):** tutorials should **not** be embedded on the website. Moving them out
sheds the heavy pretrained-model payload that drove the media-weight concern, and fits the modular
model — runnable tutorial material lives with the code, the site links to it.

**Fix:** relocate notebooks + sessions into a proposed **`tutorials/` directory in the myoassist
repo**, split `tutorials/rl/` and `tutorials/co/` (CO currently has no notebook track — this is
also where a CO tutorial would land, closing a known gap). The website links out to them rather
than hosting them. Marked *potential/proposed* per the user; confirm the exact layout before
moving files. Pretrained `.zip`s should not live in `main` history long-term — decide whether they
sit in the website branch's assets, a release attachment, or are regenerated on demand.

**Blocked by:** the tutorials' own model dependency (A6/B12 — they currently run on 22-muscle).

---

### A10. Unified RL/CO model compose + load pipeline (the "resolve mechanism")

**Status:** DECIDED (design) — 2026-07-30; implementation planned. Keystone of the §A slim-down —
A1, A3, A5, A8 reference this as "the resolve mechanism."

**Goal (user, 2026-07-30):** minimize the separation between the RL and CO frameworks — they
should share the same model/env compose + load utilities.

**Current state (verified 2026-07-30):**
- **RL** — `MyoAssistLegBase` (`rl_train/envs/myoassist_leg_base.py:17,46`) subclasses myosuite
  `env_base.MujocoEnv` (`myosuite/envs/env_base.py:41`); `model_path` → `SimScene.get_sim(model_path)`
  (`env_base.py:60`). Registered gym envs `myoAssistLeg-v0` / `…Imitation-v0` / `…ImitationExo-v0`
  (`rl_train/envs/__init__.py`). It carries RL-specific machinery (target-velocity modes, joint-limit
  sensors, obs keys, terrain via `HfieldManager` set once in `_setup`, reward/termination, lumbar).
- **CO** — `myoLeg_reflex` (`ctrl_optim/ctrl/reflex/reflex_interface.py:29`) does **not** use
  `MyoAssistLegBase`; it builds `gym.make('myoLegStandRandom-v0', model_path=pathAndModel)`
  (`:151`), a myosuite-registered env, with the path from `resolve_model_path(model, mode, path)`
  (`resolve_path.py:26`, obsolete after A3). Today RL and CO share almost nothing beyond myosuite.

**MujocoEnv input surface (confirmed — decisive):** `DMSimScene._load_simulation`
(`myosuite/physics/mj_sim_scene.py:27-45`) accepts `model_handle` **only as a `str`**: ends `.xml`
→ `from_xml_path`; contains `"<mujoco"` → `from_xml_string`; else → `from_binary_path`. A non-str
(raw `MjModel`/`MjSpec`) raises `NotImplementedError` (`:45`).
→ Option (2) "pass the env an in-memory `MjModel`" is impossible without patching myosuite. But the
env **accepts an XML string**, so in-memory compose is achievable with no disk round-trip and no
myosuite edit: `spec.to_xml()` → pass the string.

**Resolution — option (3), shared util:**
- Add **one myoassist compose+load utility** (home TBD — `myoassist_utils/` or new
  `myoassist/compose.py`) that **both RL and CO call**. It: (1) composes human+device via
  `assist_sim.load_combined`/spec; (2) merges terrain (`myoassist.terrains`) in-memory at the
  mjSpec/XML level (the figure-pipeline approach); (3) returns an **XML string** with absolute
  asset paths (env → `from_xml_string`, no temp file on the run path); (4) **optionally writes the
  combined XML to disk** (`export_path=`) for inspection / viewer / QC — the essential export path.
- The env `model_path` contract is unchanged (accepts path OR xml-string), so **myosuite is not
  modified** and both frameworks load identically.
- **Asset caveat:** `from_xml_string` resolves relative asset paths against CWD unless absolute —
  the util must absolutize mesh/hfield/texture paths. `assist_sim.export_combined_xml` already does
  this for exports; reuse that logic for the string form.

**Open sub-decision (flagged):** how far to unify. Near-term this shares the *model compose+load
util* while RL and CO keep distinct env classes (RL: `MyoAssistLegBase` + obs/reward; CO: reflex
env + delays/sensors). Fully unifying the *env class* — moving CO off `myoLegStandRandom-v0` onto a
MyoAssist env fed by the shared util — is the deeper move, entangled with A5. Recommend: land the
shared util first (unblocks A1/A3/A8), tackle env-class unification with A5.

**Blocks:** A1, A3, A8. **Blocked by:** B12 (current configs need a working `myolegs22`);
assist_sim asset-path/export reuse.

---

## §B — `assist_sim`

### B1. Version number disagreement

`assist_sim/__init__.py:40` declares `__version__ = "0.5.2"`; `pyproject.toml:7` declares
`version = "0.1.0"`; `.github/workflows/test.yml:50` asserts on the built filename
`dist/assist_sim-0.1.0-py3-none-any.whl`. `CHANGELOG.md` stops at `[0.3.0]`.

Consequence: the version is also a cache-key input (`assist_sim/cache.py`), so a stale
`pyproject` version weakens cache invalidation. Reconcile all four, and backfill 0.4.x/0.5.x
changelog entries.

### B2. CLI documentation describes the wrong subcommand

`README.md:37` and `docs/usage.md:99,108,114` document
`python -m assist_sim compile ... --export ... --cache ...`. The implemented subcommand is
`combine` with `-o/--output` and `--cache-dir` (`assist_sim/__main__.py:69,82-102`).

Every copy-pasteable command in the docs is wrong. Fix before porting these pages to the site.

### B3. `examples/quickstart.py` is broken against the current API

`examples/quickstart.py:120-127` does `msk_path, device_path = resolve(...)` then passes
`human_xml=str(msk_path)`. Since the Phase-2 in-memory change (`f560e67`), `registry.resolve()`
returns `(MjSpec, Path)` — an `MjSpec`, not a path.

This matters beyond the example: `quickstart.py` already carries tuned per-MSK camera poses
(`:150-166`) and is the natural screenshot/GIF harness for the ~22 MSK×device combinations the
docs site needs. Fixing it is a prerequisite for the media pass.

### B4. `docs/usage.md` documents a non-existent symbol

`docs/usage.md:129-134` documents `registry.MSK_MODELS`. No such attribute exists; the curated
table is `registry._COMPATIBLE_MSK_KEYS` (`registry.py:69`). Either promote it to a public name
or correct the doc. Given the docs site will publish this page, promoting is preferable —
`assist_sim/__main__.py` and `examples/quickstart.py` both already reach into the private name.

Same page, `:43-58`: the documented `resolve_model_path(msk, device, cache_dir, export_dir) -> str`
signature no longer matches `loading.py:39`.

### B5. `myofullbody` is registered but untested and undocumented

Added in `db1d71b` at `registry.py:81` and wired into seven device configs' `compatible_msk`
lists, but absent from `tests/test_smoke_combinations.py:30-47` and from every file in `docs/`.

The paper leads on 460-muscle full-body support, so this needs both a smoke test and a docs
entry before submission.

### B6. Device count disagreement between README and docs

`README.md` lists 7 devices; `docs/available-models.md` lists 11.
`docs/how-to/use-custom-devices.md:3` says "seven bundled devices". Missing from the README:
`Anatomics_L1`, `Hippo_L1`, `KFoot_L1`, `UTAnkleExo_L2`.

### B7. The `L1`/`L2` device-level convention is undefined

**Status:** DECIDED (definition established via read-only investigation, 2026-07-30). Ready to
document; no design change needed.

Every device key carries an `L1` or `L2` suffix and no document explains it. From the commit
history (`0bd1eb3`, "add UT exo config + level 2 config utils") it appears to distinguish rigid
re-parented attachment (L1) from free-rooted, equality-constrained attachment (L2) — confirmed
below.

**Findings (confirmed against code):** the level token is **not code-enforced** and **not a
fidelity tier** — no code branches on `L1`/`L2` (grep found only the docstring example at
`registry.py:184` and the literal `name:` strings in configs). It is purely the **config-file
stem** that becomes half of the registry key (`registry._device_key`, `registry.py:181-190`;
scan at `:193-217`). In current practice it correlates 1:1 with the **attachment paradigm**, which
the pipeline selects from config *content*, never from the suffix:
- **L1 — rigid attach.** Device bodies re-parented directly onto MSK bones via `attachments`
  (`parent_body: <bone>`). Original config schema. Examples: `DephyExoBoot/L1config.yaml`,
  `Hippo/L1config.yaml`. (`combine.py:270-319`, rigid branch.)
- **L2 — free-rooted + equality-constrained.** Device roots attach to `world` (keep their own
  `<freejoint>`) and are fastened to the leg with `<equality>` (`connect`/`weld`) constraints.
  Requires the level-2 config utilities added in `0bd1eb3` (`EqualityConstraint` `config.py:114`,
  per-MSK `equality:`, `_WORLD_PARENTS` `combine.py:76`, `_add_equalities` `combine.py:393-434`).
  Sole example: `UTAnkleExo/L2config.yaml` (the only config using `equality:` / `parent_body:
  world`).
- The token **before** the level (e.g. `A_`, `KA_` in `OpenSourceLeg_A_L1` / `_KA_L1`) is a device
  **variant** (ankle vs knee-ankle), not a level.
- Scheme is open-ended (autodiscovery accepts any `<variant>config.yaml`); no `L3` exists or is
  planned.

**Fix:** write the drop-in definition above into the assist_sim device docs and the docs-site
device catalog. Since it's a filename convention, consider also stating it in
`docs/device-config-reference.md` (which at `:47` only calls it a "Convention: PascalCase + `_L1`
suffix" without defining the level). No code change.

### B8. Stale install instruction

`docs/getting-started.md:36` instructs installing `myo_sim` from the `@mm_refactor` branch. That
branch was renamed to `dev` (`myo_sim` commit `d6deb6e`).

### B9. Dangling internal references

`CLAUDE.md:70` references `myo_sim-leg-integration.md`; `WHATS_NEXT.md:18` references
`scratchpad/myosim_to_xml_bug_report.md`. Neither file exists in the repo.

### B10. `myo_sim` is an undeclared runtime dependency

`pyproject.toml:36-43` declares only `mujoco`, `PyYAML`, `numpy`. `myo_sim` is imported at
`registry.py:125` and is required for every non-trivial operation. It is currently an optional
import with a helpful error, and CI deliberately omits it (24 tests skip,
`.github/workflows/test.yml:28-31`).

Decide whether this stays optional-with-clear-error (defensible, and worth documenting as
deliberate) or becomes a declared extra such as `assist_sim[msk]`. Either way the docs must state
it, since a bare `pip install assist_sim` yields a package that cannot compile anything.

### B11. Dependency on a private `myo_sim` attribute

`registry.py:166` reads `myo_sim._COMPOSED_MODELS`. This is an availability probe that avoids a
full compile, and it works, but it is a private name that upstream may rename without notice.
Request a public accessor upstream (§C4).

---

### B12. Implement `myolegs22` as an in-`assist_sim` mjspec reduction

**Status:** PLANNED — from the A6 resolution (2026-07-30).

**Where:** `assist_sim/assist_sim/registry.py:70-74` (the `myolegs22 = None` stub); a new util
module, TBD (e.g. `assist_sim/assist_sim/reduce_legs.py`).

**What:** a lightweight `MjSpec` transform that takes the imported `myolegs26` spec
(`myo_sim.build_spec("myolegs26")`) and reduces it to the planar 22-muscle model in memory, then
registers `myolegs22` so `load_combined("myolegs22", device)` behaves like any other MSK key.
Rigid-tendon / stock-MuJoCo (assist_sim requires `mujoco>=3.3.4`); the compliant-tendon variant
is a separate `myo_comp` track and out of scope here.

**Reference output:** `C:\Users\calde\Work\compile_check\myoLeg22_2D_myolegs26_rigid.xml` already
reflects the target, and its header documents the exact recipe — reproduce these as `MjSpec` ops
rather than the original ElementTree pass:
- yaw `sacrum`, `pelvis` (and the `legs_attach` frame) so the model faces +x / up +z;
- replace the free root with planar `pelvis_tx` / `pelvis_ty` / `pelvis_tilt`;
- remove frontal-plane hip DOF (`hip_adduction`, `hip_rotation`) both sides;
- remove abd/add actuators + tendons (26 → 22 muscles);
- strip orphaned trunk-muscle sites; keep torso / HAT;
- carry corrected EDL/FDL Fmax (553.241 / 332.13) — this absorbs the A6 baseline bugs;
- port the 5 keyframes from BASELINE (coupler-recompute + clamp).
Target signature: **nq 39 / nu 22 / nkey 5 / neq 28** (matches the derived twins built 2026-07-22).

**Design intent:** keep the transform self-contained and free of assist_sim-specific assumptions
so it lifts into `myo_sim` later as a composed `myolegs22` `BuildStrategy` with minimal change.
Add a smoke-combination test once it lands (also closes part of B5's untested-MSK gap pattern).

**Blocks:** A1, A3, A8. **Blocked by:** nothing — `myolegs26` already imports.

---

## §C — `myo_sim`

**Status (2026-07-30): DEFERRED this pass.** Per the A6 decision, no `myo_sim` repo edits now.
The items below are retained as known-issues to (a) surface on the docs site where they affect
users and (b) hand upstream later. The eventual home for the `myolegs22` reduction (§B12) and for
a restored `stand` keyframe (§C2) is here, but both are handled or worked around downstream for
the time being.

### C1. Wiki describes a build strategy that does not exist

`docs/wiki/build-and-composition.md:60` states that `myolegs26` uses a `LEGS26_BASE` strategy and
is a "reduced 26-muscle, legs-only base; free-floating root + `stand` keyframe".

All three claims are now wrong: the strategy is `LEGS26_BODY`, the model is torso-scaffolded
(refactored in `311157d`), and the `stand` keyframe was dropped in favour of `qpos0`. Downstream
docs already record the consequence — `assist_sim/docs/available-models.md` notes "No keyframe.
The model loads at `qpos0`."

Same file, `:23-28`: the generate-target list omits `leg/myolegs26.xml`.

### C2. `add_keyframes()` is orphaned

`myo_sim/build/utils.py:155-171` has zero callers. It was added alongside
`myolegs26_keyframes.xml` (`1e8a05e`); that file was deleted in the `311157d` refactor. Either
remove the function or restore a keyframe path — the latter would resolve a real downstream
usability issue, since `myolegs` and `myolegs26` both now open at `qpos0` rather than a stand pose.

### C3. PyPI name collision

`README.md:57-58` warns that the PyPI package `myo-sim` "currently points to an older
incompatible version. Use the git install above until a new release is published."

This affects the docs site's install instructions directly: `pip install myo-sim` installs the
wrong software. Until a release lands, Getting Started must use the git URL and say why.

### C4. Provide a public composed-model accessor

`_COMPOSED_MODELS` (`myo_sim/__init__.py:66-71`) is consumed by `assist_sim` (§B11). Promote to a
public name, or add a `myo_sim.available_models()` accessor.

### C5. `repository-map.md` is stale

Reports `meshes/ # 127 shared STL files` (actual: 86); omits `sensors/`, `legacy/`, and
`myolegs26_contacts.xml`.

### C6. `myolegs26` has no preview image

`README.md:20` has an empty preview cell, no fidelity plot (unlike `myolegs`, `myoarm`,
`myohand`, `myotorso`, which each have one under `docs/images/`), and a 25-line README section
against `myolegs`'s 54. `myolegs26` is the primary MSK model for MyoAssist's gait work and the
one most device configs target, so this is the most-visible gap.

### C7. Fragile external image hosting

The `README.md` preview column links to `user-images.githubusercontent.com` URLs and, for
MyoTorso, to a **personal fork** (`github.com/cherylwang20/myo_sim/blob/cec3ce2.../MyoBack.png`).
If the docs site reuses these previews, host them in-repo instead.

---

## §D — `myoassist.terrains`

### D1. Pillow is an undeclared hard dependency

`src/myoassist_terrains/tiles/rough.py:28` does `from PIL import Image`, but Pillow is absent
from `pyproject.toml:25-29` and from the direct dependencies in `uv.lock`. It arrives only
transitively via the `[render]` extra (`mediapy → matplotlib → pillow`).

**Consequence:** a plain `pip install -e .` produces an install where the `rough` tile — one of
the nine, and the only heightfield-backed one — raises `ImportError`. Add Pillow to the base
dependencies.

### D2. Package URLs point at the wrong GitHub organization

`pyproject.toml:44-45` gives `github.com/neumove/...`; the actual remote is
`github.com/neumovelab/myoassist.terrains`.

### D3. The velocity-map subsystem is entirely undocumented

`velocity_map.py` (456 lines) and `velocity_arrows.py` (153 lines) have no README coverage, yet
the paper's Fig. 3(b) is a velocity map and it is the repo's active development thrust.
`render_terrain_check.py`, `camera_convert.py`, `_build_velocity_config.py`,
`_make_tiled_rings.py`, `utils/style/terrain/default.xml`, and `MOSS.png` are likewise absent
from the README's utilities table.

### D4. Private-API coupling inside the package

`velocity_map.py:19` imports `composer._compute_cell_layouts` and `composer._resolve_tiles`.
Promote both, or move the layout computation into a shared internal module — the velocity map is
becoming a first-class feature and shouldn't reach through a private boundary.

### D5. No curriculum or difficulty abstraction exists

**Status:** DECIDED — 2026-07-30. **Build it (option 3).** Full handoff spec below; owner TBD
(this item is intended to be delegated to a collaborator/agent). This is a feature, not a docs
fix, and it spans two repos (`myoassist.terrains` + `myoassist`).

There is no `difficulty` parameter, no progression schedule, and no episode-level regeneration
hook. Difficulty is expressible only indirectly via `randomization.weights` and `param_ranges`.
The current site promises a "Terrain Curriculum — train on a progression of terrains from flat to
rough" (`docs/reinforcement-learning/index.md:85`), which no code backs.

---

#### D5 — Handoff spec: terrain difficulty & curriculum

*Self-contained. A collaborator should be able to act on this without re-deriving the survey.*

##### 1. Current reality (verified 2026-07-30)

**`myoassist.terrains`:**
- `build_terrain(config, output_dir) -> MjSpec` (`composer.py:172`) builds a terrain **once** from
  a static `TerrainConfig` (`config.py:113`). It is deterministic given `randomization.seed`
  (`composer.py:282`, `np.random.default_rng`).
- Difficulty is expressible only indirectly: hand-authored `tiles`, or `randomization.weights` +
  `param_ranges` (`config.py:57-83`). There is no scalar difficulty knob and no notion of
  "harder over time."
- **Two tile categories, and they behave completely differently under change:**
  - *Geom-based* (flat, stairs, slope, pyramid_stairs, discrete_obstacles, stepping_stones,
    boulders, gap) — emitted as `worldbody` geoms. **Changing them requires recompiling the
    `MjModel`.** You cannot swap a stairs tile for a taller-step stairs tile in a live model.
  - *Heightfield-based* (rough only) — writes a PNG and declares an `<hfield>`. Its **data can be
    mutated in place** on `mjModel.hfield_data` without recompiling.

**`myoassist` (the RL consumer):**
- Terrain config is two strings: `terrain_type` (`flat`/`random`/`harmonic_sinusoidal`/`slope`)
  and `terrain_params` (space-separated), `config.py:54-56`. This is the **old `HfieldManager`
  vocabulary**, not the terrains package (see A4) — the two aren't connected yet.
- `myoassist_utils/hfield_manager.py` mutates a single hfield named `terrain` in place
  (`self._hfield.data[:] = ...`). It never recompiles. It also builds a "safe zone" mask around
  the start so the model doesn't spawn on a cliff (`hfield_manager.py:45`).
- **Terrain is static for the entire run.** `set_hfield` is called once in `_setup`
  (`myoassist_leg_base.py:185-186`); `reset()` (`myoassist_leg_base.py:373`) re-rolls target
  velocity and buffers but **never touches terrain**.
- **There is no curriculum machinery anywhere in `rl_train`** (verified by grep). The site's
  "Terrain Curriculum" claim is aspirational.

##### 2. The core problem

A curriculum must make terrain **get harder over training** and ideally **vary per episode**. The
cheap path (in-place `hfield.data` mutation) only covers heightfield terrain; the terrains
package's expressive vocabulary is geom-based and needs a recompile to change. So the design must
explicitly decide *what can change without a recompile* and *how/when recompiles happen* — under
vectorized training (`num_envs` up to 32), where each recompile stalls a worker and invalidates
running sim state.

##### 3. Requirements

- **R1 — difficulty scalar.** A single `difficulty ∈ [0, 1]` that monotonically scales terrain
  challenge, with a documented, reproducible mapping onto tile selection + parameters.
- **R2 — deterministic resample.** `(difficulty, seed) → terrain` is reproducible, for eval
  parity and debugging.
- **R3 — progression schedule.** Advance difficulty during training from a signal — timesteps
  (simple) and/or eval performance (adaptive). Configurable.
- **R4 — episode regeneration.** A hook to obtain a fresh terrain at the current difficulty on
  (or between) episode resets, honoring the fast-path/recompile split of §2.
- **R5 — safe-zone preservation.** Whatever regenerates must keep a walkable start region
  (port/generalize `HfieldManager._make_safe_zone`).
- **R6 — vectorized-training-safe.** Works with SB3 vectorized envs; recompiles (if any) are rare
  and coordinated, not per-episode-per-worker.

##### 4. Proposed design

**Terrains side (`myoassist.terrains`):**
- Add a `difficulty` concept to the config layer. Recommended: a `CurriculumSpec` (or extend
  `RandomizationSpec`) that declares, per tile type, how its `weights` and each `param_range`
  interpolate as a function of `difficulty` — e.g. `rough.vertical_relief: [0.1 @ d=0 → 1.5 @ d=1]`,
  and weight ramps that phase harder tiles in as `d` rises. Provide
  `build_terrain_at(base_config, difficulty, seed) -> MjSpec` layered on the existing
  `build_terrain`.
- Formalize the **structure vs. field** split as public API:
  - `resample_field(model, data, difficulty, seed)` — fast path; updates `hfield_data` in place
    (this is the generalization of `HfieldManager`, now driven by the terrains noise generator and
    the difficulty mapping). No recompile.
  - `build_terrain_at(...) -> MjSpec` — full path; used when geom structure must change (new tile
    types/counts), requiring the consumer to recompile.
  - Document a **"what triggers a recompile" contract** so consumers know which difficulty changes
    are hot-swappable and which aren't.
- Keep everything deterministic in `(difficulty, seed)`. Unit tests: monotonicity (higher `d` ⇒
  higher relief/step metrics), reproducibility, safe-zone intact.

**myoassist RL side (`myoassist`, ties into A4):**
- Replace the `terrain_type`/`terrain_params` strings with a terrain-config reference (a
  `myoassist.terrains` JSON) plus a `curriculum` block (schedule kind, start/end difficulty, pace,
  and whether difficulty is timestep- or performance-driven).
- Add a **curriculum callback** (new, alongside `learning_callback.py`) that owns the difficulty
  schedule and pushes difficulty into envs.
- Wire regeneration into the env: prefer `resample_field` on `reset()` for the common
  (heightfield) case; escalate to a rebuild only at **stage boundaries** (rare), coordinated
  across the vector to bound cost. Preserve the safe zone (R5).
- Realtime/eval mode must pin `(difficulty, seed)` for reproducible evaluation.

##### 5. Phasing

1. **Terrains — difficulty + fast path.** `difficulty` mapping, `build_terrain_at`,
   `resample_field` (hfield), safe zone, tests. Self-contained; no myoassist dependency.
2. **Terrains — rebuild path + contract.** Geom-tile regeneration + the documented recompile
   contract; tests over all 9 tile types.
3. **myoassist — scheduler + wiring.** Curriculum callback, `reset()` regeneration, config-schema
   change, vectorized-env integration. Depends on A4 landing (HfieldManager → terrains) and on the
   RL config-schema change.

##### 6. Open questions for the implementer

- **Difficulty signal:** fixed timestep schedule (simplest, reproducible) vs. performance-gated
  (advance when reward/steadiness clears a threshold — more effective, more complex). Support both?
- **Per-env vs shared difficulty** under vectorization: one global difficulty, or a spread across
  workers (implicit curriculum via a difficulty *distribution*)?
- **Recompile cost:** is a per-stage `MjModel` rebuild acceptable, or must the geom vocabulary be
  approximated on a heightfield so *everything* is hot-swappable? (The latter sacrifices sharp
  stairs/boxes but makes curricula trivial — a real design fork.)
- **Determinism contract** for eval: exact terrain from `(difficulty, seed)` across machines.
- Relationship to the **velocity-map / target-velocity curriculum** (paper §IV-A-3, and D3): terrain
  difficulty and velocity difficulty are separate axes — decide whether they share one schedule.

##### 7. Acceptance criteria

- A training run advances terrain difficulty per a configured schedule, visible in logs.
- Higher difficulty demonstrably yields harder terrain (measurable: relief, step height, obstacle
  density) — monotone and reproducible from `(difficulty, seed)`.
- Episode-level regeneration works without per-episode recompiles in the common case; any rebuilds
  are stage-bounded and vector-safe.
- Safe zone preserved at every difficulty.
- The docs "Terrain Curriculum" claim becomes true; the RL and terrains pages document it.

**Cross-refs:** A4 (HfieldManager → terrains is the delivery vehicle), D3 (velocity map — adjacent
"episode-varying" axis), paper §IV-A-3 (velocity profiles). D4's private-API coupling should be
resolved as part of exposing the resample/layout functions cleanly.

### D6. Stale docstring on `pyramid_stairs`

`tiles/pyramid_stairs.py:14-16` says inverted pyramids are "deferred to v2". They are implemented
(`:95`, `:99`, `:109`, `:157`).

### D7. Media assets + website hosting → standalone docs repository

**Status:** DECIDED — 2026-07-30, **REVISED**: the website is a **standalone repository**, not a
branch of myoassist. Aggregation mechanism (manual-first) still applies (below).

Exactly one image is tracked in git (`utils/style/CONCRETE.png`). All 37 hero renders
(~200 MB), the mesh tree, and every generated heightfield PNG are gitignored
(`.gitignore:221-234`) and exist only on the author's machine.

**Resolution — standalone docs repository (user, 2026-07-30):** the website (aggregated hub
reflecting all four repos + Jekyll build + **all media/figures**) lives in a **new standalone repo**
(name TBD, e.g. `myoassist-docs`), separate from the myoassist code repo, deploying to Pages
(myoassist.neumove.org) on its own.

- **myoassist sheds the site entirely.** `docs/`, `.github/workflows/pages.yml`, and `CNAME` are
  **removed** from myoassist (executed in the 2026-07-30 clean-slate — see Work log). The current
  docs content is preserved on `dev`/history for migration into the new repo.
- **Each modular repo keeps its own full docs with its code** (myo_sim wiki, assist_sim `docs/`,
  terrains README); the docs repo aggregates them into the hub. (Refines the original "single hub"
  stance: hub is still single, source docs live with their code, the docs repo is the assembly
  point.)
- **Media** = committed static assets in the docs repo; the user drops finals in. No LFS/external.
- **Clone-size — now strictly better.** A separate repo means the myoassist clone *and its `.git`*
  carry none of the site/media. This is exactly the isolation a branch could not give (a branch
  shares the object store), and is why the repo beats the branch. (Supersedes the earlier
  clone-size caveat.)
- **New repo needs:** its own `pages.yml` + `CNAME` (myoassist.neumove.org). **Merge-ordering
  dependency:** the new docs repo must be live *before* the myoassist `refactor` branch merges to
  `main`, so myoassist.neumove.org doesn't go dark in the gap (removing `docs/`+`pages.yml` on
  `refactor` doesn't affect the live site until that merge).

**Aggregation mechanism — RECOMMENDED: manual-first (unchanged).** The *fetching* is
trivial in every option; the real, unavoidable cost is a **transform + curation layer** that
exists regardless of mechanism:
1. **Front matter** — just-the-docs places pages via `title`/`nav_order`/`parent`. Sibling docs
   (myo_sim wiki, assist_sim `docs/`, terrains README) are plain `.md` with none; nav placement is
   an editorial decision a script can't infer.
2. **Link/image rewriting** — sibling relative links and (often gitignored) image paths don't
   resolve on the site and must be rewritten per page.
3. **Curation** — a public hub should NOT mirror everything: myo_sim's `agent-workflow.md`,
   engineering-standards, and contributor wikis are repo-internal, not user-facing. A verbatim
   mirror drags them onto the public site.

Because (3) forces a selection step no matter what, and the doc set is modest and fairly stable
(myo_sim ~9 pages, assist_sim ~13, terrains 1 README), **manual copy** into the docs repo is
the lower-total-effort path: you hand-place front matter and fix links once per page — the same
work a script would have to encode — and get curation for free. **Escalation path** if drift
becomes a burden: a thin **local sync script** (copies a designated file list, injects front
matter from a small manifest, run on demand — not CI). Full **CI-automated build-time clone** is
the last resort, only if the doc set grows large and churns fast. (Submodules: not recommended —
pinning ceremony without solving the transform layer.)

This supersedes the earlier "dedicated website branch" direction — same committed-static-assets and
manual-first aggregation ideas, but the assets, build, and Pages workflow live in a standalone docs
repository, not a branch of myoassist.

### D8. Branch consolidation

Active work is on `velocity-map`, ahead of `main`, with a substantial uncommitted working tree
(velocity jitter, emissive arrow materials, the retuned house-palette `terrain_style.xml`) and
18 untracked paths including the entire `ndseg_*` config family. Merge to `main` before the docs
site cites file paths from this repo.

---

## §E — Cross-repo consistency

### E1. Reconcile the device catalog to the paper's 15

**Status:** DECIDED — 2026-07-30. Canonical list of 15 fixed for documentation purposes (below).
Model files / properties / configs for the not-yet-present devices are finalized later and
separately; the docs proceed on this list regardless.

The paper claims 15 assistive-device models. `assist_sim` ships 11 keys, one of which
(`Tutorial_L1`) is an onboarding stub, not a real-world device — so **10** of assist_sim's keys
map to the paper's 15, and **5** are not yet present.

**Canonical 15 (authoritative for docs).** `Tutorial_L1` is deliberately excluded (teaching
device, not a real product; kept only so tutorials don't bias toward one environment).

*Gait-assistive (12) — all `myoassist.terrains`-compatible:*

| # | Paper / Fig. 2 name | Category | assist_sim key | Origin | Status |
|---|---|---|---|---|---|
| 1 | OSL Ankle | Powered prosthetic ankle | `OpenSourceLeg_A_L1` | Open-Source Leg | present |
| 2 | OSL Knee+Ankle | Powered prosthetic knee+ankle | `OpenSourceLeg_KA_L1` | Open-Source Leg | present |
| 3 | K-Foot | Passive prosthetic ankle | `KFoot_L1` | Northeastern | present |
| 4 | NEUankle | Powered prosthetic ankle | *(planned)* | Northeastern | **not finalized** |
| 5 | CR EXO *(unnamed)* | Ankle exoskeleton | *(planned)* | Northeastern | **not finalized** |
| 6 | Humotech EXO-010 | Ankle exoskeleton emulator | `Humotech_L1` | Humotech | present |
| 7 | Dephy ExoBoots | Ankle exoskeleton | `DephyExoBoot_L1` | Dephy | present |
| 8 | OpenExo Ankle Module | Ankle exoskeleton | `OpenExo_L1` | OpenExo (NAU) | present |
| 9 | U. Twente exo *(unnamed)* | Ankle exoskeleton | `UTAnkleExo_L2` | U. Twente | present; name TBD |
| 10 | UCLA exo *(unnamed)* | Ankle exoskeleton (passive) | `Anatomics_L1` | UCLA | present; name TBD ("Anatomics" = UCLA) |
| 11 | Hurotics H-MEDI | Hip exoskeleton | `HMEDI_L1` | Hurotics | present |
| 12 | Hippo | Hip exoskeleton | `Hippo_L1` | Northeastern | present |

*Upper-body & seated-mobility (3):*

| # | Paper / Fig. 2 name | Category | assist_sim key | Origin | Status |
|---|---|---|---|---|---|
| 13 | Auxivo Liftsuit 1.0 | Load-bearing passive back exo | *(planned)* | Auxivo | planned; **NOT terrains-compatible** (special env) |
| 14 | Modular Prosthetic Limb | Bimanual manipulation (paired with `myoArm`) | *(planned)* | JHU/APL | planned; **NOT terrains-compatible** (special env) |
| 15 | Wheelchair Coordination | Seated device interaction | *(planned)* | — | planned; terrains-compatible *(target — needs work, owner TBD)* |

**Docs stance (user, 2026-07-30):** the 5 not-yet-present devices "will likely live in assist_sim
eventually" and are external collaborator models not yet the priority — for documentation we
**pretend they are assist_sim device keys**. The 2 not-finalized gait devices (NEUankle, CR EXO)
render as placeholders in Fig. 2 and are labelled *not finalized* on the site. Auxivo and MPL are
explicitly special environments not compatible with `myoassist.terrains`. The wheelchair
environment **is intended to be `myoassist.terrains`-compatible**, but additional work is required
to make it so and ownership of that work is undetermined — the docs may state the target while
noting it is not yet realized.

**Downstream:** this table is the single source of truth. `assist_sim/README.md` (currently 7),
`assist_sim/docs/available-models.md` (currently 11), and the docs site device catalog must all
be reconciled to it (see B6). The not-yet-present device keys, config YAML, and mesh assets are
out of scope for this pass and tracked separately.

### E2. Two different things are named `myo_sim`

The packaged `myo_sim` 0.2.0 (mjspec-composed, `MyoHub/myo_sim`) and the vendored
`myoassist/myosuite/simhive/myo_sim` 0.1.0 (flat XML) are distinct codebases sharing a name, and
`myoassist` currently loads the latter. This is undocumented everywhere.

A2 removes the collision. Until then, every cross-repo document must disambiguate explicitly.

### E3. Repo name vs. import name for terrains

**Status:** DECIDED — 2026-07-30. Two-form convention (below); prefer the dotted form and use the
underscore form only where Python requires it.

The repository is `myoassist.terrains`; the import is `myoassist_terrains`. It is **not** a
namespace package. `assist_sim`'s prose uses the dotted form throughout
(`assist_sim/assist_sim/utils.py:29,96,163,189,209`; `docs/concepts.md:27,127`;
`docs/troubleshooting.md:105`; `docs/how-to/export-and-load.md:53`; `tests/test_terrain_strip.py:4`).

**Resolution — two forms only, split by "is this a Python identifier?":**

| Form | Where | Why |
|---|---|---|
| **`myoassist.terrains`** (dot) | repo name, GitHub, git URLs, all prose, **and install commands** — `pip install myoassist.terrains`, `pip install git+https://github.com/neumovelab/myoassist.terrains.git` | pip normalizes distribution names per PEP 503 (runs of `. - _` collapse, case-insensitive), so the dotted form resolves to the same package as `myoassist-terrains` / `myoassist_terrains`. The hyphen form never needs to appear. |
| **`myoassist_terrains`** (underscore) | Python code only — `import myoassist_terrains`, `from myoassist_terrains import build_terrain` | a `.` in `import` syntax denotes a submodule; a flat top-level module cannot be named with a dot. Underscore is the only legal module identifier here. |

**Actions:** correct `assist_sim`'s prose at the citations above to use `myoassist_terrains` only
in code/module contexts and `myoassist.terrains` otherwise; present all install commands with the
dotted form. The declared distribution `name` in `pyproject.toml` (currently `myoassist-terrains`)
need not change — normalization makes the docs' dotted install command resolve either way — but it
may be set to `myoassist.terrains` for consistency if desired (optional, no functional effect).

**Deferred alternative (not this pass):** the only way to make `import myoassist.terrains` itself
legal is to restructure `myoassist` into a real namespace package with `terrains` (and, plausibly,
`rl` / `co`) as members. That is the cleanest long-term identity if MyoAssist heads toward a
unified `myoassist.*` namespace, but it is a cross-repo packaging change, not a docs fix. Flagged
as an optional future consolidation; the two-form convention above stands until then.

### E4. MuJoCo version floor

`myoassist` pins `mujoco==3.3.3`; `assist_sim` requires `>=3.3.4` (`MjSpec.delete`);
`myo_sim` declares `>=3.0` but its passive-torso models effectively require `>=3.3.4`;
`myoassist.terrains` requires `>=3.3.3`.

Agree one floor across all four — `>=3.3.4` is the lowest that satisfies everyone — and state it
once, on the docs site's install page. Related: `myo_sim`'s declared `>=3.0` is misleading and
should be raised to match reality.

### E5. Python version floor

`myo_sim` CI tests 3.10 and 3.13; `assist_sim` CI tests 3.10/3.11/3.12; `myoassist.terrains`
requires ≥3.10 and pins 3.12 via `.python-version`; `myoassist` requires ≥3.11 and the docs tell
users to install 3.11 specifically. Pick a supported range and test it.

### E6. Packaging/tooling divergence

`myo_sim` and `myoassist.terrains` use `pyproject.toml` + `uv` + ruff + pre-commit; `assist_sim`
uses `pyproject.toml` + ruff and plans a `uv` migration; `myoassist` still uses `setup.py` with
no linter and no test CI. Converging is optional, but the docs site will document four different
contributor workflows unless something changes.

---

## §F — Outstanding feature work (not refactor)

Feature work that is *not* part of the slim-down/alignment but should ride along on the same
`refactor` branch and land for 1.0. Distinct from §A (which is structural).

### F1. Implement outstanding CO device-specific control features

**Status:** NOTED — 2026-07-30; to be enumerated.

**Where:** `ctrl_optim/ctrl/exo/` (device controllers today: `fourparam_spline_ctrl.py`,
`npoint_spline_ctrl.py`); `ctrl_optim/ctrl/reflex/` (human reflex controller).

**What:** a handful of controller-optimization features for **device-specific control**, developed
across various projects, need to be brought into the CO framework. Today CO ships the generic
device controllers the paper describes — the four-parameter stance spline, the generalized N-point
spline, and the OSL four-state impedance controller — but the additional per-device control
strategies from those projects are not yet integrated.

**To do:** enumerate the specific features and their source projects (list TBD — user to supply),
then scope each: which device(s) it targets, whether it's a new controller class under
`ctrl_optim/ctrl/exo/` or an extension of an existing one, and its config/cost-function surface.
Ties into the device catalog (§E1) — device-specific controllers should map onto the canonical 15.

**Blocked by:** the feature list itself (not yet provided).

---

## Summary

Status legend: **open** (not yet discussed) · **decided** (approach locked) · **planned** (decided, work item scoped) · **deferred** (intentionally not this pass) · **done**.

| ID | Item | Repo | Status | Blocks docs? | Blocked by |
|---|---|---|---|---|---|
| A1 | Remove bundled `models/` | myoassist | **removed** (rewiring pending) | yes | A10, B12 |
| A2 | Vendored myosuite → dependency | myoassist | **removed** (dep-wiring pending) | yes | A5 |
| A3 | CO model resolver → assist_sim | myoassist | open | yes | A10, B12 |
| A4 | HfieldManager → myoassist_terrains | myoassist | open | yes | — (enables D5 ph.3) |
| A5 | CO env unify (retire myoLegStandRandom) | myoassist | **decided** | no | A10 |
| A6 | **22-muscle 2D decision** | myoassist | **decided** | yes | — (see B12) |
| A7 | Packaging + `test_setup.py` | myoassist | open | yes | A1, A2 |
| A8 | Hardcoded tutorial model path | myoassist | open | yes | A10, B12 |
| A9 | Relocate tutorials off the website | myoassist | **planned** | yes | A6/B12 |
| A10 | **Unified RL/CO compose+load pipeline** | myoassist | **decided** | yes | B12 |
| B1 | Version disagreement | assist_sim | **done** (changelog paused) | no | — |
| B2 | CLI docs wrong subcommand | assist_sim | **done** | yes | — |
| B3 | `quickstart.py` broken | assist_sim | **done** (viewer manual) | yes (media) | — |
| B4 | `MSK_MODELS` doesn't exist | assist_sim | **done** | yes | — |
| B5 | `myofullbody` untested/undocumented | assist_sim | open | yes | — |
| B6 | README vs docs device count | assist_sim | **done** | yes | — |
| B7 | `L1`/`L2` convention (define+doc) | assist_sim | **decided** | yes | — (doc-only) |
| B8 | Stale `@mm_refactor` install | assist_sim | **done** | yes | — |
| B9 | Dangling internal references | assist_sim | **done** | no | — |
| B10 | `myo_sim` undeclared dependency | assist_sim | open | yes | decision |
| B11 | Private `_COMPOSED_MODELS` use | assist_sim | open | no | C4 |
| B12 | Implement `myolegs22` reduction util | assist_sim | **planned** | yes | — |
| C1 | Wiki: nonexistent `LEGS26_BASE` | myo_sim | deferred | yes | — |
| C2 | Orphaned `add_keyframes()` | myo_sim | deferred | no | — |
| C3 | PyPI name collision | myo_sim | deferred | **yes** | upstream release |
| C4 | Public composed-model accessor | myo_sim | deferred | no | — |
| C5 | `repository-map.md` stale | myo_sim | deferred | no | — |
| C6 | `myolegs26` has no preview image | myo_sim | deferred | yes (media) | — |
| C7 | Fragile external image hosting | myo_sim | deferred | yes (media) | — |
| D1 | Pillow undeclared | terrains | **done** | **yes** | — |
| D2 | Wrong GitHub org in URLs | terrains | **done** | no | — |
| D3 | Velocity map undocumented | terrains | open | **yes** | — |
| D4 | Private-API coupling | terrains | open | no | — |
| D5 | Terrain curriculum (build it) | terrains + myoassist | **planned** | yes | A4 (phase 3) |
| D6 | Stale `pyramid_stairs` docstring | terrains | **done** | no | — |
| D7 | Media + website → standalone docs repo | myoassist | **decided** | yes | site removed from repo |
| D8 | Branch consolidation | terrains | open | yes | — |
| E1 | **Reconcile the 15 devices** | all | **decided** | yes | — |
| E2 | Two things named `myo_sim` | all | open | yes | A2 |
| E3 | Repo vs import name | all | **decided** | yes | — |
| E4 | MuJoCo version floor | all | open | yes | — |
| E5 | Python version floor | all | open | yes | decision |
| E6 | Packaging/tooling divergence | all | open | no | — |
| F1 | Outstanding CO device-specific control features | myoassist | noted | no | feature list TBD |

**All five gating decisions are resolved.** Documentation work is unblocked.
- **A6** — 22-muscle models: Path A via the §B12 reduction util in assist_sim.
- **E1** — canonical 15 devices fixed (table in §E1).
- **E3** — two-form terrains naming convention (`myoassist.terrains` / `myoassist_terrains`).
- **D5** — build the terrain curriculum; full handoff spec in §D5, delegated.
- **D7** — media as committed static assets on a dedicated website branch (hub + build + media),
  `main` keeps only myoassist docs; tutorials relocate off-site (§A9). Sibling-docs aggregation:
  recommended **manual-first copy** (the transform + curation layer is unavoidable in any option),
  escalate to a local sync script only if drift bites.

Remaining items are implementation/fixes, not decisions — see the status column above.
