# Review 1 — cross-repo /code-review (2026-08-05)

First formal code-review pass over the MyoAssist 1.0 refactor. Method: one read-only
review agent per repo (parallel), each writing a local `review.md`, then a central
synthesis + cross-repo **seam pass**. Per-repo detail lives in each repo's `review.md`;
this file is the consolidated tracker.

Scopes reviewed:
- **assist_sim** `git diff 6caa7d0..HEAD` — the four upper-body collaboration envs + docs + CI hotfixes.
- **terrains** `git diff main...docs-restructure` — D4 public-API refactor + D3 docs + README→`docs/` split.
- **myoassist** `git diff main...refactor` — the §A composed-architecture refactor (27 commits).

**Headline:** 0 blockers anywhere; the merged/exercised paths are sound. 2 confirmed majors, both on not-yet-live paths. The rest is dead-code / coupling / test-gap / docs cleanup + a cross-repo CI/lint-parity gap.

**Split (2026-08-05):** the myoassist framework findings are separated by pipeline:
- **`RL_PIPELINE_HANDOFF.md`** — RL (`rl_train/`) items → RL author (need a train/eval run).
- **`CO_PIPELINE_HANDOFF.md`** — CO (`ctrl_optim/`) items → CO author. **3 verified-dead CO items are
  fixed in this pass** (resolve_model_path, setupTerrain, `_CO_DEVICE_MAP["custom"]`); the real bug
  (evaluate_cost `NameError`) + runtime open-Qs stay with the author.
- **Deferred — shared compose pipeline** (below): touch `myoassist_utils/compose.py`; need the full
  stack to test; addressed in a later compose/framework pass.

We keep the parts we can verify: **assist_sim** + **terrains** fixes, the 3 safe CO deletions, and
the CI/lint parity + `ruff format` on those repos.

## Deferred — shared compose pipeline (`myoassist_utils/compose.py`)
Neither RL nor CO; need the full RL/CO stack to test. Address in a compose/framework pass.
- Temp-dir leak (`compose.py:216-244`) — register for `atexit` cleanup.
- Terrains-private-import seam (`compose.py:30` `_config_from_dict`) — switch to terrains' new public
  `config_from_dict` **after** terrains `review-1-fixes` merges + installs (currently blocked on that).
- Non-flat-terrain seating approximation (`_seat_dz_by_collision`) — exact for flat, approximate on
  slopes; revisit with D5.

## Verified before landing here
- assist_sim `_add_ground` interpenetration — **CONFIRMED** (floor 4.3 cm above lowest wheel surface; pops 3.8 cm, settles, no blow-up).
- assist_sim wheelchair stale leg sensors — **CONFIRMED** (4 foot/toe sensors kept vs bionic's 0).
- assist_sim `_arm_joint` left mapping — **REFUTED** (38/38 keyframe joints land for both arms).
- assist_sim Auxivo weld torquescale — effectively refuted (earlier drift test matched the original).
- terrains private-symbol rename — **safe** (grep: no external importer of the old `_`-names).

## Findings + wave assignment (status: TODO / DONE / WONTFIX)

### Wave 1 — real bugs + the CI/lint-parity capstone
| # | repo | finding (file:line) | sev | status |
|---|---|---|---|---|
| W1.1 | assist_sim | `_add_ground:315` seats floor at geom center not surface → wheelchair interpenetrates ground; use `_lowest_geom_z(collision_only=True)` | major | **DONE** (45c3758; floor at wheel surface, pen +0.0000, 6mm settle) |
| W1.2 | myoassist | `imitation.json` composes 24-actuator model but declares 26-muscle/no-exo action layout; no guard | major | **HANDOFF** → RL_PIPELINE_HANDOFF.md RL-1, RL-2 |
| W1.3 | assist_sim | wheelchair keeps 4 stale foot/toe sensors (`_freeze_legs_seated`) vs bionic (`_freeze_legs_standing:443`) — add the sensor-delete loop | minor | **DONE** (45c3758; nsensor 4→0) |
| W1.C | terrains + myoassist | **CI/lint parity** — ruff config + pin + CI workflow + format + safe fixes | — | **DONE** — terrains (7dbd3e8, 321e580; ruff green, 64 tests pass); myoassist (f6ed61f: 62-file format + 136 safe fixes + CI; format-check green, **78 author lint findings** incl. the F821 bug left flagged for the RL/CO passes) |

### Wave 2 — hygiene (low-risk, mostly deletions)
| # | repo | finding (file:line) | sev | status |
|---|---|---|---|---|
| W2.1 | myoassist (CO) | `resolve_model_path` dead code (+ dead `get_available_models`/`validate_model_config`) | minor | **DONE** (f6ed61f: deleted + exports dropped; py_compile OK) |
| W2.2 | myoassist (CO) | `setupTerrain` unreachable, manipulates absent hfield | minor | **DONE** (f6ed61f: method deleted) |
| W2.3 | assist_sim | `upper_body.py` imported `myo_sim` at module top — made lazy | minor | **DONE** (0ef5840; import-clean, `build_mpl` myo_sim-free; builders + 10 tests pass) |
| W2.4 | **seam** (terrains + myoassist) | `compose.py:30` imports terrains private `_config_from_dict` | nit | **DONE (terrains half)** (e46e3c2: public `config_from_dict`); myoassist import switch → **compose-deferred** (blocked on terrains merge) |
| W2.5 | myoassist (compose) | `compose.py:216-244` terrain-asset temp-dir leak (latent) | minor | **DEFERRED** → compose pipeline |
| W2.6 | myoassist (RL) | `create_environment` silent fallback to absent `model_path` | nit | **HANDOFF** → RL_PIPELINE_HANDOFF RL-3 |
| W2.7 | myoassist (CO) | dead `_CO_DEVICE_MAP["custom"]` | nit | **DONE** (f6ed61f: entry dropped) |

### Wave 3 — tests + docs + maintainability nits
| # | repo | finding (file:line) | sev | status |
|---|---|---|---|---|
| W3.1 | assist_sim | reload test only checked counts — add stepped-contact on the reloaded bionic | minor | **DONE** (a369544: `test_bionic_reload_holds_object`) |
| W3.2 | assist_sim | `_strip_scene_decor` weaker than `strip_myosuite_scene_spec` (Auxivo export kept scene cameras + orphan meshes) | minor | **DONE** (a369544: delegates to the util) |
| W3.3 | assist_sim | `_reassert_named_geom_contacts` swallowed exceptions silently | minor | **DONE** (a369544: warns). *compile-once perf left (would need to thread the model through export)* |
| W3.4 | assist_sim | nit: hardcoded `_BIONIC_BASELINE` path; pedestal magic constant | nit | **DONE (baseline)** (a369544: env-var, skip when unset). *pedestal-constant cross-comment left (trivial)* |
| W3.5 | terrains | `resolve_tiles` documented row-major but only sorts under randomization | minor | **DONE** (e46e3c2: sorts row-major on both paths; 64 tests pass) |
| W3.6 | terrains | two "public API" declarations differ (`__init__.py` vs `docs/python-api.md`) — one-line note | nit | **DONE** (e46e3c2: `__init__` note) |

### Deferred — need a myoassist/myosuite-enabled run to confirm
- myoassist: keyframe-DOF-extension assumption (assist_sim extends device keyframes to `nq`); `adjust_model_height` touch-site fallback; `imitation.json` obs-key defaults (pre-existing).

## Status — Review 1 COMPLETE + MERGED (2026-08-05)
Landed via one `review-1-fixes` branch per repo, after a green sweep (assist_sim 149 pass +1 skip;
terrains 64 pass; myoassist ruff format-check green + compileall clean):
- **terrains** `review-1-fixes` → `main` (bundled D3/D4 + docs split + ruff/CI parity + fixes).
- **assist_sim** `review-1-fixes` → `main` (Review-1 upper-body fixes + export hardening).
- **myoassist** `review-1-fixes` → `refactor` (PR #21; ruff parity + 3 CO dead-code deletions).
  `refactor` → `main` remains a later step.

**Follow-ups (owned elsewhere):** the **RL pass** (`RL_PIPELINE_HANDOFF.md` — `rl_train/`'s ~35
ruff-check findings remain), and the **deferred compose items** (above). These docs live on
`refactor` for that work.

**CO pass — DONE** (`co-pass` → `refactor`): CO-1 (`evaluate_cost` `NameError`) resolved as dead
code; both runtime open-Qs (keyframe-DOF width, `adjust_model_height` site fallback) confirmed +
hardened; `get_plot_data` `F811` shadowing fixed. `ctrl_optim/` is now fully `ruff check`-clean.
Runtime-validated (`run_ctrl_minimal`: Walking 0.280 s). See `CO_PIPELINE_HANDOFF.md`.
