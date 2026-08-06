# CO pipeline (`ctrl_optim/`) — Review 1 handoff

CO-specific findings from Review 1 (originally misfiled under the RL handoff). Full context in
`myoassist/review.md`. RL findings are in `RL_PIPELINE_HANDOFF.md`; shared compose-pipeline items
are deferred (see `REVIEW_1.md`).

## Fixed in the Review 1 pass (dead code — verified no live callers, deleted)
These were confirmed dead (defs/exports/comments only, no calls) and removed on the myoassist
`review-1-fixes` branch (commit `f6ed61f`; py_compile + repo grep confirm no dangling refs):
- **`resolve_model_path`** (`optim/optim_utils/resolve_path.py:26` + `__init__.py:17,32` exports) —
  dead since the compose rewrite; built paths against the deleted `models/` tree.
- **`setupTerrain`** (`ctrl/reflex/reflex_interface.py:270`) — unreachable (slope now raises
  `NotImplementedError` first); wrote `hfield_data`/`geom 'terrain'` a composed model lacks.
- **`_CO_DEVICE_MAP["custom"]`** (`ctrl/reflex/reflex_interface.py:39`) — unreachable map entry
  (`--model` choices exclude `custom`; `create_testenv_from_bat` no longer forwards `model_path`).

## CO-1. Latent `NameError` in the cost function — RESOLVED (dead branch, deleted)
- **`ctrl_optim/optim/cost_functions/evaluate_cost.py`** — the `effort_cost`/`pain_cost` `F821`s (×4)
  lived inside `check_optimization_constraints`, which had **no callers** anywhere in the repo. It
  was dead code, so the whole function was deleted (along with the now-orphaned `const_theta_tgt`
  and `scruff_cost` locals). No live cost path ever reached the `NameError`. Verified: repo-wide grep
  (no callers), `py_compile`, and a `run_ctrl_minimal` end-to-end run (Walking 0.280 s, no crash).

## Open questions — RESOLVED with a CO-stack run
- **Keyframe-DOF extension (CO side):** CONFIRMED. On a real composed model `key_qpos` is `(nkey, nq)`
  (MuJoCo pads keyframes to `nq`), so `ReflexEnvV0`'s `init_qpos[:] = key_qpos[0]` stays full-width
  even when the device adds DOFs (checked `nq==39`, `key_qpos[0].shape==(39,)`). Clarifying comment
  added at `ctrl/reflex/reflex_env.py`.
- **`adjust_model_height` site fallback:** CONFIRMED the composed model exposes the `*_touch` sites.
  Hardened anyway: the `A if cond else B` (which silently fell back to `*_touch` even when neither
  set was present, failing obscurely later) is now an explicit guard that raises a clear `KeyError`
  naming both expected site sets (`ctrl/reflex/reflex_interface.py`).

## Also landed in the CO pass
- **`get_plot_data` shadowing (`F811`) fixed:** the no-arg `get_plot_data(self)` (now
  `get_full_plot_data`) shadowed the list-based `get_plot_data(self, data_list)`, so the only caller
  (`reflex_interface.py:394`) would have raised `TypeError`. Renamed the no-arg version;
  `get_full_plot_data` currently has no callers — **CO author to confirm the intended wiring.**
- **`MinDelay = 0.001`** (`reflex_ctrl.py`, `create_delay_struct`): assigned but never used, kept
  intentionally (documented physiological constant, referenced by `reflex_interface.py:274`); carries
  a `# noqa: F841`. Two genuinely-redundant local shorthand copies (`ph_st_csw`, `knee_sw_tgt`) in
  the control function were removed — the live dict state they copied is untouched.
- **`ctrl_optim/` is now fully `ruff check`-clean** (E402 import-order in the 5 entry-point files is
  handled via `[lint.per-file-ignores]` in `ruff.toml`); `ruff format --check` green.

## Note — remaining myoassist ruff lint (RL pass)
The `rl_train/` tree still carries ~35 findings (`F811`/`E402`/`E722` etc.) — style/smell, not bugs,
but they keep the whole-repo `ruff check` gate red until the RL pass triages them. See
`RL_PIPELINE_HANDOFF.md`.

## Pre-existing bugs found while validating (out of CO scope — flag for the owners)
- `ctrl_optim/results/evaluation/__init__.py:8` does `from .eval import main`, but there is no
  `eval.py` in that package → importing `ctrl_optim.results.evaluation` raises `ModuleNotFoundError`.
- `ctrl_optim/optim/train.py:24` does `from optim_utils.tracker import ...` (top-level `optim_utils`),
  which only resolves when run with `ctrl_optim/optim/` on `sys.path` (hence its `E402` ignore).
Neither is caused by the CO pass; both are packaging/import-path issues for the CO author.
