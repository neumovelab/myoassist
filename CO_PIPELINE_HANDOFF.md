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

## CO-1. Latent `NameError` in the cost function (real bug, needs the CO author)
- **`ctrl_optim/optim/cost_functions/evaluate_cost.py:633-636`** (post-`ruff format`) — references `effort_cost` and
  `pain_cost` that are never defined in scope (ruff `F821`, ×4). If that branch executes it raises
  `NameError`. Likely a real bug or a dead branch. **Not** auto-fixable — needs the author to know
  the intended values. **Test needed:** run the cost path that reaches these lines (or confirm it's
  unreachable) and define/repair the names.

## Open questions — need a run with the CO stack
- **Keyframe-DOF extension (CO side):** `ReflexEnvV0` (`ctrl/reflex/reflex_env.py:67-69`) indexes
  `key_qpos[0]`, assuming assist_sim extends the composed keyframes to the device DOFs. Confirm the
  merged `key_qpos` width == `nq`.
- **`adjust_model_height` site fallback** (`reflex_interface.py:~896`): picks `*_touch` sites when
  `*_btm` are absent; `KeyError`s if a composed model exposes neither. Confirm myo_sim always
  provides the touch sites.

## Note — the rest of the myoassist ruff lint (context)
Beyond CO-1's `F821`, the myoassist `ruff check` surface is ~12 `F811` redefinitions + ~34 `E402`
import-order + a few `E722`/`E721` — mostly across `rl_train/` and `ctrl_optim/`. These are style/
smell (not bugs) but will keep the `ruff check` CI gate red until triaged; the Review-1 pass applies
`ruff format` + safe auto-fixes but leaves these author-judgment findings for the RL/CO passes.
