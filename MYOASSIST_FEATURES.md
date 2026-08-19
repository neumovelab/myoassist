# Bilateral Reflex Control: Optimization-Side Implementation Guide

Rewritten 2026-08-14 to answer three questions:

1. Given the baseline (symmetric) reflex framework, how do you add bilateral reflex control?
2. How does the current `bilat` implementation differ from the earlier asymmetric implementation in
   `C:\Users\calde\Work\ME5374_PROJECT` (`--reflex_mode amp`, optimized on the KAP model)?
3. What is needed to reimplement the ROM constraint parameters, including flagging them for
   optimization, in `neumove-anatomics`?

Everything below is optimization-side: parameter layout, bounds, CLI, config round-trip, and the
device/ROM parameter tail. Model surgery and cost-function redesign are out of scope except where
they change the parameter vector.

Marks: `[verified]` = measured in this tree today. `[read]` = read from code, not executed.

Three code bases are referenced:

| Tag | Path | Asymmetry mode | Layout (2D, no exo) | Model optimized |
|---|---|---|---|---|
| **BASE** | any pre-bilateral checkout of `ctrl_optim` | none (symmetric) | 77 | ANATOMICS / BASELINE |
| **ANAT** | `neumove-anatomics/myoassist` @ `4610d33` | `--reflex_mode bilat` | 128 | ANATOMICS (intact) |
| **ME5374** | `ME5374_PROJECT` @ `c7ab3fd` | `--reflex_mode amp` | 120 (+2 stiffness) | **KAP** (right transtibial, passive foot) |

`ME5374` is the earlier asymmetric implementation. Confirmed by search: the string `bilat` appears
nowhere in `ME5374_PROJECT`, and `amp` appears nowhere in `neumove-anatomics` except as dead
configuration in the two `process_stiffness_results.py` copies. `[verified]`

---

## Implementation decisions (2026-08-17, porting to public myoassist)

These lock the open questions below for the public `myoassist/ctrl_optim` port.
They override the "Questions" and "design questions" sections where they conflict.

1. **D1 hidden L1 penalty: REMOVE ENTIRELY.** The `sum(|params - 1|)` term
   (`walk_cost.py:117` -> `evaluate_cost.py:293` and `:646`, via the empty
   `mus_len_key` making the slice `params[0:]`) is deleted, not merely disabled.
   `muslen_param` comes out of the `evaluateCost` / `_calculate_final_cost`
   signatures so it cannot silently return. Every new bilateral / stiffness / ROM
   number is therefore measured against the documented cost.
   *Future, if a pull-toward-baseline prior is ever wanted:* reinstate it as a
   deliberate, NAMED, REPORTED regularizer, scoped to the reflex+pose block only
   (never the bounded device params, where `1.0` is a ceiling, not a nominal), with
   an explicit weight defaulting to `0` and its own entry in the cost breakdown. Do
   not bring back the whole-vector L1.

2. **ROM is a CONSTRAINT (never optimized), and it lives in the OPTIMIZATION
   config, not the model config.** Drop Section 5.3 entirely (`--optimize_rom`, the
   device-tail, `set_joint_rom.py`, ROM in `param_domains`). ROM limits are a study
   variable swept from the training/optimization config (one CLI arg ->
   `env_dict` -> interface), NOT baked into the assist_sim device yaml -- so a sweep
   is a config edit, not N duplicated model configs, and the limit is not treated as
   an intrinsic device property.
   **Keep the HARD constraint.** MuJoCo's soft range limit (`limited="true"`, via
   solref/solimp) is empirically not strong enough -- the joint overshoots the
   intended ROM under load, which is why ANAT pivoted to a per-step clamp. So do NOT
   rely on a `joint_overrides` range alone: keep the per-step qpos clamp into
   `[lo, hi]` with `qvel` zeroed at the limit. Implement it as a dedicated module in
   the control (exo) section, driven from the optimization config and applied each
   control step.
   NOTE: `--ankle_range` and the six runtime hacks in Section 5.1 exist only in ANAT
   (a grep for `ankle_range` across public myoassist hits this document and nothing
   else), so we build this clean from scratch -- but carry over none of ANAT's bugs:
   do NOT write ROM into a pose-parameter bound (the 5.2 defect), make the
   ground-penetration tolerance an explicit argument rather than coupling it to
   `ankle_range is not None`, and support per-side limits (Anatomics is bilateral),
   not right-only.

3. **Sequencing:** D1 removal (shared prerequisite) -> bilateral `bilat` -> KFoot
   stiffness -> Anatomics ROM.

4. **Amputee `amp` keys are DERIVED from the composed model, not hardcoded.** The
   amputation lives in the assist_sim device layer (e.g. `KFoot/L1config.yaml`
   removes `soleus_r/tibant_r/edl_r/fdl_r`, re-anchors gastroc as a knee flexor,
   attaches the passive `df_/pf_ankle_angle_r` foot; `OpenSourceLeg/A` and `/KA` do
   the powered equivalents). So the per-leg reflex key set is whatever survives in
   the composed model's actuator inventory -- device-agnostic across KFoot / OSL-A /
   OSL-KA -- and `amp` becomes `bilat` + model-derived per-leg key filtering, layered
   on after the bilateral split. There is no separate amputee MSK and no ME5374
   prune list to port.

---

## 1. Recipe: baseline framework to bilateral reflex control

Six edits. The controller math does not change at all. Only the parameter vector, its split, and the
optimizer's view of it change.

### Step 1. Accept a doubled reflex block in the control law

`ctrl/reflex/reflex_ctrl.py`, `MyoLocoCtrl.set_control_params`:

```python
def set_control_params(self, params):
    # Symmetric: n_par mirrored to both legs.
    # Asymmetric (bilat): n_par*2 split as [r_leg | l_leg].
    if len(params) == self.n_par:
        self.set_control_params_leg('r_leg', params)
        self.set_control_params_leg('l_leg', params)
    elif len(params) == self.n_par * 2:
        self.set_control_params_leg('r_leg', params[:self.n_par])
        self.set_control_params_leg('l_leg', params[self.n_par:])
    else:
        raise Exception(f"Wrong params: {len(params)} vs {self.n_par} "
                        f"(or {self.n_par * 2} for asymmetric)")
```

`set_control_params_leg` already writes `self.cp[s_leg]`, and every module already reads
`self.cp[s_leg]`. That is the whole reason this change is cheap: the reflex law was already per-leg,
only the parameter feed was mirrored.

`n_par` is 51 in 2D and 63 in 3D (length of `MyoLocoCtrl.cp_keys` for the mode).

### Step 2. Make the interface layout arithmetic mode-driven

`ctrl/reflex/reflex_interface.py`, `myoLeg_reflex.__init__`. Replace the hard-coded `77` / `97`:

```python
self.reflex_mode = reflex_mode
self.is_bilat_mode = (reflex_mode == 'bilat')

reflex_n_par  = 51 if mode == '2D' else 63
reflex_block  = reflex_n_par * 2 if self.is_bilat_mode else reflex_n_par
pose_block    = 26 if mode == '2D' else 34
base_params   = reflex_block + pose_block
spline_params = 0 if not exo_bool else (4 if use_4param_spline else n_points * 2)

expected_params    = base_params + spline_params
self._reflex_block = reflex_block
self._pose_block   = pose_block
```

Then every slice in `__init__` and `reset` derives from `_reflex_block` / `_pose_block`:

```python
# __init__
pose_end = len(self.CONTROL_PARAM) - spline_params if spline_params > 0 else len(self.CONTROL_PARAM)
self.update_init_pose_param_cmaes(self.CONTROL_PARAM[self._reflex_block:pose_end])

# reset
expected_params = self._reflex_block + self._pose_block + spline_params
if len(self.CONTROL_PARAM) != expected_params:
    raise ValueError(f"Expected {expected_params} parameters, got {len(self.CONTROL_PARAM)}")
reflex_params = self.CONTROL_PARAM[0:self._reflex_block]
pose_params   = self.CONTROL_PARAM[self._reflex_block:self._reflex_block + self._pose_block]
```

The pose block and the exo tail are untouched by bilateral mode. Only the reflex block doubles.

### Step 3. Double the reflex bounds block

`optim/optim_utils/bounds.py`, in `get_bounds`, after the per-model bound list is built:

```python
is_bilat = bool(input_args and getattr(input_args, 'reflex_mode', None) == 'bilat')
if is_bilat and musc_model in ('22', '26'):
    reflex_n_par = 51 if control_mode == '2D' else 63
    bound_start, bound_end = bounds
    bound_start = bound_start[:reflex_n_par] + bound_start   # [r_reflex | (l_reflex + pose + exo)]
    bound_end   = bound_end[:reflex_n_par]   + bound_end
    bounds = [bound_start, bound_end]
```

Prepending a copy of the first `n_par` entries is the minimal change: the tail of the original list
(left reflex, pose, exo) keeps its position relative to the end, so nothing downstream shifts.

`train.py` derives everything else from the bounds length, so no other optimizer change is needed:

```python
bound_start, bound_end = get_bounds(musc_model, control_mode, ankle_range=ankle_range)
param_num  = len(bound_start)          # 128 in 2D bilateral, no exo
params_0   = np.ones(param_num)
sigma_mult = np.ones(param_num) * input_args.sigma_gain
opts.set('CMA_stds', sigma_mult)
opts.set('bounds', [bound_start, bound_end])
```

Note the coupling: `bounds.py` reads `input_args` through a module global that `train.py` sets
(`bounds_mod.input_args = input_args`). Any new entry point must set it too.

### Step 4. CLI and env_dict passthrough

`optim/config/arg_parser.py`:

```python
optim_group.add_argument("--reflex_mode", required=False,
                         choices=["uni", "ind", "bilat"],
                         help="uni|ind are legacy 80-muscle modes; bilat = asymmetric L/R blocks")
```

`optim/config/environment.py` (the copy `train.py` actually imports through `config/__init__.py`):

```python
reflex_mode = args.reflex_mode if args.reflex_mode is not None else ('uni' if args.musc_model == 'leg_80' else None)
env_dict['reflex_mode'] = reflex_mode
```

`optim/cost_functions/walk_cost.py` must forward it into the constructor, or every worker silently
runs symmetric:

```python
Myo_env = myoLeg_reflex(..., reflex_mode=env_dict.get('reflex_mode', None))
```

### Step 5. Parameter-file continuation

`train.py`, the `--param_path` branch, needs the same arithmetic so a bilateral run can be continued
or extended with exo parameters:

```python
reflex_n_par = 51 if control_mode == '2D' else 63
pose_block   = 26 if control_mode == '2D' else 34
is_bilat     = (getattr(input_args, 'reflex_mode', None) == 'bilat')
reflex_block = reflex_n_par * 2 if is_bilat else reflex_n_par
base_params  = reflex_block + pose_block
```

### Step 6. Config round-trip for evaluation

`optim/optim_utils/config_parser.py` must parse `--reflex_mode` out of the archived `.bat` and pass
it to `myoLeg_reflex`, otherwise a 128-value file loads into a 77-value env and the length check
fires:

```python
'--reflex_mode': (r'--reflex_mode\s+(\w+)', str),
...
TestEnv = myoLeg_reflex(..., reflex_mode=config.get('reflex_mode', None))
```

### The invariant to test

```
len(get_bounds(...)[0]) == myoLeg_reflex(...)._expected_params == len(param_file)
```

Nothing in ANAT enforces this. The failure mode is silent: `__init__` prints
`"Wrong number of params, Defaulting to N"` and replaces the vector with `np.ones(N)`
(`reflex_interface.py:143-145`), so a mismatched run optimizes from an all-ones start with no error.
Add the assertion in `walk_cost.func_Walk_FitCost` before the env is built.

### Verified end state (ANAT)

Loading the committed 128-value file gives `_reflex_block = 102`, `_pose_block = 26`, `dt = 0.01`,
`nu = 27`, and `cp['r_leg']['theta_tgt'] = 0.16524` versus `cp['l_leg']['theta_tgt'] = 0.19839`, so
the two legs really are independent. It walks 2 s with 7 footsteps and no termination. `[verified]`
The 20 s optimized solution (`anatomics_bilat_0713_1419`, `-kine`) reaches 1.2424 m/s but terminates
at 7.08 s of 20 s. `[verified]`

---

## 2. Layout reference

| Block | BASE 2D | ANAT `bilat` 2D | ME5374 sym 2D | ME5374 `amp` 2D |
|---|---|---|---|---|
| reflex | 51 | 51 + 51 = 102 | 51 | 43 (right, pruned) + 51 (left) = 94 |
| pose | 26 | 26 | **27** | 26 (own key order) |
| exo tail | 0 / 4 / 2n | 0 / 4 / 2n | 0 / 4 / 2n | 0 / 4 / 2n |
| device tail | none | none | 2 if `--optimize_stiffness` | 2 if `--optimize_stiffness` |
| **base total** | **77** | **128** | **78** | **120** |

ME5374's symmetric pose block is 27, not 26: the bound list carries an extra `mtp_angle_r` entry that
`update_init_pose_param_cmaes` strips at runtime into `self.EXTRA_POSE_PARAMS['mtp_angle_r']` and
then ignores. `[verified: 78-entry and 120-entry bound lists]` Consequence: **ME5374 parameter files
are not loadable in ANAT and vice versa.** A 78-value symmetric ME5374 file must have index 56
(`mtp_angle_r`) deleted to become a 77-value ANAT file.

Pose block contents (2D, both repos): 8 joint/velocity parameters then 18 initial activations.
ANAT order is `pose_map` = `pelvis_tilt, hip_flexion_r, hip_flexion_l, knee_angle_r, knee_angle_l,
ankle_angle_r, ankle_angle_l, vel_pelvis_tx`. ME5374 `amp` order is `amp_pose_joint_keys_2d` =
`pelvis_tilt, hip_flexion_r, hip_flexion_l, knee_angle_r, knee_angle_l, ankle_angle_l, mtp_angle_l,
vel_pelvis_tx`: the amputated side contributes no ankle or MTP pose parameter, and the left MTP
does.

---

## 3. Diff: ANAT `bilat` versus ME5374 `amp`

Both are the same idea (one reflex parameter block per leg) with different bookkeeping. The
differences that matter to the optimizer:

### 3.1 Block width: full mirror versus pruned right block

* **ANAT**: both blocks are the full 51 keys. `set_control_params` splits at 51.
* **ME5374**: the amputated (right) block is 43 keys. `MyoLocoCtrlAmp.cp_keys_r_leg` drops the 8 keys
  that depend on muscles the amputee model does not have:
  `ankle_tgt, mtp_tgt, 1_SOL_FG, 1_FDL_FG, 5_TA_PG, 5_TA_SOL_FG, 5_EDL_PG, 5_EDL_FDL_FG`.
  At `reset` the 43 values are expanded back to 51 by
  `reflex_param_utils.expand_right_amp_params_to_full` (zero-filling the dropped keys), concatenated
  with the left 51, and handed to the **standard** `MyoLocoCtrl.set_control_params` as 102 values.

So ME5374 already contains the ANAT split, one layer down. The visible layout is 94, the internal
layout is 102.

Two consequences worth carrying forward:

* Pruning is the right default for an amputee run. It removes 8 dimensions from CMA-ES per leg that
  cannot affect the simulation, which is 8 fewer directions in a 120-dimensional search.
* `MyoLocoCtrlAmp` is dead weight. It is a full 41 KB copy of the controller, but the interface never
  instantiates it: `reflex_interface.py:253-254` says "In amp mode, use regular controller -
  interface will filter out missing actuators". Only its `cp_keys_r_leg` list is used, through
  `reflex_param_utils`. If the amp path is revived, port the key list and delete the copy.

### 3.2 Bounds: generated versus hand written

* **ANAT**: `bounds.py` duplicates the first `n_par` entries programmatically (4 lines). Cheap, and
  it cannot drift out of sync with itself.
* **ME5374**: a hand-written 120-entry list for the amp case, separate from the 78-entry symmetric
  list. `[verified: 120 and 78 entries]` Both lists carry label drift. In the ME5374 amp list the
  left block ends `..., 10_GLU_PG_l, 10_VAS_PG_l, 1_FDL_FG_l`, so `1_FDL_FG` sits at position 50 of
  the left block instead of position 15, and every label from 15 onward is off by one. In ANAT the
  legacy 2D list has only 49 reflex comments for 51 keys (`1_FDL_FG` and one `5_EDL_*` are missing)
  and two spurious `mtp_angle_*` pose entries, so indices 49 and 50 (`10_GLU_PG`, `10_VAS_PG`)
  receive `[-inf, inf]` instead of `[0, inf]` and are free to go negative.

Neither approach is correct. Generate bounds from `cp_keys` and `pose_map` with a per-key sign rule.
That is the one place where a rewrite pays for itself twice, since it also removes the ROM index
arithmetic in Section 5.

### 3.3 Bootstrapping symmetric to asymmetric

**ME5374 has this and ANAT does not.** `reflex_param_utils.convert_sym51_to_amp_blocks` plus
`myoLeg_reflex._upgrade_legacy_amp_params` accept a legacy 77-value symmetric file and rebuild it as
a 120-value amp file: the symmetric 51 are copied into both blocks (right by key name, so pruned
keys are skipped), the 26 pose values are preserved, and any exo/device tail is re-appended. It fires
automatically when the file length equals the legacy expectation
(`reflex_interface.py:156-157, 357-358`).

This is the fastest way to start a bilateral run: seed both legs from a converged symmetric solution
instead of from `np.ones`. Port it to ANAT as
`upgrade_symmetric_to_bilateral(params, layout) -> params` and call it from the `--param_path`
branch of `train.py`, not only from the interface. Then `--param_path <symmetric_run>` with
`--reflex_mode bilat` just works.

### 3.4 Device parameter tail and CCO

**ME5374 only.** This is the machinery the ROM work in Section 5 should reuse.

* `--optimize_stiffness` appends exactly 2 parameters and 2 `[0, 1]` bounds
  (`bounds.py:272, 306`), tracked as `self.stiffness_param_count` so the reflex/pose/exo/device
  boundaries stay computable (`reflex_interface.py:126-150`).
* At `reset`, `apply_stiffness(self.env.sim, pf_norm, df_norm)`
  (`ctrl/prosthetic/set_ankle_stiffness.py`) denormalizes and writes
  `model.jnt_stiffness[joint_id]` for `pf_ankle_angle_r` (PF range 30 to 305 N.m/rad) and
  `df_ankle_angle_r` (DF range 100 to 1050 N.m/rad). Runtime write, no recompile.
* `optim_utils/param_domains.py:get_parameter_domains` splits the vector into
  `human_indices` and `device_indices` from the flags, with the amp base hard-coded at 120.
* `--optim_mode cco` then runs two CMA-ES instances over those two index sets, with a phase-1
  feasibility threshold (`--cco_phase1_threshold`, `--cco_phase1_consec`), separate populations
  (`--cco_human_pop`, `--cco_device_pop`), separate objectives (`--cco_human_cost eff_sym`,
  `--cco_device_cost sl`) and a Pareto archive (`optim_utils/cco_archive.py`).

`stiffness_plan.md` describes an earlier mjSpec design that compiled a temporary XML per candidate.
The shipped code does **not** do that: `set_ankle_stiffness.py` writes the live model instead. Prefer
the live-model approach for ROM as well. It has no temp files, no compile cost, and no cleanup.

### 3.5 Cost functions and gates

ME5374 added objectives that exist because of the asymmetric model, and they are the ones ANAT is
missing for amputee work:

| Flag | Objective |
|---|---|
| `-sl` / `-eff_sl` | socket load, from the `kap_force_sensor` in the KAP model |
| `-eff_sym` | effort plus an explicit symmetry term |
| `-eff_wsj` | effort plus weighted symmetry and jerk |
| `-kine_sym` | kinematics plus symmetry |

ANAT still gates `Kine` and `Velocity` on `sym_cost < tgt_sym` with `sym_cost` computed as a direct
left-versus-right knee and ankle comparison. For an intact bilateral run that gate is a useful
regularizer. For an amputee run it is an obstacle, which is why ME5374 moved symmetry into the
objective (`eff_sym`, `kine_sym`) instead of the gate.

### 3.6 What ME5374 did not have

`--ankle_range` and the arm-swing / lumbar gates do not exist in ME5374. `[verified: full
`add_argument` list]` They are ANAT-only. Conversely `--optimize_stiffness`, `--optim_mode cco` and
the `cco_*` flags are ME5374-only.

### 3.7 The model used for optimization

ME5374's asymmetric runs used `--model kap`, `models/22muscle_2D/myoLeg22_2D_KAP.xml`. `[verified]`
Relevant properties, because they set the shape of the amputee parameter vector:

* Right side has 7 muscles (`glutmax, hamstrings, bifemsh, iliopsoas, rectfem, vasti, gastroc`).
  `soleus_r, tibant_r, fdl_r, edl_r` are absent. Left side is the full 11.
* The prosthetic foot is **passive**, with two spring joints and no motor:
  `df_ankle_angle_r` range `0 .. 0.5236` rad, `stiffness=500`, `damping=5`;
  `pf_ankle_angle_r` range `-0.5236 .. 0` rad, `stiffness=50`, `damping=5`.
  Those two stiffnesses are what `--optimize_stiffness` tunes.
* `_get_ankle_angle('r')` returns `df + pf`, which is the "combines df+pf" comment that the dead
  ANAT call sites still refer to.
* Sensors: `r_ankle_df_sensor` and `r_ankle_pf_sensor` replace `r_ankle_sensor`, and
  `kap_force_sensor` provides socket load.
* `talus_r` does not exist, so the contralateral placement sensing falls back to body
  `kap_tibial_pylon` (`reflex_interface.py:656-666`).
* The model carries `r_shoulder`, `l_shoulder`, `r_elbow_flex`, `l_elbow_flex` and
  `lumbar_rotation` actuators, so it is on the same lineage as ANATOMICS.

Note the ANAT amputee model `myoLeg22_2D_OSL_A.xml` is a different device: one **powered**
`osl_ankle_angle_r` joint with a `gear=58.4`, `ctrlrange=+/-2.88` motor, and no arm or lumbar
actuators. `[verified]` A KAP-style passive-stiffness parameter tail does not apply to it, and a
KAP parameter file will not transfer.

### 3.8 Port list, ANAT priority order

1. `upgrade_symmetric_to_bilateral` (Section 3.3). Immediate value for the intact bilateral runs.
2. Programmatic bounds generation (Section 3.2). Fixes the two negative-gain bounds as a side effect.
3. Pruned per-leg key lists (Section 3.1), as a `LegSpec`-style inventory rather than a second
   controller class.
4. Device parameter tail plus `param_domains` (Section 3.4). This is the mechanism Section 5 needs.
5. Symmetry as an objective rather than a gate (Section 3.5).
6. CCO, only if two-objective human/device competition is still wanted.

---

## 4. Known optimization-side defects in ANAT

Only the ones that change what CMA-ES sees. Fix 1 before any new tuning run.

1. **A hidden regularizer is applied to the whole parameter vector.**
   `walk_cost.py:128` is `muslen_param = params[-1 * len(Myo_env.mus_len_key):]`. `mus_len_key` is
   `[]` (`reflex_interface.py:81`), so the slice is `params[0:]`, the entire vector.
   `evaluate_cost.py:289-290` then adds `sum(|params - 1|)` to the minimized cost, and
   `create_cost_dictionary` never reports it. Every run in this repo carries a large undocumented
   pull toward the all-ones start point. `[read, arithmetic verified]`
   Fix: `muslen_param = params[-n:] if n else np.array([])`.
2. **Two reflex gains have sign-free bounds** (Section 3.2): indices 49 and 50, `10_GLU_PG` and
   `10_VAS_PG`, get `[-inf, inf]`.
3. **`--ankle_range` is written into a pose-parameter bound as if it were an angle** (Section 5.2).
4. **No layout invariant check**, so a wrong-length parameter file silently becomes `np.ones(N)`.
5. **`theta_tgt` for the trunk cost is read from `r_leg` only** (`reflex_interface.py:647`, comment
   "any leg will do, since they are the same"). Under bilateral control they are not.
6. **Delay buffers are seeded before the candidate gains are installed**
   (`reflex_interface.py:356-359`): `reset_delay_buffers` reads `self.cp` for `alpha_0, C_d, C_v`,
   but `ReflexCtrl.reset(reflex_params)` runs after it. Harmless at `--delayed 0` (buffer length 1),
   wrong for the first 25 ms at `--delayed 1`. Swap the two calls.
7. **Toe muscles are outside the effort and EMG costs**: `evaluate_cost.py:197, 199, 538, 540` filter
   on `'TOFL'` and `'TOEX'` while `muscles_dict` uses `'FDL'` and `'EDL'`.

---

## 5. ROM constraint parameters in `neumove-anatomics`

### 5.1 What exists now

`--ankle_range MIN MAX` (radians, `arg_parser.py:31-32`) is a **fixed** constraint, not a parameter.
It is the independent variable of the ROM study: the run directories `00`, `1010`, `1015`, `1510`,
`1515`, `nolim` under `results/sample_run_output` are plantarflexion/dorsiflexion limit pairs in
degrees. The ANATOMICS default is `range="-1.134 0.349"`, that is 65 deg plantarflexion and 20 deg
dorsiflexion, so `range[0]` is the plantarflexion limit (negative) and `range[1]` is the
dorsiflexion limit (positive). `[verified]` `--ankle_range -0.174533 0.261799` is the `1015`
condition: 10 deg PF, 15 deg DF.

It touches six places, all hard-coded to the right side:

| Location | Effect |
|---|---|
| `reflex_interface.py:193-200` | writes `model.joint('ankle_angle_r').range[0:2]` at construction, then `env.forward()` |
| `reflex_interface.py:386-395` | after every `run_reflex_step`, clamps `qpos` and zeroes `qvel` at the limit |
| `reflex_interface.py:579-587` | same clamp in `run_reflex_step_Cost`, the loop optimization uses |
| `reflex_interface.py:886-892` | clamps the CMA-ES initial pose for `ankle_angle_r` |
| `reflex_interface.py:970-974` | relaxes the ground-penetration tolerance from -0.02 to -0.035 m whenever `ankle_range is not None` |
| `bounds.py:49-61` | overwrites the CMA-ES bound at index `reflex_block + 5` |

Plumbing: `arg_parser.py:31` to `environment.py:70-72` (`env_dict['ankle_range']`) to
`walk_cost.py:86` to the constructor, and `config_parser.py:69-74` recovers it from an archived
`.bat` for evaluation.

Two things to know before reusing any of it:

* The joint-range write and the per-step clamp are redundant but not equivalent. MuJoCo already
  enforces `range` with a constraint. The explicit clamp additionally **zeroes the velocity** at the
  limit, which is a hard stop with no bounce and no restitution. That is a modelling decision baked
  into the controller loop, not a joint property. If you want a compliant end stop instead, remove
  the clamp and give the joint stiffness and damping, which is what the KAP model does.
* `check_pose_validity` changing its ground tolerance based on `ankle_range is not None` means the
  ROM flag also changes the pose-rejection rate, so ROM conditions are not perfectly comparable to
  `nolim`. Make the tolerance an explicit argument.

### 5.2 The bounds defect to fix first

`bounds.py:49-61` writes the radian limits into the bound for parameter `reflex_block + 5`. That
parameter is the **initial-pose multiplier** for `ankle_angle_r`, mapped by
`x * 5 deg - 5 deg` in `update_init_pose_param_cmaes`. It is not an angle. With
`--ankle_range -0.174533 0.261799` the optimizer can only place the initial right ankle between
-5.87 deg and -3.69 deg, which is a silent 20-fold narrowing of that search dimension. The runtime
clamp still enforces the real ROM, so the constraint itself is correct and only the search is
crippled.

Fix, keeping the intent (do not let the initial pose start outside the ROM):

```python
# bounds.py, replacing the ankle_range block
if ankle_range is not None:
    idx = layout.pose_index('ankle_angle_r')          # reflex_block + 5, from the layout object
    lo_ang, hi_ang = ankle_range                      # radians
    # invert the pose mapping x * 5 deg - 5 deg
    to_param = lambda ang: (ang + np.deg2rad(5)) / np.deg2rad(5)
    bound_start[idx] = to_param(lo_ang)
    bound_end[idx]   = to_param(hi_ang)
```

With `1015` that gives roughly `[-1.0, 4.0]` instead of `[-0.1745, 0.2618]`, which is the intended
meaning.

### 5.3 Making ROM an optimized parameter

Follow the ME5374 device-tail pattern exactly (Section 3.4), with the ROM values written into the
live model at `reset` rather than into a recompiled XML.

**CLI** (`arg_parser.py`, new group "ROM Configuration"):

```python
rom_group.add_argument("--rom_joints", nargs='+', default=['ankle_angle_r'],
                       help="Joints whose range is constrained or optimized")
rom_group.add_argument("--ankle_range", type=float, nargs=2, default=None,
                       help="Fixed ROM [min max] in radians. Ignored when --optimize_rom is set")
rom_group.add_argument("--optimize_rom", action="store_true",
                       help="Append 2 normalized ROM parameters (PF limit, DF limit) per rom joint")
rom_group.add_argument("--rom_pf_bounds", type=float, nargs=2, default=[0.035, 0.524],
                       help="Absolute PF-limit denormalization range in radians (2 to 30 deg)")
rom_group.add_argument("--rom_df_bounds", type=float, nargs=2, default=[0.035, 0.349],
                       help="Absolute DF-limit denormalization range in radians (2 to 20 deg)")
rom_group.add_argument("--fixed_rom", action="store_true",
                       help="Keep ROM parameters at their initial values during optimization")
```

`--fixed_rom` mirrors the existing `--fixed_exo` so a ROM value can be carried in the vector without
being searched.

**Denormalizer** (`ctrl_optim/ctrl/prosthetic/set_joint_rom.py`, the ROM analogue of
`set_ankle_stiffness.py`):

```python
import numpy as np

def _denorm(x, lo, hi):
    return lo + float(np.clip(x, 0.0, 1.0)) * (hi - lo)

def apply_rom(sim, p_pf, p_df, joint='ankle_angle_r',
              pf_bounds=(0.035, 0.524), df_bounds=(0.035, 0.349)):
    """Write a normalized ROM pair into the live model. Returns (pf_rad, df_rad) applied."""
    jid = sim.model.joint(joint).id                      # KeyError if absent, let it raise
    xml_lo, xml_hi = sim.model.jnt_range[jid].copy()     # never widen past the XML
    pf = min(_denorm(p_pf, *pf_bounds), abs(xml_lo))
    df = min(_denorm(p_df, *df_bounds), xml_hi)
    sim.model.jnt_range[jid] = (-pf, df)
    return pf, df
```

Clamping to the XML range matters: widening a joint past its modelled limit lets the optimizer buy
performance with geometry the device does not have.

**Interface** (`reflex_interface.py`):

```python
# __init__
self.rom_param_count = (2 * len(rom_joints)) if optimize_rom else 0
expected_params = base_params + spline_params + self.rom_param_count

# reset, BEFORE set_init_pose / adjust_initial_pose_cmaes / adjust_model_height
if self.rom_param_count:
    tail = self.CONTROL_PARAM[-self.rom_param_count:]
    self.applied_rom = {}
    for k, joint in enumerate(self.rom_joints):
        pf, df = apply_rom(self.env.sim, tail[2*k], tail[2*k+1], joint=joint,
                           pf_bounds=self.rom_pf_bounds, df_bounds=self.rom_df_bounds)
        self.applied_rom[joint] = (-pf, df)
        if joint == 'ankle_angle_r':
            self.ankle_range = (-pf, df)     # reuse the existing clamp path
    self.env.forward()
```

Ordering is not optional. The ROM must be applied before `adjust_initial_pose_cmaes`, because that
function clamps the initial pose to `self.ankle_range`, and before `adjust_model_height`, because a
changed ankle limit changes which foot site is lowest. Setting `self.ankle_range` from the applied
values makes the existing per-step clamp, the pose clamp and the ground tolerance all follow the
optimized ROM with no further edits.

**Bounds** (`bounds.py`, after the exo tail):

```python
if getattr(input_args, 'optimize_rom', False):
    for _ in range(2 * len(input_args.rom_joints)):
        bound_vect.append([0, 1])
```

Two `[0, 1]` entries per joint, exactly like the stiffness tail. Keep the tail order
`[... exo | rom]` and document it, because `reset` slices the ROM tail from the end.

**Initial values** (`train.py`, next to the existing exo initialization): start ROM parameters at the
normalized position of the current study condition rather than at 1.0, since `np.ones` maps to the
widest allowed ROM:

```python
if input_args.optimize_rom:
    pf0 = (0.1745 - pf_lo) / (pf_hi - pf_lo)      # 10 deg PF
    df0 = (0.2618 - df_lo) / (df_hi - df_lo)      # 15 deg DF
    params_0[-2], params_0[-1] = pf0, df0
```

Also add `--optimize_rom`, `--fixed_rom` and the four ROM values to
`config_parser.parse_bat_config` so evaluation reproduces the optimized ROM, and record the applied
absolute values in the `*_Cost.txt` output. A run whose ROM is a free parameter is unreadable if only
the normalized value is saved.

**CCO** (only if the two-optimizer scheme is ported): in `param_domains.get_parameter_domains`, treat
the ROM tail exactly as the stiffness tail, so `device_indices` covers it and `human_indices` covers
reflex plus pose.

### 5.4 Design questions the code cannot answer

* **Is ROM a device parameter or a study condition?** As a free parameter it will run to the widest
  allowed value in almost any effort or kinematics objective, since a restricted ankle can only hurt.
  It becomes meaningful only against a competing objective that rewards restriction, which is what
  CCO with a device cost provides, or against a fixed device penalty. Otherwise keep it a swept
  condition and use Section 5.2 only.
* **Symmetric or per-side?** `--ankle_range` is right-side only today. If both ankles get a device,
  the flag needs a side, and the pose bound in Section 5.2 needs to apply to both pose indices.
* **Hard stop or compliant stop?** The current clamp zeroes velocity at the limit. Optimizing a ROM
  limit under a hard stop optimizes an impulsive contact, which is likely to be exploited. A
  stiffness and damping end stop, as in the KAP df/pf joints, is the physically defensible version,
  and it makes the ROM and stiffness parameters one coherent device model.

---

## 6. Verification checklist

| Check | Expected |
|---|---|
| `len(get_bounds(...)[0])` versus interface `expected_params` | equal, for {2D,3D} x {symmetric, bilateral} x {no exo, 4-param, n-point} x {ROM off, ROM on} |
| Bilateral split | vector `[A(51) \| B(51) \| pose]` gives `cp['r_leg']` from A and `cp['l_leg']` from B; a 51-value block mirrors |
| `upgrade_symmetric_to_bilateral` | a converged 77-value file becomes a 128-value file whose first two blocks are identical, and the env reproduces the symmetric result |
| Positive-scale reflex gains | every bound is `[0, inf]` |
| ROM applied | `model.jnt_range[ankle_angle_r]` equals the denormalized pair, never wider than the XML default `[-1.134, 0.349]` |
| ROM ordering | with ROM on, the initial pose lies inside the applied range and `adjust_model_height` still seats the model |
| Cost with no muscle length parameters | no hidden `sum(\|x - 1\|)` term |
| ANATOMICS bilateral smoke test | committed `BestLast`, 2 s, 7 footsteps, no termination `[verified today]` |

---

## 7. Questions

1. **ROM as parameter or condition?** Section 5.4. If there is no competing device objective, I would
   keep `--ankle_range` as a swept condition, fix the bound defect (5.2), and not add
   `--optimize_rom` at all. Do you want the parameter path anyway, for a CCO-style run?
2. **Which amputee device is the target in `neumove-anatomics`?** The repo ships the powered
   `OSL_A` model, while all the earlier asymmetric optimization was done on the passive-stiffness
   `KAP` model. Should the KAP model be brought over so the ME5374 parameter files and the
   stiffness tail remain usable, or is OSL_A the new target with a new device controller?
3. **Do you want the ME5374 pruned-block layout (43+51) for amputee runs, or full 51+51 with the
   inert keys frozen?** Pruning removes 8 useless CMA-ES dimensions. Freezing keeps every result
   file the same length across intact and amputee conditions, which makes the comparison plots
   trivial. I lean toward freezing, with the pruning as a `layout.freeze_keys` list.
4. **Is the hidden `sum(|params - 1|)` term (Section 4, item 1) present in the runs behind the
   current figures?** If so, the ROM comparison results were produced under a cost that is not the
   documented cost, and the sweep may need a re-run before anything is written up.
5. **Should `--reflex_mode bilat` be renamed?** `uni` and `ind` are 80-muscle legacy modes and
   `bilat` is unrelated to them, which makes the flag hard to read. `--reflex_layout
   {shared,bilateral,amputee}` with deprecated aliases would be clearer, at the cost of touching
   every archived `.bat`.

-------------------------
# Asymmetric Ankle Stiffness Optimization

Reference documentation for the as-built implementation in `ME5374_PROJECT`, plus
instructions to reimplement it in a clean control-optimization framework.

**Scope.** This document covers the stiffness optimization only: two normalized
parameters that set the passive plantarflexion and dorsiflexion springs of the
prosthetic ankle, optimized inside a single CMA-ES loop together with the reflex
controller. It does not cover the competitive co-optimization (CCO) path. All CCO
machinery (`param_domains.py`, `cco_archive.py`, `--optim_mode cco`,
`--cco_human_cost`, `--cco_device_cost`) is out of scope and must be ignored when
you follow the instructions in Part II.

---

## Part I. What the mechanism is

### 1.1 Two joints in series, one per direction

The prosthetic ankle is not one hinge. It is two hinges on the same body, at the
same anchor, on the same axis, with disjoint ranges. Each hinge carries its own
`stiffness`.

`models/22muscle_2D/myoLeg22_2D_KAP.xml:207-208`:

```xml
<body name="kap_foot_assem" pos="0.0525 -0.053 0.0025">
  <joint name="df_ankle_angle_r" pos="-0.042 0.04 0.0" axis="0 0 1" range="0 0.5236"       stiffness="500" damping="5"/>
  <joint name="pf_ankle_angle_r" pos="-0.042 0.04 0.0" axis="0 0 1" range="-0.5236 0"      stiffness="50"  damping="5"/>
```

The same pair appears in `models/22muscle_2D/myoLeg22_2D_K-Foot.xml:214-215` and in
`assist_sim/assist_sim/models/KFoot/L1model.xml:49-50`, with identical ranges,
damping, and default stiffness.

Both joints inherit `limited="true"` from the `main` default class
(`myoLeg22_2D_KAP.xml:35`), so MuJoCo enforces the ranges. `springref` is not set,
so both springs pull toward 0 rad.

The result is a piecewise-linear ankle spring:

| Ankle deflection | Active joint | Restoring torque |
| --- | --- | --- |
| `theta > 0` (dorsiflexion, up to +30 deg) | `df_ankle_angle_r` | `-K_df * theta` |
| `theta < 0` (plantarflexion, down to -30 deg) | `pf_ankle_angle_r` | `-K_pf * theta` |

The compound ankle angle is the **sum** of the two joint positions. The interface
computes it that way in `ctrl_optim/ctrl/reflex/reflex_interface.py:1096-1120`
(`_get_ankle_angle`), which falls back to `df + pf` when `ankle_angle_r` is absent.

"Asymmetric" has two meanings here, and both apply:

1. **Directional asymmetry.** `K_pf` and `K_df` are independent, and their
   optimization ranges differ by roughly a factor of 3.5. A real prosthetic foot
   is much stiffer in dorsiflexion than in plantarflexion.
2. **Limb asymmetry.** Only the right leg is prosthetic. The left ankle stays
   biological (`ankle_angle_l` plus `mtp_angle_l`). There is no left-side
   equivalent of these two parameters.

### 1.2 Physics notes that matter for a reimplementation

- **Neutral crossing is soft.** MuJoCo joint limits are soft constraints. Near
  `theta = 0` both joints sit at a limit and both can move slightly, so the
  stiffness transition is a narrow blend, not a true step. The width depends on
  `solimp`, `solref`, and `margin`. Any figure that draws a vertical step at
  0 deg (as `ctrl_optim/plot_stiffness_comparison.py` does) is an idealization.
- **Damping does not scale the same way as stiffness.** Each joint has
  `damping="5"`. Away from neutral only one joint moves, so the effective damping
  is about 5 N.m.s/rad. Near neutral both can move, so it can approach 10.
  Damping is **not** optimized. If you make stiffness a free parameter, decide
  explicitly whether damping stays fixed.
- **The two joints add two DOF.** `nq`, `nv`, and the qpos layout change relative
  to a single-hinge ankle. Any hard-coded qpos index list breaks. See the
  122-element hard-coded `initial_qpos` in
  `ctrl_optim/test_ankle_stiffness_plot.py:88-94` for an example of code that is
  coupled to this layout.
- **Direction naming is a claim, not a measurement.** The joint named `df_...`
  takes the positive range on axis `0 0 1`. Confirm by measurement that positive
  rotation on that axis is dorsiflexion for the composed model you use. Do not
  trust the name or the comment.

### 1.3 Instrumentation already present

`myoLeg22_2D_KAP.xml:545-549`:

```xml
<jointlimitfrc name="r_ankle_df_sensor" joint="df_ankle_angle_r"/>
<jointlimitfrc name="r_ankle_pf_sensor" joint="pf_ankle_angle_r"/>
<force          name="kap_force_sensor" site="r_kap_load_force"/>
```

The two `jointlimitfrc` sensors report limit-constraint force, not spring torque.
They tell you when the ankle is bottomed out. The `kap_force_sensor` at the
socket site drives the socket-load cost
(`ctrl_optim/optim/cost_functions/evaluate_cost.py:1509-1552`).

---

## Part II. As-built implementation

### 2.1 File map

| Role | File | Key lines |
| --- | --- | --- |
| Denormalize and write stiffness | `ctrl_optim/ctrl/prosthetic/set_ankle_stiffness.py` | 14-15, 18-22, 25-83 |
| Apply per evaluation | `ctrl_optim/ctrl/reflex/reflex_interface.py` | 30, 117, 146-153, 218-236, 260-261, 345-408 |
| Compound ankle angle | `ctrl_optim/ctrl/reflex/reflex_interface.py` | 1096-1120 |
| CLI flag | `ctrl_optim/optim/config/arg_parser.py` | 109-110, 255 |
| Env dict (second copy) | `ctrl_optim/optim/config/environment.py` | 65 |
| CMA-ES bounds | `ctrl_optim/optim/optim_utils/bounds.py` | 272-274, 306-308 |
| Param count, init, warm start | `ctrl_optim/optim/train.py` | 166, 175-176, 178-203, 225-232, 249-253, 297-300 |
| Pass-through to interface | `ctrl_optim/optim/cost_functions/walk_cost.py` | 74-79, 97 |
| Eval-time reconstruction | `ctrl_optim/optim/optim_utils/config_parser.py` | 47, 68, 108-122, 175 |
| Post-processing | `ctrl_optim/plot_stiffness_comparison.py` | 48, 52-56, 187-216 |
| Run config | `ctrl_optim/optim/training_configs/amp_stiffness.bat` | whole file |
| Smoke test | `ctrl_optim/ctrl/prosthetic/test_stiffness.py` | whole file |

### 2.2 Parameter vector layout

The stiffness block is always the **last two elements** of the CMA-ES vector, in
the order `[p_pf, p_df]`.

```
[ reflex + pose block ] [ exo spline block (optional) ] [ p_pf, p_df ]
```

Block sizes, 2D:

| Mode | Reflex + pose | Exo spline | Stiffness | Total |
| --- | --- | --- | --- | --- |
| `--reflex_mode amp`, no exo | 120 (43 right + 51 left + 26 pose) | 0 | 2 | **122** |
| Symmetric, no exo | 78 | 0 | 2 | 80 |
| `amp` + n-point exo | 120 | `2 * n_points` | 2 | 122 + 2n |

The interface derives the same counts independently
(`reflex_interface.py:128-153`), and `train.py:156-176` derives them a third
time. All three must agree or the run fails with "Wrong number of params".

### 2.3 Call chain, per candidate evaluation

```
train.py                      CMA-ES ask() -> X (list of 122-vectors)
  cost_wrapper_with_logging
    walk_cost.func_Walk_FitCost(params, ..., env_dict)
      myoLeg_reflex(..., optimize_stiffness=True, reflex_mode='amp')   # new env per candidate
        -> stiffness_param_count = 2                                   # reflex_interface.py:146-149
      Myo_env.reset(params)
        -> stiffness_params_values = CONTROL_PARAM[-2:]                # reflex_interface.py:392-397
        -> apply_stiffness(self.env.sim, p_pf, p_df)                   # reflex_interface.py:404-408
             model.jnt_stiffness[pf_id] = 30  + p_pf * 275             # set_ankle_stiffness.py:77-81
             model.jnt_stiffness[df_id] = 100 + p_df * 950
        -> set_init_pose / adjust pose / adjust height
      loop run_reflex_step_Cost() to sim_time or fall
      evaluateCost(...) -> scalar
```

Two properties of this design are worth keeping:

- **The model is mutated in memory, not on disk.** The original plan in
  `stiffness_plan.md` wrote a temporary XML per candidate and compiled it. The
  shipped code writes `model.jnt_stiffness` directly. That removed per-candidate
  file I/O, temp-file cleanup, and Windows file-lock retries. Keep the in-memory
  approach.
- **Application happens in `reset()`, not `__init__`.** So a reused environment
  picks up new stiffness on every reset. This is what makes the eval and video
  scripts work.

### 2.4 Normalization

`ctrl_optim/ctrl/prosthetic/set_ankle_stiffness.py:14-15`:

```python
PF_RANGE = (30.0, 305.0)    # Nm/rad
DF_RANGE = (100.0, 1050.0)  # Nm/rad
```

```python
value = lower + clip(p, 0, 1) * (upper - lower)
```

Reference points:

| Quantity | `p_pf` | `K_pf` (Nm/rad) | `p_df` | `K_df` (Nm/rad) |
| --- | --- | --- | --- | --- |
| Range lower bound | 0.000 | 30.0 | 0.000 | 100.0 |
| XML default (`KAP`, `K-Foot`, `KFoot_L1`) | 0.073 | 50.0 | 0.421 | 500.0 |
| CMA-ES initial value | 0.500 | 167.5 | 0.500 | 575.0 |
| Range upper bound | 1.000 | 305.0 | 1.000 | 1050.0 |
| Example converged run (see 2.7) | 0.876 | 270.8 | 0.818 | 877.4 |

### 2.5 Configuration surface

One flag turns the feature on:

```
--optimize_stiffness        # store_true, arg_parser.py:109-110
```

It propagates as `env_dict['optimize_stiffness']` and then as the
`optimize_stiffness` keyword of `myoLeg_reflex`. The evaluation path recovers it
by re-parsing the saved `.bat` file (`config_parser.py:108-122`), including a
regex for a value form (`--optimize_stiffness 1`) that the parser never emits.

The model must be one that has the two joints. Only `--model kap` reaches such a
model through `resolve_model_path` (`optim_utils/resolve_path.py:63-70`).

Working run configuration, `ctrl_optim/optim/training_configs/amp_stiffness.bat`:

```bat
python train.py ^
    --musc_model 22 ^
    --model kap ^
    --sim_time 20 ^
    --pose_key walk_left ^
    --num_strides 5 ^
    --delayed 0 ^
    --optim_mode single ^
    --reflex_mode amp ^
    --tgt_vel 1.25 ^
    --tgt_slope 0 ^
    --trunk_err_type ref_diff ^
    --tgt_sym_th 0.1 ^
    --tgt_grf_th 1.5 ^
    -class ^
    --optimize_stiffness ^
    --popsize 32 ^
    --maxiter 500 ^
    --threads 32 ^
    --sigma_gain 10 ^
    --save_path results/amp_stiffness ^
    --param_path ..\results\optim_results\amp_stiffness_0129_1038
```

### 2.6 Bounds, initial value, step size

- **Bounds.** `[0, 1]` for each of the two parameters, appended after the exo
  block (`bounds.py:272-274`). Every reflex gain uses `[0, inf]`, so the stiffness
  parameters are the only tightly bounded entries besides the pose velocity and
  the initial-activation scales.
- **Initial value.** 0.5 for both, written by `init_param_segments`
  (`train.py:195-201`). When warm starting from a shorter parameter file, the two
  values are appended and set to 0.5. When warm starting from a longer file during
  spline bootstrapping, the trailing two values are lifted out first, the spline is
  re-fitted, and the stiffness pair is re-appended unchanged (`train.py:249-253`,
  `297-300`).
- **Step size.** `sigma_0 = 0.01` scaled by `CMA_stds = sigma_gain * ones`
  (`train.py:324-332`). With `--sigma_gain 10` the per-coordinate standard
  deviation is 0.1 in normalized units, so about 27.5 Nm/rad for PF and
  95 Nm/rad for DF. pycma's boundary handling keeps proposals feasible, and
  `_denormalize` clips again.

### 2.7 Outputs

`train.py` writes, into `ctrl_optim/results/optim_results/<save_path>_<MMDD_HHMM>/`:

- `*_Best.txt` and `*_BestLast.txt`: raw parameter vectors, one value per line.
  The stiffness pair is the last two lines. **Absolute stiffness is never saved.**
- `*_Best_Cost.txt`, `*_BestLast_Cost.txt`: cost component breakdown.
- `*_Pickle.pkl`: CMA-ES state, for `--pickle_path` continuation.
- `<config>_<MMDD_HHMM>.bat`: copy of the run config, which is what the eval path
  later re-parses to recover `optimize_stiffness`.

Post-processing lives in `ctrl_optim/plot_stiffness_comparison.py` and
`ctrl_optim/process_stiffness_results.py`. `extract_stiffness_params`
(`plot_stiffness_comparison.py:187-216`) reads the last two values of a parameter
file and denormalizes them.

Worked example, `results/optim_results/amp_stiffness_0129_1617/..._BestLast.txt`
(122 values):

```
p_pf = 0.875740  ->  K_pf =  30 + 0.875740 * 275 = 270.83 Nm/rad
p_df = 0.818335  ->  K_df = 100 + 0.818335 * 950 = 877.42 Nm/rad
```

The optimizer moved from the XML default ratio `K_df / K_pf = 10.0` to `3.24`. It
raised plantarflexion stiffness by 5.4x and dorsiflexion stiffness by 1.8x.
Read Part III before you treat that result as a finding.

---

## Part III. Defects and traps in the as-built code

These are ordered by how much they affect a reimplementation. Each one needs an
explicit decision, not a silent copy.

### D1. A hidden L1 penalty pulls both stiffness parameters toward maximum

This is the most important item in this document.

`ctrl_optim/optim/cost_functions/walk_cost.py:193`:

```python
muslen_param = params[-1 * len(Myo_env.mus_len_key):]
```

`mus_len_key` is an empty list (`reflex_interface.py:93`). So `-1 * 0 == 0`, and
the slice is `params[0:]`, which is **the entire parameter vector**, not an empty
one. Then `evaluate_cost.py:487-488`:

```python
if len(muslen_param) > 0:
    total_cost += np.sum(np.sqrt(np.square(muslen_param - 1)))
```

`sqrt(square(x))` is `abs(x)`. So every successful evaluation adds
`sum |p_i - 1|` over all parameters to the cost.

Measured on the converged 122-parameter solution in 2.7:

| Term | Value |
| --- | --- |
| Hidden penalty, all 122 parameters | 31.111 |
| Hidden penalty, the 2 stiffness parameters only | 0.306 |
| `Effort_Cost` for the same solution | 2.218 |

The consequence for the stiffness study is direct. Both stiffness parameters live
on `[0, 1]`, and the penalty is `|p - 1|`, so the cost gradient with respect to
each one is a constant `-1`. Moving `p_pf` from 0 to 1 earns a 1.0 cost reduction
regardless of gait quality. That is about 45% of the entire effort cost for this
solution. The optimizer is paid to make the ankle as stiff as the range allows.

The high converged values (0.876, 0.818) are therefore not clean evidence that
stiff is better. They are consistent with the artifact.

This defect is also present in the MyoAssist 1.0 stack
(`myoassist/ctrl_optim/optim/cost_functions/walk_cost.py:117` and
`evaluate_cost.py:292-293`). Do not assume a fresh checkout is free of it.

**Action.** Remove the term, or reinstate it as a deliberate, named regularizer
that applies only to the muscle-length block and excludes the device block. Then
re-run the stiffness study. Do not compare new results to the old ones without
re-running the baseline.

### D2. Documented ranges do not match the code

`stiffness_plan.md:8` specifies PF `[30, 300]` and DF `[180, 1800]`. The
docstring in `set_ankle_stiffness.py:48-49` still says "PF: 30-300; DF:
180-1800". The constants at lines 14-15 are `(30, 305)` and `(100, 1050)`.

A normalized parameter is meaningless without its range. A `p_df` of 0.818 means
877 Nm/rad under the shipped constants and 1573 Nm/rad under the docstring. Every
saved parameter file in `ctrl_optim/results/optim_results/` is ambiguous unless
you know which constants were in effect.

**Action.** Persist the ranges with the results, not only in source. Fix the
docstring. Pick the ranges from a citable source for the specific foot and record
that citation next to the constants.

### D3. Two independent denormalize implementations

`_denormalize` exists in `set_ankle_stiffness.py:18-22` and again in
`plot_stiffness_comparison.py:52-56`. The plot script imports `PF_RANGE` and
`DF_RANGE` from the first module but not the function. Any change to the mapping
form, for example a log-scale mapping, silently desynchronizes analysis from
simulation.

**Action.** One function, one import site.

### D4. Stiffness optimization is unreachable in 3D

`bounds.getBounds_22_26_mus` defines `bound_vect` only inside the `mode == '2D'`
branch. The `elif mode == '3D'` branch at `bounds.py:280-289` calls
`bound_vect.extend(...)` on a name that was never assigned, which raises
`NameError`. The stiffness append at `bounds.py:306-308` sits in that dead branch.

**Action.** Fix or delete the 3D branch. Do not port it as written.

### D5. Amp mode and stiffness are entangled by a parameter-count guard

`walk_cost.py:74-79` raises if `len(params) >= 120` and `reflex_mode != 'amp'`.
This ties a length test to a mode name. A future non-amp configuration with 120 or
more parameters fails with a misleading message about `--reflex_mode`.

The two features are logically independent: the two-joint prosthetic ankle needs
a prosthetic model, not necessarily the asymmetric reflex controller.

**Action.** Validate the parameter count against the count the configuration
implies. Do not infer mode from length.

### D6. Parameter counts are derived in three places

`train.py:156-176`, `reflex_interface.py:128-153`, and `bounds.py` each compute
the layout independently. `reflex_interface.py:155-160` silently discards the
caller's vector and substitutes `np.ones(expected_params)` on a mismatch, which
turns a configuration error into a run that trains the wrong thing. The legacy
77-versus-78 base-count upgrade path (`reflex_interface.py:429-439`) adds more
surface.

**Action.** One function returns the layout. Everything else calls it. On a
mismatch, raise.

### D7. No absolute values are logged

Nothing in the optimization loop records `K_pf` and `K_df` in Nm/rad.
`reflex_interface.py:260-261` stores `last_pf_stiffness` and `last_df_stiffness`
on the instance, and nothing reads them. Every stiffness number in the analysis is
reconstructed after the fact from the parameter file plus an assumed range. See D2.

**Action.** Log absolute values per evaluation, and write the final absolute
values into a machine-readable results file.

### D8. Smaller items

- `set_ankle_stiffness.apply_stiffness` catches `KeyError` from
  `model.joint(name)`. Confirm the binding version raises `KeyError` and not
  `ValueError` for a missing name, or the error surfaces as an opaque crash inside
  a worker process.
- `--model osl` is in the `arg_parser.py:26-28` choices but absent from the
  `resolve_path.py:63-70` mapping, so it raises `ValueError`. `myoLeg22_2D_K-Foot.xml`
  has the two joints but no registry entry, so it is unreachable from the CLI.
- `config_parser.parse_bat_config` recovers the run configuration by regex over a
  `.bat` file. It has a pattern for `--optimize_stiffness <int>`, a form nothing
  writes. Round-trip the configuration as JSON instead.
- `test_stiffness.py` and `test_ankle_stiffness_plot.py` are scripts that save
  plots and video. Neither asserts anything. There is no test that
  `model.jnt_stiffness` holds the expected value after a reset.

---

## Part IV. Reimplementation instructions

Target: a control-optimization framework where the model is composed from a
musculoskeletal key plus a device key, and CMA-ES optimizes one flat parameter
vector. In the MyoAssist 1.0 stack that is `myoassist/ctrl_optim`, where
`myoLeg_reflex.__init__` takes `msk_key`, `device_key`, and `terrain`, and
`compose_env_model(msk_key, device_key, terrain=terrain)` builds the MJCF.
Stiffness optimization is currently absent from that stack: a grep for
`stiffness` across `myoassist/ctrl_optim` returns nothing, and there is no
`ctrl/prosthetic/` package.

Work through the steps in order. Steps 1 to 4 are the feature. Steps 5 to 8 are
what makes the results trustworthy.

### Step 0. Decide the two policy questions first

Do not start coding until these are answered, because both change the numbers.

1. **Keep or remove the hidden L1 penalty (D1)?** Recommendation: remove it, and
   re-run a no-stiffness baseline before any stiffness run. Without this, the
   study cannot distinguish "stiffer gait is better" from "the cost function pays
   for stiffness".
2. **Which absolute stiffness ranges, from which source (D2)?** Recommendation:
   derive both ranges from published quasi-stiffness data for the specific foot,
   record the citation in the source next to the constants, and write the ranges
   into the results.

### Step 1. Confirm the model contract

Before any optimizer work, verify by measurement, on the composed model you will
actually run:

```python
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_string(composed_xml)   # or from_xml_path
for name in ("pf_ankle_angle_r", "df_ankle_angle_r"):
    j = m.joint(name)
    print(name, "id", j.id, "range", j.range, "stiffness", m.jnt_stiffness[j.id],
          "damping", m.jnt_damping[j.id], "limited", m.jnt_limited[j.id],
          "springref", m.jnt_springref[j.id], "axis", j.axis, "body", m.body(j.bodyid[0]).name)
```

Assert all of the following:

- Both joints exist, are hinges, and are children of the same body.
- Their ranges are disjoint and meet at 0: `[-0.5236, 0]` and `[0, 0.5236]`.
- `jnt_limited` is true for both. `jnt_springref` is 0 for both.
- Their axes are identical.
- Positive rotation on that axis is dorsiflexion. Verify by setting a positive
  `qpos` on the DF joint, calling `mj_forward`, and checking that the toe moves up
  relative to the shank. Do not trust the joint name.

In the MyoAssist 1.0 stack, `device_key = "KFoot_L1"` provides this pair
(`assist_sim/assist_sim/models/KFoot/L1model.xml:49-50`), and it composes against
`myolegs22`, `myolegs26`, `myolegs`, and `myofullbody`. Run the check for each
`msk_key` you intend to use, since composition can rename or re-parent elements.

### Step 2. Write the stiffness module

One module, library only, no CLI. Suggested path
`ctrl_optim/ctrl/prosthetic/ankle_stiffness.py`.

It must provide:

```python
PF_RANGE = (lo, hi)   # Nm/rad, with a source citation in a comment
DF_RANGE = (lo, hi)

N_STIFFNESS_PARAMS = 2
PF_JOINT = "pf_ankle_angle_r"
DF_JOINT = "df_ankle_angle_r"

def denormalize(p, lower, upper) -> float: ...

def apply_stiffness(sim, p_pf, p_df, *, pf_joint=PF_JOINT, df_joint=DF_JOINT,
                    pf_range=PF_RANGE, df_range=DF_RANGE) -> tuple[float, float]:
    """Write absolute stiffness into model.jnt_stiffness. Return (K_pf, K_df)."""

def describe_ranges() -> dict:
    """Return the ranges and joint names for serialization into results."""
```

Rules:

- Mutate `sim.model.jnt_stiffness` in place. Do not write temporary XML files.
  The temp-XML design in `stiffness_plan.md` was superseded for good reason.
- Look joints up by name and **raise** with the joint name and the available joint
  names if either is missing. Never fall back to a default silently.
- Clip normalized inputs to `[0, 1]` and validate that they are finite.
- Keep `denormalize` as the single implementation. Analysis code imports it.
  Do not copy it (D3).
- `describe_ranges` is what closes D2 and D7.

### Step 3. Own the parameter layout in one place

Add one module, for example `ctrl_optim/optim/optim_utils/param_layout.py`, that
turns a configuration into a layout and is the only source of truth:

```python
@dataclass(frozen=True)
class ParamLayout:
    base: int                      # reflex + pose
    spline: int                    # 0 if exo disabled
    stiffness: int                 # 0 or 2
    @property
    def total(self) -> int: ...
    def slice_base(self) -> slice: ...
    def slice_spline(self) -> slice: ...
    def slice_stiffness(self) -> slice: ...

def layout_from_config(cfg) -> ParamLayout: ...
```

Then replace every independent count derivation with a call to this. Concretely,
in the current code that is `train.py:156-176`, `reflex_interface.py:128-160`,
`reflex_interface.py:345-397`, and the append blocks in `bounds.py:272-274` and
`306-308`. This closes D6 and removes the `np.ones(expected_params)` silent
fallback.

Keep the stiffness block **last**. Every saved parameter file, every warm start,
and every analysis script depends on that position.

### Step 4. Apply stiffness on reset

In the reflex interface:

```python
# __init__
self.optimize_stiffness = optimize_stiffness
self.layout = layout_from_config(cfg)
self.last_pf_stiffness = None
self.last_df_stiffness = None

# reset(), after env.reset() and before set_init_pose()
if self.layout.stiffness:
    p_pf, p_df = self.CONTROL_PARAM[self.layout.slice_stiffness()]
    self.last_pf_stiffness, self.last_df_stiffness = apply_stiffness(self.env.sim, p_pf, p_df)
```

Ordering requirements:

- After `env.reset()`. A gym reset restores state, not model fields, but calling
  after the reset keeps the contract obvious.
- Before `set_init_pose` / pose adjustment / height adjustment. The seating logic
  settles the model against gravity and passive springs, so it must see the
  candidate's stiffness, not the XML default.
- Before the single warm-up `env.step` used in delayed mode.

Do not apply stiffness in `__init__` only. Reused environments must pick up new
values on each reset.

### Step 5. Bounds, initial value, and step size

- Append `[0, 1]` for each of the two parameters, in the same place the layout
  puts them.
- Initialize both to 0.5, or better, initialize to the normalized equivalent of
  the model's own XML default so that iteration 0 reproduces the stock device.
  For `KFoot_L1` that is `p_pf = 0.073`, `p_df = 0.421` under the ranges in 2.4.
  Recompute if you change the ranges.
- Set the per-coordinate standard deviation for these two entries explicitly
  rather than inheriting a global `sigma_gain`. The reflex gains are unbounded and
  order 1. The stiffness parameters are bounded to a unit interval. A shared
  `CMA_stds` couples two unrelated scales.
- Warm start: when a loaded parameter file is shorter than the target length,
  append and initialize the pair. When it is longer, lift the trailing pair out
  before any spline re-fit, then re-append it unchanged. `train.py:249-253` and
  `297-300` show the pattern that works.

### Step 6. Configuration and serialization

- One boolean in the run configuration, for example `optimize_stiffness`.
  Propagate it through the env configuration into the interface constructor.
- Do **not** recover the configuration by regex over a shell script (D8). Write
  the resolved configuration to JSON next to the results, and have the evaluation
  and plotting paths load that JSON.
- Write a `stiffness.json` (or a section in the run JSON) containing:

```json
{
  "optimize_stiffness": true,
  "pf_joint": "pf_ankle_angle_r",
  "df_joint": "df_ankle_angle_r",
  "pf_range_nm_per_rad": [30.0, 305.0],
  "df_range_nm_per_rad": [100.0, 1050.0],
  "param_indices": [120, 121],
  "best": {"p_pf": 0.8757, "p_df": 0.8183, "K_pf": 270.83, "K_df": 877.42}
}
```

This is what makes a saved parameter vector self-describing and closes D2 and D7.

### Step 7. Fix the cost function before you run anything

Apply the Step 0 decision on D1. If you remove the term, also remove
`muslen_param` from the `evaluateCost` signature so it cannot come back. If you
keep it, pass an explicit muscle-length slice and assert that the device block is
excluded.

Then add a guard test, described in Step 8.

### Step 8. Tests

These are the tests the current code lacks. Write them before the first long run.

1. **Model contract.** For each `(msk_key, device_key)` pair you support, assert
   the Step 1 conditions. Parametrize over pairs.
2. **Denormalization.** `denormalize(0, lo, hi) == lo`, `denormalize(1, lo, hi) == hi`,
   midpoint is the mean, out-of-range inputs clip.
3. **Applied value.** Build the env, reset with a known vector, then assert
   `sim.model.jnt_stiffness[pf_id]` and `[df_id]` equal the expected absolute
   values to floating-point tolerance. This is the test that would have caught a
   silent no-op.
4. **Layout.** For each supported configuration, assert
   `len(bounds[0]) == layout.total` and that
   `layout.slice_stiffness()` is the final two indices.
5. **Round trip.** Save results, reload the JSON plus the parameter file, and
   assert the reconstructed `K_pf` and `K_df` match what the run applied.
6. **Cost isolation (the D1 guard).** Evaluate one gait, then evaluate the same
   gait with only `p_pf` and `p_df` changed to a value that does not alter the
   simulated trajectory, for example by pointing the joint names at a locked
   dummy joint. Assert the cost is unchanged. Any residual difference means a
   parameter-value term is still leaking into the cost.
7. **Asymmetry sanity.** Hold all joints except the ankle, sweep the compound
   ankle angle from -30 to +30 deg, record `qfrc_passive` or the joint torque,
   and assert the measured slope matches `K_pf` below 0 and `K_df` above 0 within
   tolerance, excluding a small band around 0. This measures the mechanism rather
   than assuming it, and it quantifies the soft-limit blend width described in
   1.2. Note that
   `ctrl_optim/test_ankle_stiffness_plot.py` performs this sweep for video but
   asserts nothing, and it plots the commanded stiffness constants rather than a
   measured torque.

### Step 9. Run and report

Minimum run set for a defensible result:

| Run | Purpose |
| --- | --- |
| Baseline, stiffness fixed at XML default | Reference. Required. |
| Stiffness optimized | The treatment. |
| Stiffness optimized, second seed | Confirms the pair is not seed noise. |

Report `K_pf` and `K_df` in Nm/rad, their ratio, and the ranges used. Report the
cost breakdown for each run. State whether the L1 term was present.

Compare the swept torque-angle curve from test 7 against the device's measured or
published quasi-stiffness. Two numbers inside their optimizer bounds are not
evidence on their own that the result is physically reasonable.

---

## Appendix A. Constants

| Constant | Value | Source |
| --- | --- | --- |
| PF joint name | `pf_ankle_angle_r` | `myoLeg22_2D_KAP.xml:208` |
| DF joint name | `df_ankle_angle_r` | `myoLeg22_2D_KAP.xml:207` |
| PF range of motion | `[-0.5236, 0]` rad = `[-30, 0]` deg | same |
| DF range of motion | `[0, 0.5236]` rad = `[0, 30]` deg | same |
| Damping, both joints | 5 N.m.s/rad | same |
| XML default `K_pf` | 50 Nm/rad | same |
| XML default `K_df` | 500 Nm/rad | same |
| `PF_RANGE` for optimization | `(30.0, 305.0)` Nm/rad | `set_ankle_stiffness.py:14` |
| `DF_RANGE` for optimization | `(100.0, 1050.0)` Nm/rad | `set_ankle_stiffness.py:15` |
| Stiffness parameter count | 2, `[p_pf, p_df]`, vector tail | `reflex_interface.py:146-149` |
| CMA-ES bounds per parameter | `[0, 1]` | `bounds.py:272-274` |
| CMA-ES initial value | 0.5 | `train.py:195-199` |
| `sigma_0` | 0.01, scaled by `--sigma_gain` | `train.py:324-332` |
| Models with the joint pair | `myoLeg22_2D_KAP.xml`, `myoLeg22_2D_K-Foot.xml`, `KFoot/L1model.xml` | grep |
| Device registry key, 1.0 stack | `KFoot_L1` | `assist_sim/.../KFoot/L1config.yaml:2` |

## Appendix B. Defect summary

| ID | Defect | Severity | Must fix before reuse |
| --- | --- | --- | --- |
| D1 | Hidden L1 penalty (`sum` of `abs(p - 1)`) pulls stiffness to maximum | Critical | Yes |
| D2 | Documented ranges differ from code; ranges not persisted | High | Yes |
| D3 | Duplicate `_denormalize` in analysis code | Medium | Yes |
| D4 | 3D bounds branch raises `NameError` | Medium | Yes, or delete |
| D5 | Parameter-count guard couples stiffness to `amp` mode | Medium | Yes |
| D6 | Layout derived in three places; silent `np.ones` fallback | Medium | Yes |
| D7 | Absolute stiffness never logged | Medium | Yes |
| D8 | Unreachable `--model osl`; `.bat` regex config recovery; no assertions in tests | Low | Recommended |