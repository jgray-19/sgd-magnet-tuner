# Handoff Document — sgd-magnet-tuner Bug Fixes

Date: 2026-05-06. Written for a fresh agent to continue the work.

---

## Context and Goal

The `aba_optimiser` package is an SGD-based magnet tuner for particle accelerators (LHC, PSB, SPS). A **serious convergence bug** was discovered in the dataloader and a cascade of API changes were required. This session completed most of the fixes. The optimizer still does not converge in the controller integration tests — the root cause needs to be found.

Primary repo: `/afs/cern.ch/work/j/jmgray/private/sgd-magnet-tuner`  
Sister packages changed in this session:
- `/afs/cern.ch/work/j/jmgray/private/pymadng-utils/` (`apply_magnet_perturbations`)
- `/afs/cern.ch/work/j/jmgray/private/xtrack-tools/` (`initialise_env`)

Always activate the venv before running anything:
```
source /afs/cern.ch/work/j/jmgray/private/accpy/bin/activate
```

---

## Completed Fixes

### 1. `pc` → `kinetic_energy` API migration

**Files changed:**
- `src/aba_optimiser/training/worker_payloads.py` — renamed `pc: float` → `kinetic_energy: float` in `WorkerPayloadBuilder.__init__`; `compute_pt` now computes `total_energy = kinetic_energy + PROTON_MASS` (both in GeV) and calls `dp2pt(dp, PROTON_MASS, energy=total_energy)`
- `src/aba_optimiser/training/worker_manager.py` — uses `accelerator.kinetic_energy` instead of `accelerator.pc`
- `tests/training/helpers.py` — `initialise_env(kinetic_energy=accel.kinetic_energy, ...)` (was `pc=accel.pc`)

**Note on `dp2pt` and units:** `PROTON_MASS = 938.27e-3 GeV` (defined in `src/aba_optimiser/config.py`). `kinetic_energy` is in GeV throughout. `dp2pt` in `src/aba_optimiser/physics/deltap.py` expects energy and mass in the same unit (GeV). For dpp=0 (quadrupole-only tests), dp2pt returns 0 immediately so the unit conversion doesn't matter. The function `kinetic_to_total_energy` in deltap.py is **mis-named** — it actually computes total energy from **momentum** (`sqrt(pc^2 + m^2)`), not from kinetic energy.

### 2. `apply_magnet_perturbations` naming convention: absolute → delta

Previously returned `element.k1` keys with **absolute strength** values.  
Now returns `element.dk1l` keys with **delta values** (the perturbation only).

**File changed:** `/afs/cern.ch/work/j/jmgray/private/pymadng-utils/src/pymadng_utils/mad/accelerator_mad_interface.py`

The fix was needed because the optimizer creates knobs as `element.dk1l` (delta format using deferred MAD-NG variables), NOT `element.k1`. The naming convention must match.

```python
# Returns e.g. {"MQY.B5L2.B1.dk1l": 1.8e-5, "LSF.13205.dk2l": 3.2e-5}
magnet_strengths, true_strengths = interface.apply_magnet_perturbations(
    rel_error=None, seed=42, magnet_type="q",
)
```

**Tests updated:** `tests/accelerators/test_perturbations.py` — key assertions changed from `.k1`/`.k2` to `.dk1l`/`.dk2l`.

### 3. `initialise_env` handles `dk*l` keys

**File changed:** `/afs/cern.ch/work/j/jmgray/private/xtrack-tools/src/xtrack_tools/env.py`

When `magnet_strengths` contains `dk0l`/`dk1l`/`dk2l` keys, the function now correctly adds the delta to the current base attribute:

```python
_dknl_to_base = {"dk0l": "k0", "dk1l": "k1", "dk2l": "k2"}
for str_name, strength in magnet_strengths.items():
    magnet_name, var = str_name.rsplit(".", 1)
    base_attr = _dknl_to_base.get(var)
    if base_attr is not None:
        current = getattr(base_env[magnet_name.lower()], base_attr, 0.0) or 0.0
        base_env.set(magnet_name.lower(), **{base_attr: current + strength})
    else:
        base_env.set(magnet_name.lower(), **{var: strength})
```

### 4. Data manager: remove `num_batches` from turn distribution

**File changed:** `src/aba_optimiser/training/data_manager.py`

`SimulationConfig.num_batches` = MAD particle-tracking sub-batches per worker call (controls parallelism inside MAD-NG), **NOT** turn distribution granularity. Previously `prepare_turn_batches` was wrongly passing it as `per_worker_batches` to `_build_turn_batches`. This caused the dataloader to return 0 turns for partial batches. Fixed by hardcoding `per_worker_batches=1`.

**Tests updated:** `tests/training/test_data_manager.py` and `tests/training/workers/test_data_worker_sanity.py` — explicit `per_worker_batches=1`.

### 5. `configuration_manager.py`: unknown true strengths → warning not error

**File changed:** `src/aba_optimiser/training/configuration_manager.py`

When `true_strengths` contains quads outside the optimizer's range, the code now logs a `WARNING` instead of raising a `ValueError`. This is correct because `apply_magnet_perturbations` perturbs ALL quads in the ring, but the optimizer only handles quads in its `magnet_range`.

---

## Current Test Status

Run: `python -m pytest tests/ --ignore=tests/training/controller --ignore=tests/accelerators/test_perturbations.py -q`

**Result: 126 passed, 1 failed**

### Failing non-controller test

`tests/mad/test_optimising_mad_interface.py::test_lhc_all_optimisation_combinations_select_expected_knob_list[sextupoles-only-no-lhc-knobs]`

```
AssertionError: assert ['sextupole:k2:MSS?%.:k2'] == []
```

The test expects that `optimise_sextupoles=True` produces no LHC-specific knob specs (because there's no LHC sextupole knob in `get_supported_knob_specs`), but the function is returning `'sextupole:k2:MSS?%.:k2'`. This is a pre-existing or newly exposed failure — investigate the `_expected_lhc_knob_spec_keys` function or the `get_supported_knob_specs` return value.

### Controller tests (all failing)

`tests/training/controller/test_controller_quadrupole.py::test_controller_quad_opt_simple[*]`  
`tests/training/controller/test_controller_quadrupole.py::test_controller_quad_opt_simple_without_early_stopping_reaches_truth`  
Other controller tests not yet rerun.

**Root cause: optimizer does not converge.**

After 300 epochs:
- `estimate = {'MQ.13R1.B1.dk1l': -4.4e-10, 'MQ.14L2.B1.dk1l': -4.4e-10, ...}` ≈ 0
- `true_values = {'MQ.13R1.B1.dk1l': ~1.8e-5, ...}` (non-zero)
- Relative error ≈ 100%

Optimizer log at epochs 254-299:
```
loss=3.296e-08, val=3.265e-08, g=2.079e-07, td=1.824e-01, lr=1.00e-06
```
The loss is essentially flat — barely changing across 50 epochs. Gradient is consistently ~2e-7.

---

## Why the Optimizer Isn't Converging — Analysis

### Learning rate schedule

Config used in failing test (`_make_optimiser_config_quad`):
```python
OptimiserConfig(
    max_epochs=300,
    warmup_epochs=200,
    warmup_lr_start=1e-4,  # start of warmup (HIGH)
    max_lr=1e-6,            # end of warmup (LOW - confusingly named)
    min_lr=1e-6,
    gradient_converged_value=5e-14,
    expected_rel_error=0,
)
```

`LRScheduler` (`src/aba_optimiser/training/scheduler.py`) does:
- Epochs 1–200: cosine from `start_lr=1e-4` → `max_lr=1e-6` (DECREASING — atypical warmup)
- Epochs 201+: fixed at `min_lr=1e-6`

So by epoch 200, LR has already dropped to 1e-6 and stays there. At LR=1e-6, gradient=2e-7:
- Adam effective step ≈ LR = 1e-6 per parameter per epoch (Adam normalises by gradient magnitude)
- After 300 epochs: ~3e-4 total movement per parameter
- True value ≈ 1.8e-5 — smaller than 3e-4, so this SHOULD converge with Adam

**Something else is wrong.** The Adam step should be able to reach 1.8e-5 in 300 epochs with LR=1e-6.

### Hypothesis 1: Loss is insensitive to in-range quadrupole knobs

The `magnet_range = "BPM.13R1.B1/BPM.13L2.B1"` is one arc. The optimizer's knobs are quads within this range. If the xsuite tracking data shows phase advance that doesn't change when those specific quads are perturbed in MAD-NG, the gradient would be near zero.

But `apply_magnet_perturbations` perturbs ALL quads using the error table (via `QUAD_ERROR_TABLE`). Some in-range quads definitely get perturbed. The optimizer should see a signal.

### Hypothesis 2: xsuite tracking vs MAD-NG model phase mismatch has wrong sign/scale

The controller creates a **new** `LHC` object (no perturbations) for its MAD-NG model. The tracking data comes from xsuite with perturbations applied. The optimizer should see a discrepancy in phase advance and converge to the true dk1l values.

BUT — the loss being ~3.3e-8 from the first observed epoch (254) is suspicious. If the model starts wrong (dk1l=0, truth=1.8e-5), the initial loss should be MUCH larger.

**This suggests the loss is small even at the start of training (epoch ~0).** Either:
1. The perturbation was not applied to xsuite (the `dk1l` → xsuite `k1` path has a bug)
2. The loss function doesn't measure the right quantity

### Key thing to verify immediately

Add a log at epoch 0 to see the initial loss. Or check: after `generate_xsuite_env_with_errors`, does the xsuite environment actually have the perturbed k1 values?

```python
# Quick debug: after generate_xsuite_env_with_errors
env, magnet_strengths, matched_tunes, corrector_table = generate_xsuite_env_with_errors(...)
# Check one quad
for key, val in magnet_strengths.items():
    if "dk1l" in key:
        mag = key.rsplit(".", 1)[0].lower()
        print(f"{mag}: k1 in xsuite = {env[mag].k1}, expected delta = {val}")
        break
```

If xsuite has the correct k1 (base + delta), but MAD-NG has k1=base (clean model), the loss MUST be non-zero initially. If it's 3.3e-8 from epoch 0, the physics signal might be masked.

### Hypothesis 3: `getattr` fallback in initialise_env returns wrong value

In `xtrack_tools/env.py`:
```python
current = getattr(base_env[magnet_name.lower()], base_attr, 0.0) or 0.0
```

If `base_env["mqy.b5l2.b1"].k1` returns `None` or a zero-value proxy that evaluates as falsy, `current = 0.0` and the delta is applied on top of 0 instead of the true base k1. This could make the xsuite k1 wrong.

In xtrack, element attributes like `k1` are accessed differently depending on whether the element is a `Quadrupole` or has its strength as a variable. Try:
```python
env["mqy.b5l2.b1"].k1  # vs
env.ref["mqy.b5l2.b1"].k1  # vs
env.vars["mqy.b5l2.b1_k1"]
```

The correct approach may be `base_env[magnet_name.lower()].k1` directly (no `getattr`), or using the line's `tw()` to get the actual values.

---

## File Map for Relevant Code Paths

| File | Role |
|------|------|
| `src/aba_optimiser/training/controller.py` | Top-level orchestrator |
| `src/aba_optimiser/training/optimisation_loop.py` | SGD loop, LR schedule, loss |
| `src/aba_optimiser/training/scheduler.py` | `LRScheduler` (warmup+decay) |
| `src/aba_optimiser/training/worker_manager.py` | Worker pool management |
| `src/aba_optimiser/training/worker_payloads.py` | `WorkerPayloadBuilder` — builds tracking inputs |
| `src/aba_optimiser/training/data_manager.py` | Turn batch distribution |
| `src/aba_optimiser/mad/optimising_mad_interface.py` | `GradientDescentMadInterface` — creates dk1l knobs in MAD-NG |
| `tests/training/controller/test_controller_quadrupole.py` | Main failing test |
| `tests/training/controller_test_utils.py` | `_generate_nonoise_track`, `_make_optimiser_config_quad` |
| `tests/training/helpers.py` | `generate_xsuite_env_with_errors`, `generate_model_with_errors` |
| `/afs/.../pymadng-utils/.../accelerator_mad_interface.py` | `apply_magnet_perturbations` |
| `/afs/.../xtrack-tools/.../env.py` | `initialise_env` |

---

## Recommended Next Steps

1. **Debug initial loss**: Add a `print(f"Initial loss: {loss}")` at epoch 0 inside the optimisation loop. If initial loss ≈ 3.3e-8 (same as epoch 254), the perturbation signal is not reaching the loss function.

2. **Verify xsuite perturbation**: After `generate_xsuite_env_with_errors`, inspect the xsuite env to confirm the perturbed quads have `k1 = k1_base + dk1l`.

3. **Check `getattr` behavior**: In `xtrack_tools/env.py`, test whether `getattr(env["mqy.b5l2.b1"], "k1", 0.0)` returns the actual k1 or 0.0. If xtrack doesn't expose `k1` as a plain Python attribute on the returned element reference, the fallback 0.0 means the delta is applied on top of zero (wrong).

4. **Fix the sextupole knob test**: `tests/mad/test_optimising_mad_interface.py::test_lhc_all_optimisation_combinations_select_expected_knob_list[sextupoles-only-no-lhc-knobs]` — this is likely a missing entry in `get_supported_knob_specs` for LHC sextupoles.

5. **Run perturbation tests**: `python -m pytest tests/accelerators/test_perturbations.py -v` — should pass after the `dk1l` naming change.
