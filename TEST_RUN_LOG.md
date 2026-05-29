# Test Run Log

## 2026-05-26

### Minor issues fixed

- Updated `tests/optimisers/test_lbfgs.py` to match the current clipped-step/clipped-gradient defaults and the current convergence behavior of `LBFGSOptimiser`.
- Updated accelerator test helpers to implement the now-required `copy_with()` method and to model quadrupole/sextupole optimisation flags consistently.
- Aligned accelerator expectations with current runtime semantics:
  - `kinetic_energy` is exposed as total energy.
  - `LHC` tests no longer expect a `pc` attribute.
  - `PSB` tune-variable casing is now uppercase-sensitive.
- Updated `tests/training/controller/test_controller_uncertainty.py` for the current `_finalise_results(total_hessian, writer)` signature.
- Updated `tests/training/workers/test_worker_setup.py` to pass an explicit `FullRingBpmTrackingPlan()` after the helper constructor gained a required `tracking_plan` argument.

### Major issues

- None identified yet in the non-slow test slices that were rerun after the fixes above.

### Remaining work

- Slow and heavier integration suites still merit a dedicated pass under `accpy` because they take substantially longer and may surface environment- or MAD-related issues rather than simple unit-test drift.
