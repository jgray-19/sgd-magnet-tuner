# Refactoring Plan: `src/aba_optimiser/training/`

## Overview

The training module implements a multi-process knob optimisation loop using MAD-NG. It is broadly well-structured, but has grown organically and contains several design issues that make it hard to test, extend, and reason about. This document describes the issues and proposes concrete fixes.

---

## Module Map

| File | Role |
|---|---|
| `controller.py` | Top-level orchestrator: data → workers → loop → results |
| `base_controller.py` | Abstract base: MAD setup, knob init, deferred manager creation |
| `configuration_manager.py` | MAD interface, BPM lists, initial strength loading |
| `optimisation_loop.py` | Epoch loop, gradient aggregation, knob updates, checkpointing |
| `data_manager.py` | Load/filter parquet data, prepare turn batches |
| `controller_config.py` | Dataclass config definitions |
| `tracking_mode.py` | Abstract + concrete tracking plan policies |
| `worker_manager.py` | Spawn workers, pipe comms, outlier screening, Hessian collection |
| `worker_setup.py` | Build observation plans, range specs, `WorkerConfig` objects |
| `worker_payloads.py` | Build `TrackingData` from arrays; turn stitching; weight normalisation |
| `worker_lifecycle.py` | Generic lifecycle manager (exists but unused) |
| `worker_turn_planner.py` | Group turns into batches respecting file boundaries |
| `scheduler.py` | Cosine warmup + decay LR schedule |
| `validation_selection.py` | Select validation payloads covering ≥10% of training tracks |
| `utils.py` | Shared: BPM filtering, range extraction, TFS loading |

---

## Status

| # | Issue | Status |
|---|---|---|
| 1 | Broken template method in `_init_managers` | ✅ Done |
| 2 | Dead `_convert_true_strengths_to_delta` | ✅ Done |
| 3 | `WorkerManager._sync_helpers()` contract | ✅ Already auto-called in `create_worker_payloads` |
| 4 | Plane-check helpers duplicated | ✅ Done — moved to `utils.py` |
| 5 | Position cache keyed by `id(df)` | ✅ Done — stable `(turns, bpms)` key |
| 6 | Asymmetric validation worker protocol | Pending |
| 7 | `TrackingPlan` callbacks passed at call time | ✅ Done — methods call utils directly |
| 8 | No `WorkerResult` dataclass | Pending |
| 9 | `_select_worker_class` static if/else | ✅ Done — registry dict |
| 10 | `OptimisationLoop` does too much | Pending |
| 11 | Range extraction indirection | Pending |

---

## Issues and Proposed Fixes

### 1. Broken Template Method Pattern in Initialization (Medium)

**Problem:** `BaseController.__init__` calls `_init_managers()`, but `Controller` overrides it to do nothing. The real initialization happens *after* `Controller.__init__` modifies `simulation_config`, violating the intent of the template method.

```
base_controller.py:128   self._init_managers()  # does nothing in Controller
controller.py:199-206    # real work happens here after super().__init__
```

**Fix:** Remove the `_init_managers()` call from `BaseController.__init__`. Make `Controller` explicitly call its own initialization sequence at the end of `__init__`, in the correct order. If shared post-setup logic is needed, provide a `_post_init()` hook that subclasses call explicitly.

---

### 2. Dead Code in True-Strengths Conversion (Medium)

**Problem:** `Controller._convert_true_strengths_to_delta()` (`controller.py:412-431`) handles energy conversion for true knob strengths, but this path is never reached: the conversion is already handled by `BaseController.convert_deltap_to_pt()` via `super().__init__`.

**Fix:** Audit whether `_convert_true_strengths_to_delta` is called anywhere. If not, delete it. If the two conversion paths are doing different things, consolidate into one method and document which cases each handles.

---

### 3. Mandatory State Sync on WorkerManager (Medium)

**Problem:** `WorkerManager._sync_helpers()` must be called before `create_worker_payloads()` to keep `WorkerSetupHelper` and `WorkerPayloadBuilder` consistent with current manager state (e.g. after planes or files change). This contract is implicit — nothing enforces it.

```
worker_manager.py:132-139   _sync_helpers()  # caller must remember to invoke this
```

**Fix:** Either call `_sync_helpers()` automatically at the start of `create_worker_payloads()`, or make helpers stateless (passed in at call time rather than held as fields). The former is lower risk.

---

### 4. Fragile Position Cache in WorkerPayloadBuilder (Medium)

**Problem:** `_pos_cache` and `_layout_cache` are keyed by `id(df)` (object identity). If a DataFrame is reconstructed or copied rather than mutated, cache misses occur silently, causing redundant computation. There is no invalidation path.

```
worker_payloads.py:51-52   _pos_cache: dict   # keyed by id(df)
```

**Fix:** Key the cache by a stable hash of the DataFrame content (e.g. `hash(df.to_json())` or a tuple of its index + column names + shape). Alternatively, switch to weak-reference keying so cache entries are automatically evicted when the DataFrame is garbage collected.

---

### 5. Asymmetric Validation Worker Protocol (Medium)

**Problem:** Training workers receive `(data, config, batch_idx)` tuples directly, but validation workers receive a single-element list `[validation_payload]`. This inconsistency makes the inter-process protocol hard to follow and is a latent source of protocol divergence bugs.

```
worker_manager.py:449-454   # validation wraps payload in list
```

**Fix:** Unify the communication protocol. Define a small `WorkerMessage` dataclass (or named tuple) with explicit fields for payload, config, and batch index. Use it everywhere. Document what workers should expect to receive and send back.

---

### 6. Plane-Capability Logic Duplicated (Low)

**Problem:** `bpm_supports_plane()` and `bpm_supports_both_planes()` appear in both `WorkerSetupHelper` (`worker_setup.py:114-129`) and `WorkerPayloadBuilder` (`worker_payloads.py:65-78`).

**Fix:** Move these to `utils.py` or (better) to the `Accelerator` abstraction, where knowledge of BPM capabilities logically belongs.

---

### 7. TrackingPlan Methods Take Callbacks as Parameters (Low)

**Problem:** Methods such as `get_range_bpm_names()` on `TrackingPlan` subclasses accept `extract_bpm_range_names` as a callable argument (`tracking_mode.py:122-154`). This is an anti-pattern: it means callers must supply a dependency the plan could hold at construction time.

**Fix:** Inject `extract_bpm_range_names` (or whatever it wraps) into the `TrackingPlan` at construction. Methods then call `self._range_extractor(...)` without needing it threaded through every call site.

---

### 8. No Worker Protocol Definition (Low)

**Problem:** Workers communicate via pipes using bare tuples `(_, grad, loss)`. The first element is discarded without comment (`optimisation_loop.py:635-652`). There is no single place documenting what workers send and what the controller expects.

**Fix:** Define a `WorkerResult` dataclass. Workers return instances; the loop unpacks via attribute access. This makes protocol changes type-safe and self-documenting.

---

### 9. OptimisationLoop Does Too Much (Low)

**Problem:** `optimisation_loop.py` (~735 lines) mixes: epoch iteration, convergence detection, checkpointing, TensorBoard logging, and knob update logic. This makes it hard to test each concern independently.

**Fix (incremental, not a full rewrite):**
- Extract a `ConvergenceChecker` class that holds patience state and decides when to stop.
- Extract a `CheckpointManager` class that handles save/load of checkpoint files.
- Leave loop orchestration and knob updates in `OptimisationLoop`.

This splits three independent concerns while leaving the main loop logic in one place.

---

### 10. Range Extraction Indirection (Low)

**Problem:** `extract_bpm_range_names()` in `utils.py` is wrapped by `TrackingPlan.get_range_bpm_names()`, which is in turn called from `WorkerSetupHelper.get_range_bpm_names()`. The same utility is also called directly at other points. The chain of wrappers obscures where the logic lives.

**Fix:** Callers that need range extraction should call `utils.extract_bpm_range_names()` directly. Remove the pass-through wrapper in `TrackingPlan` unless it adds genuine policy (e.g. filtering). This simplifies the call graph.

---

### 11. Static Worker Class Selection (Low)

**Problem:** `WorkerManager._select_worker_class()` (`worker_manager.py:153-169`) is a static method that switches on mode to return a class. Adding a new worker type requires modifying this method.

**Fix:** Register worker classes in a dict keyed by mode (or by `TrackingPlan` type). `_select_worker_class` becomes a dict lookup. New modes register themselves; no changes needed to `WorkerManager`.

---

## Recommended Priority Order

| # | Change | Effort | Risk |
|---|---|---|---|
| 1 | Unify validation worker protocol (`WorkerMessage`) | Low | Low |
| 2 | Fix deferred `_init_managers()` call | Low | Medium |
| 3 | Auto-call `_sync_helpers()` in `create_worker_payloads()` | Minimal | Low |
| 4 | Move plane-check helpers to `utils.py` or `Accelerator` | Low | Low |
| 5 | Fix position cache keying in `WorkerPayloadBuilder` | Low | Low |
| 6 | Define `WorkerResult` dataclass | Low | Low |
| 7 | Remove `_convert_true_strengths_to_delta` dead code | Minimal | Low |
| 8 | Register worker classes by mode (dict) | Low | Low |
| 9 | Inject `extract_bpm_range_names` into `TrackingPlan` | Medium | Low |
| 10 | Extract `ConvergenceChecker` + `CheckpointManager` | Medium | Low |
| 11 | Simplify range extraction indirection | Low | Low |

Items 1–4 are self-contained changes that reduce latent bugs without restructuring the codebase. Items 9–11 are quality-of-life improvements that improve testability and readability but can wait for a larger refactor session.

---

## What Not To Change

- The multiprocessing + pipe architecture is appropriate for this workload; replacing it (e.g. with `concurrent.futures`) would be churn without benefit.
- The `TrackingPlan` / `ArcByArcTrackingPlan` / `KickerTrackingPlan` hierarchy is sound; no restructuring needed.
- `controller_config.py` dataclass structure is clean; no changes needed.
- `scheduler.py` is simple and correct; no changes needed.
