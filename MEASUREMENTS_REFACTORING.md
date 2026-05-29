# Measurements Refactoring

This note describes a clean way to keep refactoring `src/aba_optimiser/measurements/` after the new `preprocessing` module was introduced.

## Goals

- Make each module own one job.
- Separate pure data transforms from I/O and external tools.
- Reduce cross-imports between measurement scripts, controller code, and reconstruction helpers.
- Make unit tests target stable small functions instead of large orchestration entrypoints.
- Remove legacy compatibility pressure where it blocks a cleaner API.

## Current Pain Points

- `create_datafile.py` still mixes:
  - orchestration
  - measurement loading
  - optics analysis execution
  - variance assignment
  - reconstruction setup
  - parquet shaping
- The `measurements` package contains both:
  - reusable library code
  - one-off scripts/workflows
- Some module names are task-oriented but actually contain multiple layers of logic.
- Tests still partly follow historical file layout instead of current ownership.

## Target Shape

Suggested package structure:

```text
src/aba_optimiser/measurements/
    __init__.py
    preprocessing.py
    loading.py
    analysis.py
    reconstruction.py
    variances.py
    outputs.py
    pipeline.py
    online_knobs.py
    squeeze/
```

## Suggested Module Responsibilities

### `preprocessing.py`

Own:

- closed orbit subtraction
- kicker-aligned trimming
- turn renumbering after kick
- validation of orbit reference inputs

Keep this module pure:

- input `DataFrame` in
- output `DataFrame` out
- no filesystem access
- no controller knowledge

### `loading.py`

Own:

- reading SDDS / TBT files
- converting raw turn-by-turn objects into long-form dataframes
- file-to-bunch / file-to-dataframe index mapping

Move here from:

- `tbt_io.py`
- any related helpers currently imported through `create_datafile.py`

Potential rename:

- either replace `tbt_io.py`
- or keep `tbt_io.py` only as a very thin compatibility wrapper until deleted

### `analysis.py`

Own:

- hole-in-one / optics analysis execution
- extraction of bad BPMs from generated analysis files

This isolates all external analysis-tool coupling.

### `variances.py`

Own:

- uniform variance assignment
- known-noise variance assignment
- later: variance scaling policy after SVD cleaning

This removes “how we weight data” from reconstruction flow.

### `reconstruction.py`

Own:

- per-dataframe reconstruction flow
- AC-dipole model setup helpers
- call into `calculate_pz_measurement(...)`
- NaN handling policy after reconstruction

This should become the main home for what is currently `process_single_dataframe(...)`.

### `outputs.py`

Own:

- combining per-bunch outputs
- building per-file outputs
- attaching attrs/metadata
- writing parquet or packaging result dicts

This keeps formatting and persistence separate from reconstruction.

### `pipeline.py`

Own:

- top-level orchestration now in `process_measurements(...)`

This module should coordinate:

1. load raw measurements
2. preprocess
3. analyse optics if needed
4. reconstruct
5. package outputs

It should not contain detailed data munging itself.

## Recommended Refactor Sequence

### Phase 1

- Keep `preprocessing.py` as the pattern.
- Extract variance helpers into `variances.py`.
- Extract `process_single_dataframe(...)` into `reconstruction.py`.
- Leave `process_measurements(...)` in place but make it delegate.

This gives most of the cleanliness benefit with low API churn.

### Phase 2

- Move `run_analysis(...)` and bad-BPM extraction into `analysis.py`.
- Move output combining code into `outputs.py`.
- Rename `tbt_io.py` to `loading.py`.

At this point `create_datafile.py` should be mostly glue.

### Phase 3

- Replace `create_datafile.py` with `pipeline.py`.
- Delete old wrapper names instead of preserving backward compatibility.
- Update imports across the repo in one pass.

Since backward compatibility is not a priority for `0.0.1`, this can be a real cleanup instead of a long deprecation dance.

## API Direction

Prefer small explicit functions over large configurable ones.

Good:

- `load_measurement_files(...)`
- `preprocess_measurement_dataframe(...)`
- `assign_uniform_variances(...)`
- `reconstruct_measurement_dataframe(...)`
- `combine_reconstructed_file_outputs(...)`

Avoid:

- giant “do everything” functions with many optional flags
- modules that mix CLI/script concerns with library concerns

## Testing Strategy

Split tests by ownership:

- `test_preprocessing.py`
- `test_loading.py`
- `test_variances.py`
- `test_reconstruction.py`
- `test_pipeline.py`

Testing guidance:

- pure modules get focused unit tests
- orchestration modules get a few integration-style tests
- monkeypatch external tools at module boundaries, not deep inside mixed functions

## Specific Cleanups Worth Doing

- Rename `tws` arguments to `twiss` for consistency.
- Stop passing around `Path | str` internally once inside library code; normalize early.
- Replace broad “config-like” optional arguments with small dataclasses where the grouping is real.
- Keep attrs handling in one place instead of spreading it across pipeline stages.
- Move test helpers that depend on `tmom_recon` into a dedicated measurement test helper module.

## End State

Best case, the `measurements` package becomes:

- easy to scan
- mostly pure by default
- thin at the orchestration layer
- explicit about external dependencies
- easier to rewrite without dragging historical structure forward

