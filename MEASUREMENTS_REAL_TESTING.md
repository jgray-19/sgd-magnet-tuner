# Measurements Real Testing

This file records what I was able to test by actually running the refactored measurement code against real files already present in the repo or local environment.

I did this specifically to reduce reliance on monkeypatched unit tests.

At this point, `tests/measurements` no longer uses `monkeypatch`.

The SDDS round-trip path is now codified in:

- helper: `tests/measurements/helpers.py`
- integration test: `tests/measurements/test_loading_integration.py`

The reconstruction/preprocessing monkeypatch replacements are now codified in:

- helper: `tests/measurements/helpers.py`
- real reconstruction test: `tests/measurements/test_create_datafile_acd.py::test_process_single_dataframe_reconstructs_with_generated_analysis`
- real kick-detection test: `tests/measurements/test_create_datafile_acd.py::test_preprocess_measurement_dataframe_average_trims_from_kick`
- temp-root B2 error-table test: `tests/measurements/test_b2_errors.py::test_resolve_b2_error_table_picks_closest_energy`

## Summary

What I was able to run successfully:

- `load_measurement_files(...)`
- `convert_tbt_to_dataframes(...)`
- `assign_uniform_variances(...)`
- `assign_known_noise_variances(...)` with explicit NaN-variance patterns on PSB data
- `preprocess_measurement_dataframe(..., remove_closed_orbit="average")` on already aligned kicker data
- `preprocess_measurement_dataframe(..., remove_closed_orbit="average")` on a synthetic unaligned kick sequence using the real `find_kick(...)` path
- `preprocess_measurement_dataframe(..., remove_closed_orbit="twiss")`
- `process_single_dataframe(...)` end-to-end with a generated fake analysis folder

What I was able to run and confirm currently fails on real data:

- `assign_known_noise_variances(...)` on PSB kicker files without explicit exclusions
- `process_single_dataframe(..., use_uniform_vars=False)` on PSB kicker files before adding accelerator-specific noise config
- `process_single_dataframe(..., use_uniform_vars=True)` with the local `tests/data/model_creator` folder as analysis input
- `preprocess_measurement_dataframe(..., remove_closed_orbit="average")` on one unaligned kicker parquet

What I could not test properly yet:

- `run_measurement_analysis(...)`
- the full `process_measurements(...)` pipeline from raw SDDS through analysis and parquet output

## Files Used

Real local files used during testing:

- `tmp_kicker_probe/kicker.parquet`
- `tmp_kicker_afterfix/track.parquet`
- `tests/data/model_creator/psb3_twiss.dat`
- `tests/data/model_creator/psb3_twiss_ac.dat`
- generated during testing: `tmp_generated_measurement.sdds`

## Successful Real Runs

### 1. `load_measurement_files(...)` and `convert_tbt_to_dataframes(...)`

Ran on:

- a synthetic LHC-style SDDS file generated locally during testing
- output file: `tmp_generated_measurement.sdds`

How the file was produced:

- started from `tmp_kicker_probe/kicker.parquet`
- kept only BPM rows
- reshaped into `turn_by_turn.TransverseData`
- wrote to SDDS using `turn_by_turn.write_tbt(..., datatype="lhc")`

Then tested:

- `load_measurement_files([tmp_generated_measurement.sdds])`
- `build_dataframe_file_indices(...)`
- `convert_tbt_to_dataframes(...)`

Result:

- SDDS file loaded successfully through the refactored loader
- returned `1` measurement object with `1` matrix and `10` turns
- dataframe-file index mapping returned `[0]`
- converted dataframe shape matched the source exactly: `170` rows
- numeric agreement with the source data was effectively exact:
  - `max_abs_diff_x = 1.42e-17`
  - `max_abs_diff_y = 0.0`

Observed:

- this is a proper non-mock end-to-end check of the new `loading.py` path
- it confirms the refactored loader can read an actual SDDS written in the expected format
- this path is now automated as a repeatable integration test instead of being ad hoc

### 1. `assign_uniform_variances(...)`

Ran on:

- `tmp_kicker_probe/kicker.parquet`

Result:

- function executed successfully
- returned the expected variance columns
- correctly set `inf` on a chosen bad BPM row

Observed:

- this path works fine on the local PSB-style parquet data

### 2. `assign_known_noise_variances(...)` with explicit NaN-variance patterns

Ran on:

- `tmp_kicker_probe/kicker.parquet`

Arguments:

- `accelerator_type="psb"`
- `nan_variance_patterns=[r"^BI3\.KSW1L4$", r"^BR3\.BPMT3L1$"]`

Result:

- function executed successfully
- rows matching the explicit patterns received `NaN` in `var_x` and `var_y`
- standard PSB BPM rows such as `BR3.BPM2L3` received finite known-noise variances

Observed:

- this solves the real “special non-BPM row” problem when the caller knows which names should be excluded from noise lookup
- the accelerator-specific noise table also matters: PSB data should use `accelerator_type="psb"`, not `"lhc"`

### 2. `preprocess_measurement_dataframe(..., remove_closed_orbit="average")` on already aligned kicker data

Ran on:

- `tmp_kicker_probe/kicker.parquet`

Arguments:

- `remove_closed_orbit="average"`
- `kicker_name="BI3.KSW1L4"`

Result:

- function executed successfully
- it detected the data was already kicker-aligned
- output length matched input length
- first row remained the kicker marker `BI3.KSW1L4`

Observed:

- the “skip if already aligned” path works on real local kicker data

### 3. `preprocess_measurement_dataframe(..., remove_closed_orbit="twiss")`

Ran on:

- `tmp_kicker_probe/kicker.parquet`
- twiss reference from `tests/data/model_creator/psb3_twiss.dat`

Result:

- function executed successfully
- output had the same row count as input
- BPMs missing in the reference produced `NaN` after subtraction

Observed:

- this is expected for names such as kicker markers that are not present in the reference table
- this path works mechanically, but the usefulness depends on the reference containing all relevant rows

### 4. `preprocess_measurement_dataframe(..., remove_closed_orbit="average")` on a synthetic unaligned kick sequence

Ran through the automated test:

- `tests/measurements/test_create_datafile_acd.py::test_preprocess_measurement_dataframe_average_trims_from_kick`

Setup:

- a small in-memory dataframe with three BPMs and five turns
- the first three turns are flat closed orbit
- the kick first appears at `BPM2` on turn `4`

Result:

- the real `find_kick(...)` path identified `("BPM2", 4)`
- preprocessing trimmed away upstream rows on the kick turn
- turns were renumbered to start at `1`
- the remaining data started at `BPM2`, as expected

Observed:

- this replaced an older monkeypatched unit test with a real execution of the kicker-detection path

### 5. `process_single_dataframe(...)` end-to-end with generated fake analysis files

Ran through the automated test:

- `tests/measurements/test_create_datafile_acd.py::test_process_single_dataframe_reconstructs_with_generated_analysis`

Setup:

- generated a temporary analysis directory from `tests/data/model_creator/psb3_twiss.dat`
- used `omc3.scripts.fake_measurement_from_model.generate(...)` with:
  - `BETX`
  - `BETY`
  - `PHASEX`
  - `PHASEY`
  - `X`
  - `Y`
- reconstructed a two-turn synthetic measurement against `tests/data/model_creator/psb3_twiss_ac.dat`

Result:

- `process_single_dataframe(...)` completed successfully
- returned finite `px` and `py`
- returned all expected variance columns
- no monkeypatching was needed for SVD cleaning, variance assignment, or momentum reconstruction

Observed:

- the previous monkeypatched reconstruction test has now been replaced by a real end-to-end reconstruction test
- the generated analysis folder is minimal but sufficient for this path because dispersion files are optional in `tmom_recon`

## Real Failures Observed

### 1. `assign_known_noise_variances(...)` fails on PSB kicker files without explicit exclusions

Ran on:

- `tmp_kicker_probe/kicker.parquet`

Failure:

```text
ValueError: No noise variance found for BPM BI3.KSW1L4
```

Interpretation:

- this fails when special rows such as kicker markers are sent into the noise lookup without telling the code how to treat them
- after the refactor, the helper can now be told to assign `NaN` variances to those names explicitly

### 2. `process_single_dataframe(..., use_uniform_vars=False)` initially failed on PSB kicker files because of noise lookup configuration

Ran on:

- `tmp_kicker_afterfix/track.parquet`
- twiss from `tests/data/model_creator/psb3_twiss_ac.dat`
- `analysis_dir=tests/data/model_creator`

Initial failure:

```text
ValueError: No noise variance found for BPM BR3.BPM1L3
```

Follow-up real run after adding:

- `accelerator_type="psb"`
- `nan_variance_patterns=[r"^BI3\.KSW1L4$", r"^BR3\.BPMT3L1$"]`

New result:

```text
FileNotFoundError: Measurement file not found: .../tests/data/model_creator/beta_amplitude_x.tfs
```

Interpretation:

- the noise-model problem is now bypassed correctly for this PSB-style file
- the next blocker is no longer variance assignment
- the next blocker is still the incomplete analysis fixture

### 3. `process_single_dataframe(..., use_uniform_vars=True)` failed with incomplete local analysis fixture

Ran on:

- `tmp_kicker_afterfix/track.parquet`
- twiss from `tests/data/model_creator/psb3_twiss_ac.dat`
- `analysis_dir=tests/data/model_creator`

Failure:

```text
FileNotFoundError: Measurement file not found: .../tests/data/model_creator/beta_amplitude_x.tfs
```

Interpretation:

- `calculate_pz_measurement(...)` expects a fuller measurement-analysis folder than the current local fixture provides
- the present `tests/data/model_creator` directory is not enough to exercise the real reconstruction path end-to-end

### 4. `preprocess_measurement_dataframe(..., remove_closed_orbit="average")` failed on one unaligned kicker parquet

Ran on:

- `tmp_kicker_afterfix/track.parquet`
- `remove_closed_orbit="average"`
- `n_turns_free=1`

Failure:

```text
ValueError: No kicks found above the specified threshold.
```

Interpretation:

- either this parquet is already post-processed in a way that breaks the `find_kick(...)` heuristic
- or `n_turns_free=1` is not a sensible setting for this file
- this needs a better representative raw example if we want to validate the “detect kick and trim” branch properly

## What I Could Not Test Properly Yet

### `run_measurement_analysis(...)`

I did not run this directly because it depends on the external `hole_in_one` analysis flow and a realistic raw measurement setup.

What I need:

- one small known-good raw measurement fixture set
- or a reduced saved analysis fixture plus a documented command showing how to regenerate it

### Full `process_measurements(...)` pipeline

I did not run the full raw-to-parquet pipeline because both missing pieces above are required:

- raw SDDS input
- a complete analysis directory that satisfies `calculate_pz_measurement(...)`

## Help Needed

If you want stronger real testing coverage, the most useful additions would be:

### 1. A tiny committed raw SDDS fixture

Needed for:

- `process_measurements(...)`

Ideal properties:

- very small
- deterministic
- documented source

This is less urgent now because I was able to generate a temporary SDDS locally during testing, but a stable helper or script for reproducing that generation would still help.

### 2. A complete committed measurement-analysis fixture

Needed for:

- `process_single_dataframe(...)`
- `process_measurements(...)`

Specifically, a folder that includes whatever `calculate_pz_measurement(...)` expects, such as:

- `beta_amplitude_x.tfs`
- related optics / phase / amplitude files

### 3. One documented “good” PSB or LHC example for known-noise variances

Needed for:

- `assign_known_noise_variances(...)`
- `process_single_dataframe(..., use_uniform_vars=False)`

Right now the local PSB kicker parquet is useful for testing preprocessing, but it is not compatible with the current LHC-only noise lookup.

## Notes From This Pass

- I fixed stale imports in:
  - `scripts/plot_single_bpm_svd_effect.py`
  - `scripts/plot_acdipole_cleaning_comparison.py`

- The current code appears structurally cleaner after the refactor, but real-data compatibility still depends heavily on:
  - accelerator-specific BPM naming
  - the availability of full measurement-analysis folders
  - whether the input is raw, aligned, or already partially post-processed
