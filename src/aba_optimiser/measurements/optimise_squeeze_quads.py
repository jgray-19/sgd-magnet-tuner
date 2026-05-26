"""Entry point for LHC squeeze quadrupole optimisation.

All logic lives in the squeeze/ sub-package. This module re-exports the
public API for backward compatibility and provides the CLI entry point.
"""

from aba_optimiser.measurements.squeeze import (  # noqa: F401
    MEAS_TIMES,
    MINUS_50HZ,
    MINUS_100HZ,
    MINUS_150HZ,
    MINUS_200HZ,
    MINUS_250HZ,
    MINUS_300HZ,
    MINUS_350HZ,
    PLUS_50HZ,
    PLUS_100HZ,
    PLUS_150HZ,
    PLUS_200HZ,
    PLUS_250HZ,
    PLUS_300HZ,
    PLUS_350HZ,
    ZEROHZ,
    ACDipoleOptimisationWindow,
    create_configs,
    get_ac_dipole_bpm_points,
    get_beam_paths,
    get_default_simulation_config,
    get_knob_files,
    get_measurement_time,
    get_sequence_creation_time,
    load_bad_bpms,
    load_measurements_from_reload,
    load_metadata,
    optimise_arc,
    prepare_frequency_metadata,
    process_frequency_results,
    process_measurements_fresh,
    process_squeeze_step,
    resolve_restore_resume,
    save_arc_estimates,
    save_bad_bpms,
    save_metadata,
    update_metadata,
    validate_processed_files,
    window_from_attrs,
)
from aba_optimiser.measurements.squeeze.pipeline import main

if __name__ == "__main__":
    main()
