"""Backward-compatibility shim — all content has moved to plotting/core.py."""

from aba_optimiser.measurements.plotting.core import (  # noqa: F401
    ARC56_OFFSETS_FILE,
    BEST_KNOWLEDGE_LABEL,
    BETTER_KNOWLEDGE_LABEL,
    DESIGN_OPTICS_LABEL,
    MEASUREMENT_LABEL,
    PLOT_COLORS,
    EstimateSource,
    PlotContext,
    _normalize_phase,
    add_ip_positions_to_plot,
    convert_estimates_to_optimisation_space,
    convert_uncertainties_to_optimisation_space,
    filter_estimates_by_max_uncertainty,
    find_true_values,
    get_arc_ranges,
    get_element_positions,
    get_fullring_twiss,
    get_ip_positions,
    get_measurement_phase_through_arc,
    get_twiss_through_arc,
    get_twiss_without_errors,
    load_estimates_from_checkpoints,
    load_model_metadata,
    main,
    parse_arc_spec,
    plot_fullring_comparison,
    plot_phase_advances,
    plot_quad_diffs,
    prepare_plot_context,
)

if __name__ == "__main__":
    main()
