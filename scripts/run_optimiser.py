# optimise_knobs.py

# import dataclasses
import logging
import multiprocessing as mp

from aba_optimiser.config import (
    DPP_OPTIMISER_CONFIG,
    DPP_SIMULATION_CONFIG,
    QUAD_OPTIMISER_CONFIG,
    QUAD_SIMULATION_CONFIG,
)
from aba_optimiser.training.controller import Controller
from aba_optimiser.training.controller_config import BPMConfig, MeasurementConfig, SequenceConfig

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mp.set_start_method("fork")

    # Step 1: Optimise energy (dp/p) first
    logging.info("Starting energy optimisation...")
    # energy_controller = Controller(DPP_OPT_SETTINGS, show_plots=True)
    # energy_knobs = energy_controller.run()
    # del energy_controller

    # Step 2: Optimise quadrupoles using energy results as starting point
    logging.info("Starting quadrupole optimisation with energy results...")

    # Create config objects (placeholder - adjust based on actual data)
    sequence_config = SequenceConfig.for_lhc_beam(
        beam=1,  # Adjust as needed
        magnet_range="your_magnet_range",  # Specify your magnet range
    )

    measurement_config = MeasurementConfig(
        measurement_files="your_measurement_files",  # Specify your measurement files
    )

    bpm_config = BPMConfig(
        start_points=["your_start_bpms"],  # Specify your BPM start points
        end_points=["your_end_bpms"],  # Specify your BPM end points
    )

    quad_controller = Controller(
        optimiser_config=QUAD_OPTIMISER_CONFIG,
        simulation_config=QUAD_SIMULATION_CONFIG,
        sequence_config=sequence_config,
        measurement_config=measurement_config,
        bpm_config=bpm_config,
        show_plots=True,
        # initial_knob_strengths=energy_knobs
    )
    quad_knobs, _ = quad_controller.run()
    del quad_controller, _

    # quad_opts_with_sextupole_data = dataclasses.replace(
    #     QUAD_OPT_SETTINGS, use_sextupole_data=True
    # )

    # Step 3: Optimise sextupoles using quadrupole results as starting point
    # logging.info("Starting sextupole optimisation with quadrupole results...")
    # sext_controller = Controller(SEXT_OPT_SETTINGS, show_plots=True, initial_knob_strengths=quad_knobs)
    # final_knobs = sext_controller.run()
    # del sext_controller

    logging.info("All optimisation stages completed!")
    # logging.info(f"Final optimised knobs: {len(final_knobs)} parameters")
    logging.info(f"Final optimised knobs: {len(quad_knobs)} parameters")
