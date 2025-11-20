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

    energy_controller = Controller(
        optimiser_config=DPP_OPTIMISER_CONFIG,
        simulation_config=DPP_SIMULATION_CONFIG,
        sequence_config=sequence_config,
        measurement_config=measurement_config,
        bpm_config=bpm_config,
        show_plots=True,
        initial_knob_strengths={"deltap": 1e-6},
    )
    energy_knobs, _ = energy_controller.run()
    del energy_controller
    logging.info("Energy optimisation completed!")

    logging.info("Starting quadrupole optimization with energy results...")
    # quad_controller = Controller(
    #     QUAD_OPT_SETTINGS,
    #     show_plots=True,  # initial_knob_strengths=energy_knobs
    # )
    # quad_knobs, _ = quad_controller.run()
    # del quad_controller
