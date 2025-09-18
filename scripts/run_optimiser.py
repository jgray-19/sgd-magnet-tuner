# optimise_knobs.py

# import dataclasses
import logging
import multiprocessing as mp

from aba_optimiser.config import (
    DPP_OPT_SETTINGS,
    QUAD_OPT_SETTINGS,
    # SEXT_OPT_SETTINGS
)
from aba_optimiser.training.controller import Controller

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mp.set_start_method("fork")

    # Step 1: Optimize energy (dp/p) first
    logging.info("Starting energy optimization...")
    # energy_controller = Controller(DPP_OPT_SETTINGS, show_plots=True)
    # energy_knobs = energy_controller.run()
    # del energy_controller

    # Step 2: Optimize quadrupoles using energy results as starting point
    logging.info("Starting quadrupole optimization with energy results...")
    quad_controller = Controller(
        QUAD_OPT_SETTINGS,
        show_plots=True,
        # initial_knob_strengths=energy_knobs
    )
    quad_knobs = quad_controller.run()
    del quad_controller

    # quad_opts_with_sextupole_data = dataclasses.replace(
    #     QUAD_OPT_SETTINGS, use_sextupole_data=True
    # )

    # Step 3: Optimize sextupoles using quadrupole results as starting point
    # logging.info("Starting sextupole optimization with quadrupole results...")
    # sext_controller = Controller(SEXT_OPT_SETTINGS, show_plots=True, initial_knob_strengths=quad_knobs)
    # final_knobs = sext_controller.run()
    # del sext_controller

    logging.info("All optimization stages completed!")
    # logging.info(f"Final optimized knobs: {len(final_knobs)} parameters")
    logging.info(f"Final optimized knobs: {len(quad_knobs)} parameters")
