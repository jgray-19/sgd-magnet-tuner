# optimise_knobs.py

# import dataclasses
import logging
import multiprocessing as mp

from aba_optimiser.config import DPP_OPTIMISER_CONFIG, DPP_SIMULATION_CONFIG, QUAD_OPTIMISER_CONFIG, QUAD_SIMULATION_CONFIG
from aba_optimiser.training.controller import Controller

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mp.set_start_method("fork")

    # Step 1: Optimise energy (dp/p) first
    logging.info("Starting energy optimisation...")
    energy_controller = Controller(
        optimiser_config=DPP_OPTIMISER_CONFIG, simulation_config=DPP_SIMULATION_CONFIG, show_plots=True, initial_knob_strengths={"deltap": 1e-6}
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
