# optimise_knobs.py

import logging
import multiprocessing as mp

from aba_optimiser.training.controller import Controller

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mp.set_start_method("fork")
    controller = Controller(optimise_sextupoles=False, show_plots=False)
    controller.run()

    del controller

    controller = Controller(optimise_sextupoles=True, show_plots=True)
    controller.run()
