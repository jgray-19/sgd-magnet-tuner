# optimise_knobs.py

import logging
import multiprocessing as mp

from aba_optimiser.controller import Controller

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mp.set_start_method("spawn")
    controller = Controller()
    controller.run()
