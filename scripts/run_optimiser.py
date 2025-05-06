# optimise_knobs.py

import multiprocessing as mp
from aba_optimiser.controller import Controller

if __name__ == "__main__":
    mp.set_start_method("spawn")
    controller = Controller()
    controller.run()
