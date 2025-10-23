"""
Plot the average uncertainty of the model predictions, as the number of files increases.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aba_optimiser.config import (
    ACD_ON,
    FLATTOP_TURNS,
    MAGNET_RANGE,
    NO_NOISE_FILE,
    NUM_WORKERS,
    RAMP_UP_TURNS,
    SEQUENCE_FILE,
    TRUE_STRENGTHS_FILE,
)
from aba_optimiser.dataframes.utils import (
    filter_out_marker,
    select_markers,
)
from aba_optimiser.io.utils import read_knobs
from aba_optimiser.mad.optimising_mad_interface import OptimisationMadInterface
from aba_optimiser.workers.worker import build_worker

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

run_start = time.time()  # start total timing
start_bpm, end_bpm = MAGNET_RANGE.split("/")

track_data = pd.read_parquet(NO_NOISE_FILE)
start_data = select_markers(track_data, start_bpm)
comparison_data = track_data.copy()
if "BPM" not in start_bpm:
    comparison_data = filter_out_marker(comparison_data, start_bpm)
if "BPM" not in end_bpm:
    comparison_data = filter_out_marker(comparison_data, end_bpm)

mad_iface = OptimisationMadInterface(SEQUENCE_FILE, MAGNET_RANGE)
tws = mad_iface.run_twiss()
start_bpm_idx = tws.index.get_loc(start_bpm)
end_bpm_idx = tws.index.get_loc(end_bpm)
tws = tws.iloc[start_bpm_idx : end_bpm_idx + 1]

beta_x = np.sqrt(tws["beta11"].to_numpy())
beta_y = np.sqrt(tws["beta22"].to_numpy())

uncertainty_list = []
num_files_list = [10, 40, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 6e4]
num_files_list = [int(num_files) for num_files in num_files_list]

true_strengths = read_knobs(TRUE_STRENGTHS_FILE)

for num_files in num_files_list:
    assert num_files <= FLATTOP_TURNS, (
        f"Number of files ({num_files}) must be less than or equal to the total number of tracks "
        f"({FLATTOP_TURNS})."
    )
    # Prepare worker batches
    if ACD_ON:
        indices = list(range(RAMP_UP_TURNS, num_files + RAMP_UP_TURNS + 1))
    else:
        indices = list(range(num_files + 1))

    num_workers = min(NUM_WORKERS, num_files)
    tracks_per_worker = num_files // num_workers
    batches = [
        indices[i * tracks_per_worker : (i + 1) * tracks_per_worker]
        for i in range(num_workers)
    ]
    for i, batch in enumerate(batches):
        assert batch, f"Batch {i} is empty. Check TRACKS_PER_WORKER and NUM_WORKERS."

    # Start worker processes
    parent_conns: list[Connection] = []
    workers: list[mp.Process] = []
    for i, batch in enumerate(batches):
        parent, child = mp.Pipe()
        w = build_worker(i, batch, child, comparison_data, start_data, beta_x, beta_y)
        w.start()
        parent_conns.append(parent)
        workers.append(w)

    H_global = np.zeros((len(mad_iface.knob_names), len(mad_iface.knob_names)))
    for conn in parent_conns:
        H_global += conn.recv()

    cov = np.linalg.inv(H_global)
    comb_uncertainty = np.sqrt(np.diag(cov))

    for w in workers:
        w.join()

    strength_list = np.array([true_strengths[k] for k in mad_iface.knob_names])
    rel_uncertainty = np.abs(comb_uncertainty / strength_list)
    uncertainty_list.append(np.mean(rel_uncertainty))
    print(f"Relative uncertainty for {num_files} files: {uncertainty_list[-1]:.4e}")

del mad_iface
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(num_files_list, uncertainty_list, marker="o", linestyle="", label="Data")

# Perform a power law fit: uncertainty = a * (num_files)^b.
log_x = np.log(np.array(num_files_list))
log_y = np.log(np.array(uncertainty_list))
slope, intercept = np.polyfit(log_x, log_y, 1)
a = np.exp(intercept)
b = slope

# Generate fitted curve for plotting
x_fit = np.linspace(min(num_files_list), max(num_files_list), 100)
y_fit = a * x_fit**b
plt.plot(x_fit, y_fit, color="red", label=f"Fit: y = {a:.2e} * x^{b:.2f}")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of Files (log scale)")
plt.ylabel("Relative Uncertainty")
plt.title("Relative Uncertainty vs Number of Files")
plt.grid()
plt.legend()
plt.savefig("relative_uncertainty_vs_num_files.png")
plt.show()
plt.close()
print(f"Total run time: {time.time() - run_start:.2f} seconds")
