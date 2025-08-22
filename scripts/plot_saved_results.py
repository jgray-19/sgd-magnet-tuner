import matplotlib.pyplot as plt
import numpy as np

from aba_optimiser.config import (
    MAGNET_RANGE,
    OUTPUT_KNOBS,
    RUN_ARC_BY_ARC,
    SEQUENCE_FILE,
    TRUE_STRENGTHS,
)
from aba_optimiser.mad_interface import MadInterface
from aba_optimiser.utils import read_knobs, read_results

knob_names, knob_strengths, uncertainties = read_results(OUTPUT_KNOBS)
# Convert knob_strengths and uncertainties to numpy arrays
knob_strengths = np.array(knob_strengths)
uncertainties = np.array(uncertainties)

true_strengths = read_knobs(TRUE_STRENGTHS)
true_strengths = np.array([true_strengths[k] for k in knob_names])

# Calculate the relative error
relative_diff = (knob_strengths - true_strengths) / np.abs(true_strengths)
relative_uncertainties = np.abs(uncertainties) / np.abs(true_strengths)

mad_iface = MadInterface(SEQUENCE_FILE, MAGNET_RANGE)
initial_strengths = mad_iface.receive_knob_values()
elem_pos = mad_iface.elem_spos
if RUN_ARC_BY_ARC:
    true_strengths = {knob: true_strengths[knob] for knob in mad_iface.knob_names}
initial_rel_diff = (initial_strengths - true_strengths) / np.abs(true_strengths)

x = np.arange(len(knob_names))
width = 0.5  # width of the bars

fig, ax = plt.subplots(figsize=(12, 6))

# Plot the relative difference as a single bar without error bars set to relative uncertainties
# rects = ax.bar(x, relative_diff, width, color='mediumpurple')
rects = ax.bar(
    x,
    relative_diff,
    width,
    color="mediumpurple",
    label="Final Relative Difference",
    yerr=relative_uncertainties,
    capsize=5,
)
ax.bar(
    x,
    initial_rel_diff,
    width,
    color="green",
    label="Initial Relative Difference",
    alpha=0.1,
)

ax.set_xlabel("Knob Names")
ax.set_ylabel("Relative Difference")
ax.set_title("Relative Difference between Final and True Knob Strengths")
ax.set_xticks(x)
ax.set_xticklabels(knob_names, rotation=45, ha="right")
ax.legend()

plt.figure()
plt.plot(elem_pos, abs(relative_diff), "o", label="Relative Difference")
plt.xlabel("Element Position")
plt.ylabel("Value")
plt.title("Relative Difference vs Element Position")
plt.legend()
plt.grid()
plt.tight_layout()

plt.figure()
plt.plot(elem_pos, relative_uncertainties, "o", label="Uncertainty")
plt.xlabel("Element Position")
plt.ylabel("Value")
plt.title("Uncertainty vs Element Position")
plt.legend()
plt.grid()
plt.tight_layout()

error_on_quad = true_strengths - initial_strengths

plt.figure()
plt.plot(abs(error_on_quad), abs(knob_strengths - true_strengths), "x")
plt.xlabel("Absolute Error on Quadrupole Strengths at the Start")
plt.ylabel("Absolute Difference converged to at the end")
plt.title("Absolute Difference Results vs Initial Error on Quadrupole Strengths")
# Ensure the axes have the same limits and then plot a line y=x
max_point_both = max(max(abs(error_on_quad)), max(abs(knob_strengths - true_strengths)))
plt.xlim(0, max_point_both * 1.1)
plt.ylim(0, max_point_both * 1.1)
plt.plot([0, max_point_both], [0, max_point_both], "r--", label="y=x")
plt.grid()
plt.tight_layout()

plt.show()
