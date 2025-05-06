from aba_optimiser.utils import read_results, read_knobs, read_elem_names
from aba_optimiser.config import OUTPUT_KNOBS, TRUE_STRENGTHS, ELEM_NAMES_FILE

import numpy as np
import matplotlib.pyplot as plt

elem_pos, _ = read_elem_names(ELEM_NAMES_FILE)
knob_names, knob_strengths, uncertainties = read_results(OUTPUT_KNOBS)
# Convert knob_strengths and uncertainties to numpy arrays
knob_strengths = np.array(knob_strengths)
uncertainties = np.array(uncertainties)

true_strengths = read_knobs(TRUE_STRENGTHS)
true_strengths = np.array([true_strengths[k] for k in knob_names])

# Calculate the relative error
relative_diff = (knob_strengths - true_strengths) / np.abs(true_strengths)
relative_uncertainties = np.abs(uncertainties) / np.abs(true_strengths)

x = np.arange(len(knob_names))
width = 0.5  # width of the bars

fig, ax = plt.subplots(figsize=(12, 6))

# Plot the relative difference as a single bar without error bars set to relative uncertainties
# rects = ax.bar(x, relative_diff, width, color='mediumpurple')
rects = ax.bar(x, relative_diff, width, color='mediumpurple', yerr=relative_uncertainties, capsize=5)

ax.set_xlabel('Knob Names')
ax.set_ylabel('Relative Difference')
ax.set_title('Relative Difference between Final and True Knob Strengths')
ax.set_xticks(x)
ax.set_xticklabels(knob_names, rotation=45, ha='right')

plt.figure()
plt.plot(elem_pos, abs(relative_diff), 'o', label='Relative Difference')
plt.xlabel('Element Position')
plt.ylabel('Value')
plt.title('Relative Difference vs Element Position')
plt.legend()
plt.grid()
plt.tight_layout()

plt.figure()
plt.plot(elem_pos, relative_uncertainties, 'o', label='Uncertainty')
plt.xlabel('Element Position')
plt.ylabel('Value')
plt.title('Uncertainty vs Element Position')
plt.legend()
plt.grid()
plt.tight_layout()

plt.show()
