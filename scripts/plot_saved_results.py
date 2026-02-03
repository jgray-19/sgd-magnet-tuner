import numpy as np

# from aba_optimiser.plotting.strengths import (
#     plot_strengths_comparison,
#     plot_strengths_vs_position,
# )
from aba_optimiser.accelerators import LHC
from aba_optimiser.config import (
    OUTPUT_KNOBS,
    TRUE_STRENGTHS_FILE,
)
from aba_optimiser.io.utils import get_lhc_file_path, read_knobs, read_results
from aba_optimiser.mad import GenericMadInterface
from aba_optimiser.training.result_manager import ResultManager

MAGNET_RANGE = "BPM.11R2.B1/BPM.11L3.B1"
knob_names, final_vals, uncertainties = read_results(str(OUTPUT_KNOBS))
accelerator = LHC(beam=1, beam_energy=6800, sequence_file=get_lhc_file_path(1))
mad_iface = GenericMadInterface(accelerator, magnet_range=MAGNET_RANGE)
results = ResultManager(knob_names, mad_iface.elem_spos, accelerator)

knobs_from_file = dict(zip(knob_names, final_vals))


true_strengths = read_knobs(TRUE_STRENGTHS_FILE)
true_vals = np.array([true_strengths[k] for k in knob_names])


abs_all = sum(abs(final_vals - true_vals))
abs_5_end = sum(abs(final_vals[5:] - true_vals[5:]))
abs_start_m5 = sum(abs(final_vals[:-5] - true_vals[:-5]))
abs_5_m5 = sum(abs(final_vals[5:-5] - true_vals[5:-5]))

print("- Sum of absolute differences")
print(f"    - all elements: {abs_all:.2e}")
print(f"    - elements 5 to end: {abs_5_end:.2e}")
print(f"    - elements start to -5: {abs_start_m5:.2e}")
print(f"    - elements 5 to -5: {abs_5_m5:.2e}")


# Relative differences:
rel_all = sum(abs((final_vals - true_vals) / true_vals))
rel_5_end = sum(abs((final_vals[5:] - true_vals[5:]) / true_vals[5:]))
rel_start_m5 = sum(abs((final_vals[:-5] - true_vals[:-5]) / true_vals[:-5]))
rel_5_m5 = sum(abs((final_vals[5:-5] - true_vals[5:-5]) / true_vals[5:-5]))

print("- Sum of relative differences")
print(f"    - all elements: {rel_all:.2e}")
print(f"    - elements 5 to end: {rel_5_end:.2e}")
print(f"    - elements start to -5: {rel_start_m5:.2e}")
print(f"    - elements 5 to -5: {rel_5_m5:.2e}")

initial_strengths = np.array([true_strengths[k] for k in knob_names])

results.generate_plots(
    current_knobs=knobs_from_file,
    initial_strengths=initial_strengths,
    true_strengths=true_strengths,
    uncertainties=np.array(uncertainties),
)
