from matplotlib import pyplot as plt

from aba_optimiser.config import TRUE_STRENGTHS_FILE
from aba_optimiser.io.utils import get_lhc_file_path, read_knobs
from aba_optimiser.mad.optimising_mad_interface import OptimisationMadInterface

MAGNET_RANGES = [f"BPM.11R{i}.B1/BPM.11L{i % 8 + 1}.B1" for i in range(1, 9)]
mad = OptimisationMadInterface(
    sequence_file=get_lhc_file_path(beam=1), bpm_pattern="BPM", corrector_strengths=None, tune_knobs_file=None
)
mad.mad["full_tws", "mflw"] = mad.mad.twiss(
    sequence="loaded_sequence", observe=1, savemap=True
)

true_strengths = read_knobs(TRUE_STRENGTHS_FILE)
plt.figure(figsize=(12, 8))
plt.plot(mad.mad.full_tws["s"], mad.mad.full_tws["beta11"], label="Beta X")
plt.plot(mad.mad.full_tws["s"], mad.mad.full_tws["beta22"], label="Beta Y")
plt.xlabel("Position (m)")
plt.ylabel("Beta function (m)")
plt.legend()
plt.title("LHC Beta Functions")
plt.grid()


for magnet_range in MAGNET_RANGES:
    start_bpm, end_bpm = magnet_range.split("/")
    mad.mad.send("full_tws['BPM.11R1.B1']['__map']:print()")
    x0 = mad.mad.full_tws[start_bpm]["__map"]
    x0["Aset"] = True

    tws, _ = mad.mad.twiss(
        sequence="loaded_sequence", observe=1, range=f"'{magnet_range}'", X0=x0
    )
    plt.figure(figsize=(10, 6))
    plt.plot(tws["s"], tws["beta11"], label="Beta X")
    plt.plot(tws["s"], tws["beta22"], label="Beta Y")
    plt.xlabel("Position (m)")
    plt.ylabel("Beta function (m)")
    plt.legend()
    plt.title(f"LHC Beta Functions - {magnet_range}")
    plt.grid()

    # print(f"Plotted beta functions for range: {magnet_range}")
    print(
        f"Max-min beta X: {tws['beta11'].max():.2f} - {tws['beta11'].min():.2f} = {tws['beta11'].max() - tws['beta11'].min():.2f}"
    )
    print(
        f"Max-min beta Y: {tws['beta22'].max():.2f} - {tws['beta22'].min():.2f} = {tws['beta11'].max() - tws['beta11'].min():.2f}"
    )
    print()

    tws, _ = mad.mad.twiss(
        sequence="loaded_sequence", observe=0, range=f"'{magnet_range}'", X0=x0
    )
    arc_elements = list(tws.ename)
    strengths_in_arc = {
        k: v for k, v in true_strengths.items() if k.strip("_k") in arc_elements
    }
    diffs = {}
    for k, v in strengths_in_arc.items():
        if k[:2] == "MB":
            k_index = 0
        # elif k[:2] == "MQ":
        #     k_index = 1
        # elif k[:2] == "MS":
        #     k_index = 2
        else:
            continue
        model_strength = mad.mad.recv_vars(f'MADX["{k.strip("_k")}"].k{k_index}')
        diffs[k] = v - model_strength

    # Print mean and std of the differences
    mean_diff = sum(diffs.values()) / len(diffs) if diffs else 0
    std_diff = (
        (sum((d - mean_diff) ** 2 for d in diffs.values()) / len(diffs)) ** 0.5
        if diffs
        else 0
    )
    print(f"Mean difference: {mean_diff:.3e}, Std: {std_diff:.3e} for {magnet_range}")

    # Now plot the differences as a bar chart
    # plt.figure(figsize=(12, 6))
    # plt.bar(diffs.keys(), diffs.values())
    # plt.xticks(rotation=90)
    # plt.ylabel("Strength Difference")
    # plt.title(
    #     f"Magnet Strength Differences in {magnet_range}: Avg {sum(diffs.values()) / len(diffs):.6f}"
    # )
    # plt.grid()
    # plt.tight_layout()

plt.show()
