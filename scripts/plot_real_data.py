from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tfs
from omc3.model.constants import TWISS_AC_DAT
from turn_by_turn import read_tbt

from aba_optimiser.config import module_path
from aba_optimiser.dataframes.utils import select_markers
from aba_optimiser.momentum_recon.transverse import calculate_pz

# from lhcng.model import model_to_ng
from aba_optimiser.physics.phase_space import PhaseSpaceDiagnostics

# from lhcng.tfs_utils import convert_tfs_to_madx

real_data_list = [
    "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-09/LHCB1/Measurements/Beam1@BunchTurn@2025_04_09@18_35_18_583/Beam1@BunchTurn@2025_04_09@18_35_18_583.sdds",
    "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-09/LHCB1/Measurements/Beam1@BunchTurn@2025_04_09@18_34_14_500/Beam1@BunchTurn@2025_04_09@18_34_14_500.sdds",
    "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-09/LHCB1/Measurements/Beam1@BunchTurn@2025_04_09@18_36_22_581/Beam1@BunchTurn@2025_04_09@18_36_22_581.sdds",
    # '/nfs/cs-ccr-nfs4/lhc_data/OP_DATA/FILL_DATA/10533/BPM/Beam1@BunchTurn@2025_04_27@13_50_14_431.sdds',
    # '/nfs/cs-ccr-nfs4/lhc_data/OP_DATA/FILL_DATA/10533/BPM/Beam1@BunchTurn@2025_04_27@13_51_22_434.sdds',
    # '/nfs/cs-ccr-nfs4/lhc_data/OP_DATA/FILL_DATA/10533/BPM/Beam1@BunchTurn@2025_04_27@13_52_31_321.sdds',
]

BPM_START = "BPM.12R4.B1"
Y_BPM_START = "BPM.11R4.B1"
num_files = len(real_data_list)

raw_data = [read_tbt(file).matrices[0] for file in real_data_list]
x = [data.X for data in raw_data]
y = [data.Y for data in raw_data]
for i in range(num_files):
    assert BPM_START in x[i].index
    assert Y_BPM_START in y[i].index

model_dir = Path(
    "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-09/LHCB1/Models/b1_flat_60_18cm"
    # '/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-27/LHCB1/Models/b1_60cm_injTunes'
)
ng_model_dir = module_path / model_dir.name
# model_to_ng(model_dir, beam=1, out_dir=ng_model_dir)

tws = tfs.read(ng_model_dir / TWISS_AC_DAT, index="NAME")

# Rename the columns BETX -> beta11,  BETY -> beta22, MUX -> mu1, MUY -> mu2, ALFX -> alfa11, ALFY -> alfa22
tws.rename(
    columns={
        "BETX": "beta11",
        "BETY": "beta22",
        "MUX": "mu1",
        "MUY": "mu2",
        "ALFX": "alfa11",
        "ALFY": "alfa22",
    },
    inplace=True,
)

tws.headers = {key.lower(): value for key, value in tws.headers.items()}

matching_bpms_list = []
real_data = []
# Combine the X and Y data into a single DataFrame
for i in range(num_files):
    # Find matching BPMs
    matching_bpms = tws.index.intersection(x[i].index).intersection(y[i].index)
    matching_bpms_list.append(matching_bpms)
    x[i] = x[i].loc[matching_bpms]
    y[i] = y[i].loc[matching_bpms]

    # Vectorised melt for x and y
    x_melt = x[i].reset_index().melt(id_vars="index", var_name="turn", value_name="x")
    y_melt = y[i].reset_index().melt(id_vars="index", var_name="turn", value_name="y")

    # Merge x and y on BPM name and turn
    df = pd.merge(x_melt, y_melt, on=["index", "turn"], suffixes=("_x", "_y"))
    df.rename(columns={"index": "name"}, inplace=True)
    df["x"] = df["x"]  # Convert to meters
    df["y"] = df["y"]  # Convert to meters

    real_data.append(tfs.TfsDataFrame(df[["name", "turn", "x", "y"]]))
    print(real_data[i])

plt.figure(figsize=(18, 6))
plt.suptitle("All Real Data - Phase Space Plots")

data_p_list = []
data_n_list = []
for i in range(num_files):
    data_p, data_n, _ = calculate_pz(
        orig_data=real_data[i],
        inject_noise=False,
        tws=tws.loc[matching_bpms_list[i]],
        info=False,
    )
    data_p_list.append(data_p)
    data_n_list.append(data_n)

# X phase space
plt.subplot(1, 2, 1)
for i in range(num_files):
    select_p = select_markers(data_p_list[i], BPM_START)
    select_n = select_markers(data_n_list[i], BPM_START)
    plt.scatter(
        select_p["x"],
        select_p["px"],
        label=f"Data P {i + 1}",
        alpha=0.5,
        s=1,
    )
    plt.scatter(
        select_n["x"],
        select_n["px"],
        label=f"Data N {i + 1}",
        alpha=0.5,
        s=1,
    )
plt.title("X BPM Start - Phase Space Plot")
plt.xlabel("X Position")
plt.ylabel("PX")
plt.legend()

# Y phase space
plt.subplot(1, 2, 2)
for i in range(num_files):
    select_p = select_markers(data_p_list[i], Y_BPM_START)
    select_n = select_markers(data_n_list[i], Y_BPM_START)
    plt.scatter(
        select_p["y"],
        select_p["py"],
        label=f"Data P {i + 1}",
        alpha=0.5,
        s=1,
    )
    plt.scatter(
        select_n["y"],
        select_n["py"],
        label=f"Data N {i + 1}",
        alpha=0.5,
        s=1,
    )
plt.title("Y BPM Start - Phase Space Plot")
plt.xlabel("Y Position")
plt.ylabel("PY")
plt.legend()

plt.tight_layout()

# plt.show()

# Prepare ellipse for X_BPM_START using the first real data set
x_bpm_data = select_markers(data_p_list[0], BPM_START)
ps_diag = PhaseSpaceDiagnostics(
    bpm=BPM_START,
    x_data=x_bpm_data["x"],
    px_data=x_bpm_data["px"],
    y_data=x_bpm_data["y"],
    py_data=x_bpm_data["py"],
    # tws=tws,
    analysis_dir="/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-09/LHCB1/Results/b1_flat_60_18cm_after_coupling_cor",
)
# Get ellipse points (center, +1 sigma, -1 sigma)
x_ellipse, px_ellipse, y_ellipse, py_ellipse = ps_diag.ellipse_points()
x_upper, px_upper, y_upper, py_upper = ps_diag.ellipse_sigma(sigma_level=1.0)
x_lower, px_lower, y_lower, py_lower = ps_diag.ellipse_sigma(sigma_level=-1.0)

# Plot both (x, px) and (y, py) phase spaces with ellipses in the same figure
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# (x, px) subplot
axs[0].scatter(x_bpm_data["x"], x_bpm_data["px"], s=5, alpha=0.5, label="Data P 1")
axs[0].plot(x_ellipse, px_ellipse, color="orange", label="Ellipse")
axs[0].plot(x_upper, px_upper, color="red", linestyle="--", label="+1 Sigma")
axs[0].plot(x_lower, px_lower, color="blue", linestyle="--", label="-1 Sigma")
axs[0].set_title(f"Phase Space at {BPM_START} (X/PX) with Ellipse")
axs[0].set_xlabel("X Position (m)")
axs[0].set_ylabel("PX")
axs[0].legend()

# (y, py) subplot
axs[1].scatter(x_bpm_data["y"], x_bpm_data["py"], s=5, alpha=0.5, label="Data P 1")
axs[1].plot(y_ellipse, py_ellipse, color="orange", label="Ellipse")
axs[1].plot(y_upper, py_upper, color="red", linestyle="--", label="+1 Sigma")
axs[1].plot(y_lower, py_lower, color="blue", linestyle="--", label="-1 Sigma")
axs[1].set_title(f"Phase Space at {BPM_START} (Y/PY) with Ellipse")
axs[1].set_xlabel("Y Position (m)")
axs[1].set_ylabel("PY")
axs[1].legend()

plt.tight_layout()
plt.show()
