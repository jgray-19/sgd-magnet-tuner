import gc
import re

import numpy as np
import pandas as pd
from pymadng import MAD

from aba_optimiser.config import (
    BEAM_ENERGY,
    BPM_RANGE,
    ELEM_NAMES_FILE,
    FILTERED_FILE,
    FLATTOP_TURNS,
    REL_K1_STD_DEV,
    SEQ_NAME,
    SEQUENCE_FILE,
    TRUE_STRENGTHS,
    TUNE_KNOBS_FILE,
)
from aba_optimiser.ellipse_filtering import filter_noisy_data
from aba_optimiser.make_noisy_track_data import make_noisy_track_data

# Initialize MAD-NG interface
mad = MAD(debug=False)
mad.load("MADX")

# Load the sequence
mad.MADX.load(f"'{SEQUENCE_FILE.absolute()}'")
seq = mad.MADX[SEQ_NAME]
mad["SEQ_NAME"] = SEQ_NAME

num_tracks = 4

np.random.seed(42)  # reproducibility

# Define parameters
tunes = [0.28, 0.31]
knob_start, end_pos = BPM_RANGE.split("/")
mad["knob_range"] = BPM_RANGE

# Collect quadrupole element names
quad_in_range = []
quad_all = []
start, end, _ = mad.MADX[SEQ_NAME].range_of("knob_range")

for i, elm in enumerate(seq):
    if elm.kind == "quadrupole" and elm.k1 != 0 and "MQ." in elm.name:
        quad_all.append(elm.name)
        if i >= start and i <= end:
            quad_in_range.append(elm.name)

print(f"Found {len(quad_in_range)} quadrupoles in range {BPM_RANGE}.")


# Group elements by base name (handling aliases)
def extract_base(name):
    return re.sub(r"\.[AB](\d+[RL]\d\.B\d)$", r".\1", name)


groups_all = {}
groups = {}
for name in quad_all:
    base = extract_base(name)
    groups_all.setdefault(base, []).append(name)
    if name in quad_in_range:
        groups.setdefault(base, []).append(name)

# Apply noise to quadrupoles
for base, aliases in groups_all.items():
    noise = REL_K1_STD_DEV * np.random.randn()
    for name in aliases:
        k1 = mad[f"MADX['{name}'].k1"]
        mad[f"MADX['{name}'].k1"] = k1 + noise * abs(k1)

# Create element data for Python instead of file

mad["base_list"] = list(groups.keys())
mad["groups"] = groups
mad["elem_names_file"] = str(ELEM_NAMES_FILE.absolute())
mad.send("""
local outFile = io.open(elem_names_file, "w")
for _, base in ipairs(base_list) do
    local alias_list = groups[base]
    for _, name in ipairs(alias_list) do
        local spos
        for i, e, s, ds in MADX[SEQ_NAME]:siter(knob_range) do
            if e.name == name then
                spos = s
                break
            end
        end
        outFile:write(spos .. "\\t" .. name)
    end
    outFile:write("\\n")
end
outFile:close()
""")

# Set beam
seq.beam = mad.beam(particle='"proton"', energy=BEAM_ENERGY)

# Match tunes
mad["result"] = mad.match(
    command="\ -> twiss{sequence=MADX[SEQ_NAME]}",
    variables=[
        {"var": "'MADX.dqx_b1_op'", "name": "'dQx.b1_op'"},
        {"var": "'MADX.dqy_b1_op'", "name": "'dQy.b1_op'"},
    ],
    equalities=[
        {"expr": f"\\t -> math.abs(t.q1)-(62+{tunes[0]})", "name": "'q1'"},
        {"expr": f"\\t -> math.abs(t.q2)-(60+{tunes[1]})", "name": "'q2'"},
    ],
    objective={"fmin": 1e-18},
    info=2,
)

# Store matched tunes in Python variables
matched_tunes = {key: mad[f"MADX['{key}']"] for key in ("dqx_b1_op", "dqy_b1_op")}

# Save matched tunes to file using a loop
with open(TUNE_KNOBS_FILE, "w") as f:
    for key, val in matched_tunes.items():
        f.write(f"{key}\t{val: .15e}\n")

# Save final strengths to Python
true_strengths = {
    base: mad[f"MADX['{aliases[0]}'].k1"] for base, aliases in groups.items()
}
with open(TRUE_STRENGTHS, "w") as f:
    for base, val in true_strengths.items():
        f.write(f"{base}_k1\t{val: .15e}\n")

# Twiss before tracking
tw, mflw = mad.twiss(sequence=seq)
df_twiss = tw.to_df()
del tw, mflw

# Prepare BPM selection
seq.deselect(mad.element.flags.observed)
seq.select(mad.element.flags.observed, {"pattern": "'BPM'"})

# Tracking
track_results = []
noisy_results = []
sign_list = [(1, 1), (-1, 1), (1, -1), (-1, -1)]

for ntrk in range(num_tracks):
    x_sign, y_sign = sign_list[ntrk]
    beta11 = df_twiss[df_twiss["name"] == knob_start]["beta11"].values[0]
    beta22 = df_twiss[df_twiss["name"] == knob_start]["beta22"].values[0]

    X0 = {
        "x": x_sign * 5e-3 / np.sqrt(beta11),
        "px": -x_sign * 1e-6,
        "y": y_sign * 5e-3 / np.sqrt(beta22),
        "py": -y_sign * 1e-6,
        "t": 0,
        "pt": 0,
    }

    trk, mflw = mad.track(sequence=seq, X0=X0, nturn=FLATTOP_TURNS, info=1)
    df_track = trk.to_df(columns=["name", "turn", "x", "px", "y", "py"])
    del trk, mflw

    # Downcast types for memory efficiency
    df_track["name"] = df_track["name"].astype("category")
    df_track["turn"] = df_track["turn"].astype(np.int32)

    # Make noisy data
    noisy_df, _ = make_noisy_track_data(df_track)
    del df_track

    # Offset the 'turn' column so turns don't overlap
    noisy_df["turn"] += ntrk * FLATTOP_TURNS
    noisy_results.append(noisy_df)
    del noisy_df
    gc.collect()

# Combine all noisy dataframes
all_noisy = pd.concat(noisy_results, ignore_index=True)
del noisy_results
gc.collect()

# Filter the combined noisy data
filtered_data = filter_noisy_data(all_noisy)
del all_noisy
gc.collect()

# Downcast filtered data before saving
filtered_data["name"] = filtered_data["name"].astype("category")
filtered_data["turn"] = filtered_data["turn"].astype(np.int32)
# Save the filtered data
filtered_data.to_feather(FILTERED_FILE, compression="lz4")
del filtered_data
gc.collect()

print("All calculations and filtering completed successfully.")
