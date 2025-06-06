import pandas as pd
import tfs

from aba_optimiser.config import (
    FILTERED_FILE,
    NOISE_FILE,
    TRACK_DATA_FILE,
)
from aba_optimiser.ellipse_filtering import filter_noisy_data
from aba_optimiser.make_noisy_track_data import make_noisy_track_data

if not NOISE_FILE.exists() or TRACK_DATA_FILE.stat().st_mtime > NOISE_FILE.stat().st_mtime:
    print(f"Creating noisy TFS data file: {NOISE_FILE}")
    orig_data = tfs.read(TRACK_DATA_FILE)
    data_p, data_n = make_noisy_track_data(orig_data)
    data_p.to_feather(NOISE_FILE, compression="lz4")
else:
    print("Noisy data already exists and is up-to-date. Creating filtered data only.")
    data_p = pd.read_feather(NOISE_FILE)

filtered_data = filter_noisy_data(data_p)

filtered_data.to_feather(FILTERED_FILE, compression="lz4")
print("→ Saved filtered data:", FILTERED_FILE)
    