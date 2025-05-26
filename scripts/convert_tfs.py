from pathlib import Path

import turn_by_turn as tbt
from aba_optimiser.config import (
    FILTERED_FILE,
    TRACK_DATA_FILE,
    NOISE_FILE
)

for file in [FILTERED_FILE, TRACK_DATA_FILE, NOISE_FILE]:
    data = tbt.read_tbt(file, 'madng')
    new_path = Path('sdds') / (file.stem + '.sdds')
    tbt.write_tbt(new_path, data)
