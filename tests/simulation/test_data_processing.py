from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from aba_optimiser.config import MOMENTUM_STD_DEV, POSITION_STD_DEV
from aba_optimiser.simulation.data_processing import prepare_track_dataframe, single_writer_loop


def test_prepare_track_dataframe_offsets_turns_and_tags_bunch() -> None:
    true_df = pd.DataFrame({"turn": [0, 1, 2], "x": [1.0, 2.0, 3.0]})

    result = prepare_track_dataframe(true_df, ntrk=3, flattop_turns=100)

    assert list(result["turn"]) == [300, 301, 302]
    assert (result["bunch_number"] == 3).all()
    assert (result["var_x"] == POSITION_STD_DEV**2).all()
    assert (result["var_y"] == POSITION_STD_DEV**2).all()
    assert (result["var_px"] == MOMENTUM_STD_DEV**2).all()
    assert (result["var_py"] == MOMENTUM_STD_DEV**2).all()


def test_single_writer_loop_writes_all_queued_tables_then_stops(tmp_path: Path) -> None:
    out_path = tmp_path / "written.parquet"
    queue: mp.Queue = mp.Queue()
    table1 = pa.table({"turn": [0, 1], "x": [1.0, 2.0]})
    table2 = pa.table({"turn": [2, 3], "x": [3.0, 4.0]})
    queue.put(table1)
    queue.put(table2)
    queue.put(None)

    single_writer_loop(queue, str(out_path))

    result = pq.read_table(out_path).to_pandas()
    assert list(result["turn"]) == [0, 1, 2, 3]
    assert list(result["x"]) == [1.0, 2.0, 3.0, 4.0]
