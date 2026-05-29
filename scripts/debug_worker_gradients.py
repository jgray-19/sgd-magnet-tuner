from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aba_optimiser.accelerators import LHC
from aba_optimiser.mad.aba_mad_interface import AbaMadInterface
from aba_optimiser.training.controller import Controller
from aba_optimiser.training.controller_config import MeasurementConfig, OutputConfig, SequenceConfig
from tests.training.controller_test_utils import (
    _generate_nonoise_track,
    _make_optimiser_config_quad,
    _make_simulation_config_quad,
)


def _build_controller(start_marker: str, *, disable_screening: bool) -> Controller:
    repo = Path(__file__).resolve().parent.parent
    seq_b1 = repo / "tests" / "data" / "sequences" / "lhcb1.seq"
    magnet_range = "BPM.9R1.B1/BPM.9L2.B1"
    bpm_start_points = [f"BPM.{i}R1.B1" for i in range(9, 14)]
    bpm_end_points = [f"BPM.{i}L2.B1" for i in range(9, 14)]
    flattop_turns = 1000

    tmp_root = Path(tempfile.mkdtemp(prefix="debug_worker_gradients_"))
    off_magnet_path = tmp_root / "track_off_magnet.parquet"

    loaded_interface = AbaMadInterface(accelerator=LHC(beam=1, sequence_file=seq_b1))
    corrector_file, magnet_strengths, tune_knobs_file = _generate_nonoise_track(
        loaded_interface,
        flattop_turns,
        off_magnet_path,
        0.0,
        start_marker=start_marker,
        perturb_quads=True,
    )

    optimiser_config = _make_optimiser_config_quad()
    simulation_config = _make_simulation_config_quad()
    if disable_screening:
        simulation_config.enable_preloop_outlier_screening = False

    ctrl = Controller(
        LHC(
            beam=1,
            pc=6800,
            sequence_file=seq_b1,
            optimise_quadrupoles=True,
        ),
        optimiser_config,
        simulation_config,
        SequenceConfig(
            magnet_range=magnet_range,
            first_bpm=start_marker,
        ),
        MeasurementConfig(
            measurement_files=off_magnet_path,
            corrector_files=corrector_file,
            tune_knobs_files=tune_knobs_file,
            flattop_turns=flattop_turns,
            bunches_per_file=1,
        ),
        bpm_start_points,
        bpm_end_points,
        output_config=OutputConfig(
            show_plots=False,
            plots_dir=tmp_root / "plots",
            mad_logfile=tmp_root / "mad_logfile.log",
            write_tensorboard_logs=False,
        ),
        true_strengths=magnet_strengths,
        debug=False,
        optimise_knobs=None,
    )
    return ctrl


def _summarise_results(ctrl: Controller, batch: int, top_k: int) -> list[dict[str, object]]:
    ctrl.worker_manager.start_workers(
        ctrl.data_manager.track_data,
        ctrl.data_manager.turn_batches,
        ctrl.data_manager.file_map,
        ctrl.config_manager.start_bpms,
        ctrl.config_manager.end_bpms,
        ctrl.simulation_config,
        ctrl.machine_deltaps,
        ctrl.initial_knobs,
    )

    if ctrl.simulation_config.enable_preloop_outlier_screening:
        ctrl.worker_manager.screen_initial_outliers(
            ctrl.initial_knobs,
            bpm_sigma_threshold=ctrl.simulation_config.bpm_loss_outlier_sigma,
            worker_sigma_threshold=ctrl.simulation_config.worker_loss_outlier_sigma,
        )

    channels = ctrl.worker_manager.channels
    if channels is None:
        raise RuntimeError("Worker channels were not initialised")

    channels.send_all((ctrl.initial_knobs, batch))
    results = channels.recv_all()

    knob_names = ctrl.config_manager.knob_names
    summaries: list[dict[str, object]] = []
    for meta, result in zip(ctrl.worker_manager.worker_metadata, results, strict=True):
        worker_id, grad, loss = result
        grad_arr = np.asarray(grad, dtype=float).reshape(-1)
        top_idx = np.argsort(np.abs(grad_arr))[::-1][:top_k]
        top_terms = [
            {"knob": knob_names[i], "grad": float(grad_arr[i])}
            for i in top_idx
            if grad_arr[i] != 0.0
        ]
        summaries.append(
            {
                "worker_id": int(worker_id),
                "start_bpm": meta.start_bpm,
                "end_bpm": meta.end_bpm,
                "sdir": int(meta.sdir),
                "kick_plane": str(meta.kick_plane),
                "n_bpms": len(meta.bpm_names),
                "loss": float(loss),
                "grad_norm": float(np.linalg.norm(grad_arr)),
                "max_abs_grad": float(np.max(np.abs(grad_arr))) if grad_arr.size else 0.0,
                "nonzero_grads": int(np.count_nonzero(grad_arr)),
                "top_terms": top_terms,
            }
        )

    ctrl.worker_manager.termination_and_hessian(len(ctrl.initial_knobs), estimate_hessian=False)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-marker", default="MSIA.EXIT.B1")
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--disable-screening", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    ctrl = _build_controller(args.start_marker, disable_screening=args.disable_screening)
    summaries = _summarise_results(ctrl, args.batch, args.top_k)

    payload = {
        "start_marker": args.start_marker,
        "batch": args.batch,
        "turn_batches": [len(batch) for batch in ctrl.data_manager.turn_batches],
        "total_turns": ctrl.data_manager.get_total_turns(),
        "worker_count": len(summaries),
        "workers": summaries,
    }

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.output_json}")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
