from __future__ import annotations

import datetime
import gc
import logging
import multiprocessing as mp
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter

from aba_optimiser.config import (
    ACD_ON,
    BPM_START_POINTS,
    DECAY_EPOCHS,
    # FILTER_DATA,
    FLATTOP_TURNS,
    GRAD_NORM_ALPHA,
    GRADIENT_CONVERGED_VALUE,
    KNOB_TABLE,
    MAGNET_RANGE,
    MAX_EPOCHS,
    MAX_LR,
    MIN_LR,
    N_COMPARE_TURNS,
    # N_RUN_TURNS,
    NOISE_FILE,
    NUM_TRACKS,
    NUM_WORKERS,
    OPTIMISER_TYPE,
    OUTPUT_KNOBS,
    POSITION_STD_DEV,
    RAMP_UP_TURNS,
    RUN_ARC_BY_ARC,
    SEQUENCE_FILE,
    TRACK_DATA_FILE,
    TRACKS_PER_WORKER,
    TRUE_STRENGTHS,
    USE_NOISY_DATA,
    WARMUP_EPOCHS,
    WARMUP_LR_START,
)
from aba_optimiser.dataframes.utils import select_markers
from aba_optimiser.io.utils import read_knobs, save_results, scientific_notation
from aba_optimiser.mad.mad_interface import MadInterface
from aba_optimiser.optimisers.adam import AdamOptimiser
from aba_optimiser.optimisers.amsgrad import AMSGradOptimiser
from aba_optimiser.plotting import (
    plot_strengths_comparison,
    plot_strengths_vs_position,
    show_plots,
)
from aba_optimiser.training.scheduler import LRScheduler
from aba_optimiser.workers.arc_by_arc import ArcByArcWorker
from aba_optimiser.workers.ring import RingWorker

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

LOGGER = logging.getLogger(__name__)
random.seed(42)  # For reproducibility


def circular_far_samples(arr, k, start_offset=0):
    """
    Pick k indices as far apart as possible on a circular array.
    k must be <= len(arr). start_offset rotates the selection.
    """
    n = len(arr)
    if k > n:
        raise ValueError("k must be <= len(arr)")
    step = n / k
    # bin centers → unique indices even when step is non-integer
    idx = (np.floor((np.arange(k) + 0.5) * step + start_offset) % n).astype(int)
    return arr[idx], idx


class Controller:
    """
    Orchestrates multi-process knob optimisation using MAD-NG.
    """

    def __init__(self):
        # Top-level construction broken into small helpers for readability.
        self.true_strengths = read_knobs(TRUE_STRENGTHS)

        # MAD and knobs
        self._init_mad()

        # Optimiser and LR scheduler
        self._init_optimiser()

        # Track data and comparison set
        self._load_track_data()

        # Prepare list of available turns to process
        self._prepare_turn_list()

        # State used during training
        self.smoothed_grad_norm = None

        self._set_n_data_points()

    def _init_mad(self) -> None:
        """Set up MAD interface and read model knobs/positions."""
        self._setup_mad_interface()
        self._determine_worker_and_bpms()
        self._initialize_knob_strengths()

    def _setup_mad_interface(self) -> None:
        """Initialize the MAD-NG interface and get basic model parameters."""
        self.mad_iface = MadInterface(
            SEQUENCE_FILE, MAGNET_RANGE, discard_mad_output=True
        )
        self.knob_names = self.mad_iface.knob_names
        self.elem_spos = self.mad_iface.elem_spos
        tws = self.mad_iface.run_twiss()
        LOGGER.info(f"Found tunes: {tws['q1']}, {tws['q2']}")
        self.all_bpms = tws.index.to_numpy()

    def _determine_worker_and_bpms(self) -> None:
        """Determine the worker type and BPM start points based on the run mode."""
        if RUN_ARC_BY_ARC:
            LOGGER.warning(
                "Arc by arc chosen, ignoring N_RUN_TURNS, OBSERVE_TURNS_FROM, N_COMPARE_TURNS"
            )
            for bpm in BPM_START_POINTS:
                if bpm not in self.all_bpms:
                    raise ValueError(f"BPM {bpm} not found in the sequence.")
            self.bpm_start_points = BPM_START_POINTS
            self.Worker = ArcByArcWorker
        else:
            LOGGER.warning(
                "Whole ring chosen, BPM start points ignored, taking an even distribution based on NUM_WORKERS"
            )
            # self.bpm_start_points = self.all_bpms[:1]
            self.bpm_start_points, _ = circular_far_samples(
                self.all_bpms, min(NUM_WORKERS, len(self.all_bpms))
            )
            self.Worker = RingWorker

    def _initialize_knob_strengths(self) -> None:
        """Initialize knob strengths from MAD and filter true strengths."""
        initial_strengths = self.mad_iface.receive_knob_values()
        self.initial_strengths = initial_strengths
        self.current_knobs = dict(zip(self.knob_names, initial_strengths))

        # Restrict true strengths to knobs we actually have in model
        if RUN_ARC_BY_ARC:
            self.true_strengths = {
                knob: self.true_strengths[knob] for knob in self.knob_names
            }
        else:
            missing = set(self.knob_names) ^ set(self.true_strengths)
            if missing:
                raise ValueError(
                    f"Mismatch between model knobs and true strengths: {missing}"
                )

    def _init_optimiser(self) -> None:
        """Construct optimiser and learning-rate scheduler."""
        optimiser_kwargs = {
            "shape": self.initial_strengths.shape,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0,
        }
        if OPTIMISER_TYPE == "amsgrad":
            self.optimiser = AMSGradOptimiser(**optimiser_kwargs)
        else:
            self.optimiser = AdamOptimiser(**optimiser_kwargs)

        self.scheduler = LRScheduler(
            warmup_epochs=WARMUP_EPOCHS,
            decay_epochs=DECAY_EPOCHS,
            start_lr=WARMUP_LR_START,
            max_lr=MAX_LR,
            min_lr=MIN_LR,
        )

    def _load_track_data(self) -> None:
        """Load and pre-process track/comparison data into `self.comparison_data`."""
        if USE_NOISY_DATA is False:
            track_data = pd.read_parquet(TRACK_DATA_FILE)
        # elif FILTER_DATA:
        #     from aba_optimiser.config import KALMAN_FILE

        #     track_data = pd.read_feather(KALMAN_FILE)
        else:
            track_data = pd.read_parquet(NOISE_FILE)

        # Keep only the columns we need
        cols = ["turn", "name", "x", "px", "y", "py"]
        missing = [c for c in cols if c not in track_data.columns]
        if missing:
            raise ValueError(f"Missing columns in track data: {missing}")
        track_data = track_data[cols]

        # Downcast and optimise memory
        track_data["turn"] = track_data["turn"].astype("int32", copy=False)
        track_data["name"] = track_data["name"].astype("category", copy=False)

        # Filter data to BPMs present in the sequence
        self.comparison_data = select_markers(track_data, self.all_bpms)
        del track_data

    def _prepare_turn_list(self) -> None:
        """Build the list of turns to be processed and validate availability."""
        self.var_x = (POSITION_STD_DEV) ** 2
        self.var_y = (POSITION_STD_DEV) ** 2

        if RUN_ARC_BY_ARC:
            turn_list = self.comparison_data["turn"].unique().tolist()
            turn_list.pop(-1)
        else:
            turn_list = [
                int(turn)
                for turn in self.comparison_data["turn"].unique()
                if turn < FLATTOP_TURNS * NUM_TRACKS - 1
            ]

        num_turns_needed = TRACKS_PER_WORKER * NUM_WORKERS // len(BPM_START_POINTS)
        if len(turn_list) < num_turns_needed:
            raise ValueError(
                f"Not enough available turns for x or y BPMs. Need {num_turns_needed}, but found {len(turn_list)} turns."
            )

        if ACD_ON:
            turn_list = [turn for turn in turn_list if turn >= RAMP_UP_TURNS]

        random.shuffle(turn_list)

        # Determine how many turns each batch should contain. We split the
        # available turn_list into NUM_WORKERS batches which will then be
        # re-used for every BPM start point so that each BPM sees the same
        # collection of turn batches.
        tracks_per_worker = min(len(turn_list) // NUM_WORKERS, TRACKS_PER_WORKER)
        n_compare = N_COMPARE_TURNS if not RUN_ARC_BY_ARC else 1
        # Build the canonical set of turn batches (length == NUM_WORKERS)
        num_batches = NUM_WORKERS // len(self.bpm_start_points)
        self.turn_batches = [
            turn_list[i * tracks_per_worker : (i + 1) * tracks_per_worker]
            for i in range(num_batches)
        ]
        # Total workers will be number of BPM start points times number of batches
        self.total_turns = NUM_WORKERS * tracks_per_worker * n_compare

    def _create_worker_payloads(self, comp: pd.DataFrame) -> list[dict]:
        """Create payloads for all workers."""
        payloads = []
        wid = 0
        for start_bpm in self.bpm_start_points:
            for batch_idx, turn_batch in enumerate(self.turn_batches):
                assert turn_batch, (
                    f"Turn batch {batch_idx} for BPM {start_bpm} is empty. Check TRACKS_PER_WORKER and NUM_WORKERS."
                )
                x_comp, y_comp, init_coords = self._make_worker_payload(
                    comp, turn_batch, start_bpm, self.n_data_points[start_bpm]
                )
                payloads.append(
                    {
                        "wid": wid,
                        "x_comp": x_comp,
                        "y_comp": y_comp,
                        "init_coords": init_coords,
                        "start_bpm": start_bpm,
                    }
                )
                wid += 1
        return payloads

    def _make_worker_payload(
        self,
        comparison_dataframe: pd.DataFrame,
        turn_batch: list[int],
        start_bpm: str,
        n_data_points: int,
    ):
        """Create x/y comparison arrays and init_coords for a worker.

        Args:
            comp: indexed comparison DataFrame (index: (turn, name))
        """
        x_list, y_list, init_coords = [], [], []
        for t in turn_batch:
            starting_row = comparison_dataframe.loc[(t, start_bpm)]
            init_coords.append(
                [
                    starting_row["x"],
                    starting_row["px"],
                    starting_row["y"],
                    starting_row["py"],
                    0.0,
                    0.0,
                ]
            )
            obs_turns = self.Worker.get_observation_turns(t)

            blocks = []
            for ot in obs_turns:
                pos = comparison_dataframe.index.get_loc((ot, start_bpm))
                blocks.append(comparison_dataframe.iloc[pos : pos + n_data_points])

            filtered = pd.concat(blocks, axis=0)
            if filtered.shape[0] == 0:
                raise ValueError(f"No data available for turn {t}")

            x_list.append(filtered["x"].to_numpy(dtype="float64", copy=False))
            y_list.append(filtered["y"].to_numpy(dtype="float64", copy=False))

        x_comp = np.stack(x_list, axis=0)
        y_comp = np.stack(y_list, axis=0)
        return x_comp, y_comp, init_coords

    def _write_results(self, uncertainties) -> None:
        """Write final knob strengths and markdown table to file."""
        logging.info("Writing final knob strengths and markdown table...")
        save_results(self.knob_names, self.current_knobs, uncertainties, OUTPUT_KNOBS)

        # Prepare rows with index, knob, true, final, diff, relative difference, and uncertainty.
        rows = []
        for idx, knob in enumerate(self.knob_names):
            true_val = self.true_strengths[knob]
            final_val = self.current_knobs[knob]
            diff = final_val - true_val
            rel_diff = diff / true_val if true_val != 0 else 0
            uncertainty_val = uncertainties[idx]
            rows.append(
                {
                    "index": idx,
                    "knob": knob,
                    "true": true_val,
                    "final": final_val,
                    "diff": diff,
                    "reldiff": rel_diff,
                    "uncertainty": uncertainty_val,
                    "rel_uncertainty": uncertainty_val / abs(true_val)
                    if true_val != 0
                    else 0,
                }
            )

        # Order rows by relative difference (descending order)
        rows.sort(key=lambda row: abs(row["reldiff"]), reverse=True)

        with Path(KNOB_TABLE).open("w") as f:
            f.write(
                "| Index |   Knob   |   True   |   Diff   | Uncertainty | Relative Diff | Relative Uncertainty |\n"
                "|-------|----------|----------|----------|-------------|---------------|----------------------|\n"
            )
            for row in rows:
                f.write(
                    f"|{row['index']}|{row['knob']}|"
                    f"{scientific_notation(row['true'])}|"
                    f"{scientific_notation(row['diff'])}|"
                    f"{scientific_notation(row['uncertainty'])}|"
                    f"{scientific_notation(row['reldiff'])}|"
                    f"{scientific_notation(row['rel_uncertainty'])}|\n"
                )
        LOGGER.info("Optimisation complete.")

    def _set_n_data_points(self):
        self.n_data_points = {}
        for start_bpm in self.bpm_start_points:
            # Decide bpm_range similar to worker behaviour
            bpm_range = self.Worker.get_bpm_range(start_bpm)
            n_bpms = MadInterface(
                SEQUENCE_FILE, magnet_range=MAGNET_RANGE, bpm_range=bpm_range
            ).nbpms
            self.n_data_points[start_bpm] = self.Worker.get_n_data_points(n_bpms)

    def run(self) -> None:
        """Execute the optimisation loop."""
        run_start = time.time()
        writer = self._setup_logging()

        parent_conns, workers = self._start_workers()

        try:
            self._optimisation_loop(parent_conns, writer, run_start)
        except KeyboardInterrupt:
            logging.warning(
                "\nKeyboardInterrupt detected. Terminating early and writing results."
            )
        finally:
            self._cleanup_and_save(parent_conns, workers, writer)

    def _setup_logging(self) -> SummaryWriter:
        """Sets up TensorBoard logging."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return SummaryWriter(log_dir=f"runs/{ts}_opt")

    def _start_workers(self) -> tuple[list[Connection], list[mp.Process]]:
        """Start worker processes and return their connections and process objects."""
        parent_conns: list[Connection] = []
        workers: list[mp.Process] = []

        comp = self.comparison_data.set_index(["turn", "name"])
        payloads = self._create_worker_payloads(comp)

        for payload in payloads:
            parent, child = mp.Pipe()
            w = self.Worker(
                child,
                payload["wid"],
                payload["x_comp"],
                payload["y_comp"],
                payload["init_coords"],
                payload["start_bpm"],
            )
            w.start()
            parent_conns.append(parent)
            workers.append(w)
            parent.send((self.var_x, self.var_y))

        # Clean up memory
        del self.comparison_data, self.var_x, self.var_y, self.turn_batches
        del self.bpm_start_points, self.n_data_points, comp, payloads
        gc.collect()
        return parent_conns, workers

    def _optimisation_loop(
        self, parent_conns: list[Connection], writer: SummaryWriter, run_start: float
    ) -> None:
        """Main optimisation loop."""
        for epoch in range(MAX_EPOCHS):
            epoch_start = time.time()

            for conn in parent_conns:
                conn.send(self.current_knobs)

            lr = self.scheduler(epoch)
            total_loss, agg_grad = self._collect_worker_results(parent_conns)

            self._update_knobs(agg_grad, lr)
            grad_norm = np.linalg.norm(agg_grad)
            self._update_smoothed_grad_norm(grad_norm)

            self._log_epoch_stats(
                writer, epoch, total_loss, grad_norm, lr, epoch_start, run_start
            )

            if self.smoothed_grad_norm < GRADIENT_CONVERGED_VALUE:
                LOGGER.info(
                    f"\nGradient norm below threshold: {self.smoothed_grad_norm:.3e}. "
                    f"Stopping early at epoch {epoch}."
                )
                break

    def _collect_worker_results(
        self, parent_conns: list[Connection]
    ) -> tuple[float, np.ndarray]:
        """Collect results from all workers for an epoch."""
        total_loss = 0.0
        agg_grad: None | np.ndarray = None
        for conn in parent_conns:
            _, grad, loss = conn.recv()
            agg_grad = grad if agg_grad is None else agg_grad + grad
            total_loss += loss
        total_loss /= self.total_turns
        agg_grad = agg_grad.flatten() / self.total_turns
        return total_loss, agg_grad

    def _update_knobs(self, agg_grad: np.ndarray, lr: float) -> None:
        """Update knob values using the optimiser."""
        param_vec = np.array([self.current_knobs[k] for k in self.knob_names])
        new_vec = self.optimiser.step(param_vec, agg_grad, lr)
        self.current_knobs = dict(zip(self.knob_names, new_vec))

    def _update_smoothed_grad_norm(self, grad_norm: float) -> None:
        """Update the exponential moving average of the gradient norm."""
        if self.smoothed_grad_norm is None:
            self.smoothed_grad_norm = grad_norm
        else:
            self.smoothed_grad_norm = (
                GRAD_NORM_ALPHA * self.smoothed_grad_norm
                + (1.0 - GRAD_NORM_ALPHA) * grad_norm
            )

    def _log_epoch_stats(
        self,
        writer: SummaryWriter,
        epoch: int,
        total_loss: float,
        grad_norm: float,
        lr: float,
        epoch_start: float,
        run_start: float,
    ) -> None:
        """Log statistics for the current epoch."""
        true_diff = [
            abs(self.current_knobs[k] - self.true_strengths[k]) for k in self.knob_names
        ]
        rel_diff = [
            diff / abs(self.true_strengths[k]) if self.true_strengths[k] != 0 else 0
            for k, diff in zip(self.knob_names, true_diff)
        ]

        sum_true_diff = np.sum(true_diff)
        sum_rel_diff = np.sum(rel_diff)

        writer.add_scalar("loss", total_loss, epoch)
        writer.add_scalar("grad_norm", grad_norm, epoch)
        writer.add_scalar("true_diff", sum_true_diff, epoch)
        writer.add_scalar("rel_diff", sum_rel_diff, epoch)
        writer.add_scalar("learning_rate", lr, epoch)
        writer.flush()

        epoch_time = time.time() - epoch_start
        total_time = time.time() - run_start

        LOGGER.info(
            f"\rEpoch {epoch}: "
            f"loss={total_loss:.3e}, "
            f"grad_norm={grad_norm:.3e}, "
            f"true_diff={sum_true_diff:.3e}, "
            f"rel_diff={sum_rel_diff:.3e}, "
            f"lr={lr:.3e}, "
            f"epoch_time={epoch_time:.3f}s, "
            f"total_time={total_time:.3f}s",
        )

    def _cleanup_and_save(
        self,
        parent_conns: list[Connection],
        workers: list[mp.Process],
        writer: SummaryWriter,
    ) -> None:
        """Clean up resources and save final results."""
        logging.info("\nTerminating workers...")
        h_global = np.zeros((len(self.knob_names), len(self.knob_names)))
        for conn in parent_conns:
            conn.send(None)  # Signal workers to stop

        for conn in parent_conns:
            h_global += conn.recv()

        cov = np.linalg.inv(h_global)
        comb_uncertainty = np.sqrt(np.diag(cov))

        for w in workers:
            w.join()

        writer.close()
        self._write_results(comb_uncertainty)
        self._plot_results(comb_uncertainty)

    def _plot_results(self, uncertainties) -> None:
        """Generate all plotting results using the plotting module."""
        logging.info("Generating plots...")
        knob_names = self.knob_names
        magnet_names = [knob[:-3] for knob in knob_names]
        initial_vals = np.array(
            [self.initial_strengths[i] for i in range(len(knob_names))]
        )
        final_vals = np.array([self.current_knobs[k] for k in knob_names])
        true_vals = np.array([self.true_strengths[k] for k in knob_names])

        save_prefix = "plots/"
        show_errorbars = True
        # Relative difference comparison
        for plot_real in [False, True]:
            _ext = "_real" if plot_real else ""
            plot_strengths_comparison(
                magnet_names,
                final_vals,
                true_vals,
                uncertainties,
                initial_vals=initial_vals,
                show_errorbars=show_errorbars,
                plot_real=plot_real,
                save_path=f"{save_prefix}relative_difference_comparison{_ext}.png",
                unit="$m^{-1}$",
            )

            plot_strengths_vs_position(
                self.elem_spos,
                final_vals,
                true_vals,
                uncertainties,
                initial_vals=initial_vals,
                show_errorbars=show_errorbars,
                plot_real=plot_real,
                save_path=f"{save_prefix}relative_difference_vs_position_comparison{_ext}.png",
            )
        show_plots()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    ctrl = Controller()
    ctrl.run()
