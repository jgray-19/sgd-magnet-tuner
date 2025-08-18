from __future__ import annotations

import datetime
import logging
import multiprocessing as mp
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tfs
from tensorboardX import SummaryWriter

from aba_optimiser.adam import AdamOptimiser
from aba_optimiser.amsgrad import AMSGradOptimiser
from aba_optimiser.config import (
    ACD_ON,
    BPM_START_POINTS,
    DECAY_EPOCHS,
    FILTER_DATA,
    FLATTOP_TURNS,
    GRAD_NORM_ALPHA,
    KNOB_TABLE,
    MAGNET_RANGE,
    MAX_EPOCHS,
    MAX_LR,
    MIN_LR,
    N_COMPARE_TURNS,
    N_RUN_TURNS,
    NOISE_FILE,
    NUM_TRACKS,
    NUM_WORKERS,
    OPTIMISER_TYPE,
    OUTPUT_KNOBS,
    POSITION_STD_DEV,
    RAMP_UP_TURNS,
    # BPM_START_POINTS,
    RUN_ARC_BY_ARC,
    SEQUENCE_FILE,
    TRACK_DATA_FILE,
    TRACKS_PER_WORKER,
    TRUE_STRENGTHS,
    USE_NOISY_DATA,
    WARMUP_EPOCHS,
    WARMUP_LR_START,
)
from aba_optimiser.mad_interface import MadInterface
from aba_optimiser.scheduler import LRScheduler
from aba_optimiser.utils import (
    read_knobs,
    save_results,
    scientific_notation,
    select_markers,
)
from aba_optimiser.worker import build_worker

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

LOGGER = logging.getLogger(__name__)


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
        # Read true strengths
        self.true_strengths = read_knobs(TRUE_STRENGTHS)

        # Initialise MAD interface and knobs
        self.mad_iface = MadInterface(
            SEQUENCE_FILE, MAGNET_RANGE, discard_mad_output=True
        )
        self.knob_names = self.mad_iface.knob_names
        self.elem_spos = self.mad_iface.elem_spos

        # Run a twiss and to get the beta functions
        tws = self.mad_iface.run_twiss()
        LOGGER.info(f"Found tunes: {tws['q1']}, {tws['q2']}")

        # Remove all rows that are before the start BPM and after the end BPM
        all_bpms = tws.index.to_numpy()
        if RUN_ARC_BY_ARC:
            LOGGER.warning(
                "Arc by arc chosen, ignoring N_RUN_TURNS, OBSERVE_TURNS_FROM, N_COMPARE_TURNS"
            )
            for bpm in BPM_START_POINTS:
                if bpm not in all_bpms:
                    raise ValueError(f"BPM {bpm} not found in the sequence.")
            self.bpm_start_points = BPM_START_POINTS
        else:
            LOGGER.warning(
                "Whole ring chosen, BPM start points ignored, taking an even distribution based on NUM_WORKERS"
            )
            self.bpm_start_points, _ = circular_far_samples(
                all_bpms, min(NUM_WORKERS, len(all_bpms))
            )

        initial_strengths = self.mad_iface.receive_knob_values()
        self.initial_strengths = initial_strengths
        self.current_knobs = dict(zip(self.knob_names, initial_strengths))

        if RUN_ARC_BY_ARC:
            self.true_strengths = {
                knob: self.true_strengths[knob] for knob in self.knob_names
            }
        else:
            # Validate
            missing = set(self.knob_names) ^ set(self.true_strengths)
            if missing:
                raise ValueError(
                    f"Mismatch between model knobs and true strengths: {missing}"
                )

        # Set up optimiser and scheduler
        optimiser_kwargs = {
            "shape": initial_strengths.shape,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.0,
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

        if USE_NOISY_DATA is False:
            track_data = tfs.read(TRACK_DATA_FILE)
        elif FILTER_DATA:
            from aba_optimiser.config import FILTERED_FILE

            track_data = pd.read_feather(FILTERED_FILE)

            # from aba_optimiser.config import KALMAN_FILE

            # track_data = pd.read_feather(KALMAN_FILE)
        else:
            track_data = pd.read_feather(NOISE_FILE)

        # if not FILTER_DATA:
        track_data["var_x"] = (POSITION_STD_DEV) ** 2
        track_data["var_y"] = (POSITION_STD_DEV) ** 2

        # Filter data out BPMs that are not used
        self.comparison_data = select_markers(track_data, all_bpms)

        # Get the variances for each BPM
        self.var_x = (
            self.comparison_data.groupby("name", observed=True)["var_x"]
            .mean()
            .to_numpy()
        )
        self.var_y = (
            self.comparison_data.groupby("name", observed=True)["var_y"]
            .mean()
            .to_numpy()
        )

        # Convert every entry to an integer
        self.turn_list = [
            int(turn)
            for turn in self.comparison_data["turn"].unique()
            if turn < FLATTOP_TURNS * NUM_TRACKS - N_RUN_TURNS
        ]

        num_turns_needed = TRACKS_PER_WORKER
        if len(self.turn_list) < num_turns_needed:
            raise ValueError(
                f"Not enough available turns for x or y BPMs. "
                f"Need {num_turns_needed}, but found {len(self.turn_list)} turns."
            )

        # Remove turns < RAMP_UP_TURNS if ACD_ON
        if ACD_ON:
            self.turn_list = [turn for turn in self.turn_list if turn >= RAMP_UP_TURNS]

        self.smoothed_grad_norm = None  # Initialize smoothed gradient norm

    def _write_results(self, uncertainties) -> None:
        """Write final knob strengths and markdown table to file."""
        # Save final strengths
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

    def run(self) -> None:
        """Execute the optimisation loop."""
        run_start = time.time()  # start total timing
        # Randomize available turns for x and y
        tracks_per_worker = min(len(self.turn_list) // NUM_WORKERS, TRACKS_PER_WORKER)
        self.turn_list = random.sample(self.turn_list, tracks_per_worker * NUM_WORKERS)
        batches = [
            self.turn_list[i * tracks_per_worker : (i + 1) * tracks_per_worker]
            for i in range(NUM_WORKERS)
        ]
        for i, batch in enumerate(batches):
            assert batch, (
                f"Batch {i} is empty. Check TRACKS_PER_WORKER and NUM_WORKERS."
            )
        total_turns = NUM_WORKERS * tracks_per_worker * N_COMPARE_TURNS

        # Start worker processes
        parent_conns: list[Connection] = []
        workers: list[mp.Process] = []
        for i in range(NUM_WORKERS):
            batch_i = batches[i]
            parent, child = mp.Pipe()
            # start_bpm = BPM_START_POINTS[i % len(BPM_START_POINTS)]
            start_bpm = self.bpm_start_points[i % len(self.bpm_start_points)]
            w = build_worker(
                child,
                i,
                batch_i,
                self.comparison_data,
                start_bpm,
            )
            w.start()
            parent_conns.append(parent)
            workers.append(w)
            parent.send((self.var_x, self.var_y))

        # TensorBoard logging
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(log_dir=f"runs/{ts}_opt")

        try:
            # Main optimisation
            for epoch in range(MAX_EPOCHS):
                epoch_start = time.time()  # start epoch timing

                # Broadcast updated knob values, and window range to workers
                for conn in parent_conns:
                    conn.send(self.current_knobs)

                lr = self.scheduler(epoch)
                # Linearly increase learning rate for the first 100 epoch, and then reset to min

                total_loss = 0.0
                agg_grad = None

                # Collect raw per-BPM data from each worker.
                for conn in parent_conns:
                    _, grad, loss = conn.recv()
                    agg_grad = grad if agg_grad is None else agg_grad + grad
                    total_loss += loss
                total_loss /= total_turns

                agg_grad = agg_grad.flatten() / total_turns

                param_vec = np.array([self.current_knobs[k] for k in self.knob_names])
                new_vec = self.optimiser.step(param_vec, agg_grad, lr)
                self.current_knobs = dict(zip(self.knob_names, new_vec))

                grad_norm = np.linalg.norm(agg_grad)

                # —— update EMA of grad_norm ——
                if self.smoothed_grad_norm is None:
                    self.smoothed_grad_norm = grad_norm
                else:
                    self.smoothed_grad_norm = (
                        GRAD_NORM_ALPHA * self.smoothed_grad_norm
                        + (1.0 - GRAD_NORM_ALPHA) * grad_norm
                    )

                writer.add_scalar("grad_norm", grad_norm, epoch)
                true_diff = [
                    abs(self.current_knobs[k] - self.true_strengths[k])
                    for k in self.knob_names
                ]
                rel_diff = [
                    diff / abs(self.true_strengths[k])
                    if self.true_strengths[k] != 0
                    else 0
                    for k, diff in zip(self.knob_names, true_diff)
                ]

                # ----------------------------------------------
                true_diff = np.sum(true_diff)
                rel_diff = np.sum(rel_diff)
                writer.add_scalar("loss", total_loss, epoch)
                writer.add_scalar("grad_norm", grad_norm, epoch)
                writer.add_scalar("true_diff", true_diff, epoch)
                writer.add_scalar("rel_diff", rel_diff, epoch)
                writer.add_scalar("learning_rate", lr, epoch)
                writer.flush()

                epoch_time = time.time() - epoch_start
                total_time = time.time() - run_start

                LOGGER.info(
                    f"\rEpoch {epoch}: "
                    f"loss={total_loss:.3e}, "
                    f"grad_norm={grad_norm:.3e}, "
                    f"true_diff={true_diff:.3e}, "
                    f"rel_diff={rel_diff:.3e}, "
                    f"lr={lr:.3e}, "
                    f"epoch_time={epoch_time:.3f}s, "
                    f"total_time={total_time:.3f}s",
                )
                if self.smoothed_grad_norm < 1e-8:
                    LOGGER.info(
                        f"\nGradient norm below threshold: "
                        f"{self.smoothed_grad_norm:.3e}. "
                        f"Stopping early at epoch {epoch}."
                    )
                    break

        except KeyboardInterrupt:
            logging.warning(
                "\nKeyboardInterrupt detected. Terminating early and writing results."
            )
        else:
            # fig.savefig(f"runs/{ts}_relative_difference.png")
            logging.info("\nTerminating workers...")
            h_global = np.zeros((len(self.knob_names), len(self.knob_names)))
            for conn in parent_conns:
                conn.send(None)  # Signal workers to stop the training loop

            for conn in parent_conns:
                h_global += conn.recv()

            cov = np.linalg.inv(h_global)
            # Have zero uncertainty for now
            # cov = np.zeros_like(h_global)
            comb_uncertainty = np.sqrt(np.diag(cov))

            for w in workers:
                w.join()

            writer.close()
            self._write_results(comb_uncertainty)
            self._plot_results(comb_uncertainty)

    def _plot_results(self, uncertainties) -> None:
        import matplotlib.pyplot as plt
        import numpy as np

        knob_names = self.knob_names
        initial_vals = np.array(
            [self.initial_strengths[i] for i in range(len(knob_names))]
        )
        final_vals = np.array([self.current_knobs[k] for k in knob_names])
        true_vals = np.array([self.true_strengths[k] for k in knob_names])
        uncertainties = np.array(uncertainties)

        # Calculate the relative differences
        initial_relative_diff = np.abs(initial_vals - true_vals) / np.abs(true_vals)
        final_relative_diff = np.abs(final_vals - true_vals) / np.abs(true_vals)

        x = np.arange(len(knob_names))
        width = 0.35  # width of the bars

        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot both initial and final relative differences as side-by-side bars
        ax.bar(
            x - width / 2,
            initial_relative_diff,
            width,
            color="lightcoral",
            label="Initial Relative Difference",
        )
        ax.bar(
            x + width / 2,
            final_relative_diff,
            width,
            color="mediumpurple",
            label="Final Relative Difference",
        )

        ax.set_xlabel("Knob Names")
        ax.set_ylabel("Relative Difference")
        ax.set_title(
            "Initial vs Final Relative Difference between Knob Strengths and True Values"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(knob_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.savefig("plots/relative_difference_comparison.png", bbox_inches="tight")

        # Original single bar plot for final relative differences only
        plt.figure(figsize=(12, 6))
        plt.bar(
            x,
            final_relative_diff,
            width=0.5,
            color="mediumpurple",
            # yerr=relative_uncertainties,
            capsize=5,
        )
        plt.xlabel("Knob Names")
        plt.ylabel("Relative Difference")
        plt.title("Relative Difference between Final and True Knob Strengths")
        plt.xticks(x, knob_names, rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/relative_difference_plot.png", bbox_inches="tight")

        # Position plots comparing initial and final
        plt.figure(figsize=(12, 6))
        plt.plot(
            self.elem_spos,
            initial_relative_diff,
            "o",
            label="Initial Relative Difference",
            color="lightcoral",
        )
        plt.plot(
            self.elem_spos,
            final_relative_diff,
            "o",
            label="Final Relative Difference",
            color="mediumpurple",
        )
        plt.xlabel("Element Position")
        plt.ylabel("Relative Difference")
        plt.title("Initial vs Final Relative Difference vs Element Position")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(
            "plots/relative_difference_vs_position_comparison.png", bbox_inches="tight"
        )

        # Original position plot for final only
        plt.figure()
        # plt.plot(self.spos_list, relative_uncertainties, "o", label="Uncertainty")
        plt.plot(self.elem_spos, final_relative_diff, "o", label="Relative Difference")
        plt.xlabel("Element Position")
        plt.ylabel("Value")
        # plt.title("Uncertainty and Relative Difference vs Element Position")
        plt.title("Relative Difference vs Element Position")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.savefig("plots/relative_difference_vs_position.png", bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    ctrl = Controller()
    ctrl.run()
