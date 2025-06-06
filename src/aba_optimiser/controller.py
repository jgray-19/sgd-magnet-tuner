import datetime
import multiprocessing as mp
import time
from multiprocessing.connection import Connection

import numpy as np
import tfs
from tensorboardX import SummaryWriter
import pandas as pd

from aba_optimiser.config import (
    ACD_ON,
    BPM_RANGE,
    DECAY_EPOCHS,
    ELEM_NAMES_FILE,
    FILTER_DATA,
    FILTERED_FILE,
    GRAD_NORM_ALPHA,
    # GRAD_PENALTY_COEFF,
    # REL_K1_STD_DEV,
    KNOB_TABLE,
    MAX_EPOCHS,
    MAX_LR,
    MIN_LR,
    TRACK_DATA_FILE,
    # MOMENTUM_STD_DEV,
    # MIN_FRACTION_MAX,
    NOISE_FILE,
    NUM_WORKERS,
    OPTIMISER_TYPE,
    OUTPUT_KNOBS,
    POSITION_STD_DEV,
    RAMP_UP_TURNS,
    SEQUENCE_FILE,
    TOTAL_TRACKS,
    TRACKS_PER_WORKER,
    TRUE_STRENGTHS,
    USE_NOISY_DATA,
    WARMUP_EPOCHS,
    WARMUP_LR_START,
    WINDOWS,
    X_BPM_START,
    Y_BPM_START,
    XY_MIN,
    PXPY_MIN,
)
from aba_optimiser.mad_interface import MadInterface
from aba_optimiser.adam import AdamOptimiser
from aba_optimiser.amsgrad import AMSGradOptimiser
from aba_optimiser.scheduler import LRScheduler
from aba_optimiser.utils import (
    read_elem_names,
    read_knobs,
    save_results,
    scientific_notation,
    select_marker,
)
from aba_optimiser.worker import Worker


class Controller:
    """
    Orchestrates multi-process knob optimisation using MAD-NG.
    """

    def __init__(self):
        # Read element names using the util function; each line now becomes a list of aliases.
        self.elem_pos, _ = read_elem_names(ELEM_NAMES_FILE)
        # Read true strengths
        self.true_strengths = read_knobs(TRUE_STRENGTHS)

        # Initialise MAD interface and knobs
        self.mad_iface = MadInterface(SEQUENCE_FILE, BPM_RANGE)
        self.knob_names = self.mad_iface.knob_names

        # Run a twiss and to get the beta functions
        tws = self.mad_iface.run_twiss()
        print("Found tunes:", tws.q1, tws.q2)

        # Remove all rows that are before the start BPM and after the end BPM
        start_bpm, end_bpm = BPM_RANGE.split("/")
        start_bpm_idx = tws.index.get_loc(start_bpm)
        end_bpm_idx = tws.index.get_loc(end_bpm)
        tws = tws.iloc[start_bpm_idx : end_bpm_idx + 1]

        self.all_bpms = tws.index.tolist()
        self.beta_x = tws["beta11"].to_numpy()
        self.beta_y = tws["beta22"].to_numpy()

        assert len(self.beta_x) == len(self.beta_y), (
            "Beta functions are not the same length"
        )
        assert len(self.beta_x) == self.mad_iface.nbpms, (
            "Beta functions do not match the number of BPMs"
        )

        init_vals = self.mad_iface.receive_knob_values()
        self.current_knobs = dict(zip(self.knob_names, init_vals))

        # Validate
        missing = set(self.knob_names) ^ set(self.true_strengths)
        if missing:
            raise ValueError(
                f"Mismatch between model knobs and true strengths: {missing}"
            )

        # Set up optimiser and scheduler
        optimiser_kwargs = {
            "shape": init_vals.shape,
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
            track_data = pd.read_feather(FILTERED_FILE)
        else:
            track_data = pd.read_feather(NOISE_FILE)

        # if not FILTER_DATA:
        track_data["var_x"] = (POSITION_STD_DEV) ** 2
        track_data["var_y"] = (POSITION_STD_DEV) ** 2

        try:
            start_idx = self.all_bpms.index(start_bpm)
            end_idx = self.all_bpms.index(end_bpm)
        except ValueError:
            raise ValueError("Start or end BPM not found in the BPM list.")

        # Get the markers to keep
        markers_to_keep = self.all_bpms[start_idx : end_idx + 1]
        assert "BPM" in end_bpm, "End BPM not a BPM"
        assert "BPM" in start_bpm, "Start BPM not a BPM"

        # Filter data out BPMs that are not used
        self.comparison_data = track_data[
            track_data["name"].isin(markers_to_keep)
        ].copy()
        del track_data  # Free memory
        
        self.bpm_names = list(self.comparison_data["name"].unique())
        assert len(self.bpm_names) == self.mad_iface.nbpms, (
            "BPM names from track data do not match the number of BPMs in the sequence"
        )

        # Get the variances for each BPM
        self.var_x = (
            self.comparison_data.groupby("name", observed=True)["var_x"].mean().to_numpy()
        )
        self.var_y = (
            self.comparison_data.groupby("name", observed=True)["var_y"].mean().to_numpy()
        )
        print("Average variances for x and y BPMs:")

        self.bpm_idx = {name: i for i, name in enumerate(self.bpm_names)}
        self.window_slices: dict[str, slice] = {}
        for start, end in WINDOWS:
            assert start not in self.window_slices.keys(), (
                f"Start BPM {start} already exists in window slices."
            )
            i0 = self.bpm_idx[start]
            i1 = self.bpm_idx[end]
            assert i0 < i1, 'Start BPM is after or the same as End BPM'
            self.window_slices[start] = slice(i0, i1 + 1)

        self.smoothed_grad_norm = None  # Initialize smoothed gradient norm
        
        # no_noise_data = tfs.read(TRACK_DATA_FILE)
        # no_noise_data["var_x"] = (POSITION_STD_DEV) ** 2
        # no_noise_data["var_y"] = (POSITION_STD_DEV) ** 2
        # start_bpm_df_x = no_noise_data.loc[
        #     no_noise_data["name"] == X_BPM_START
        # ]
        # start_bpm_df_y = no_noise_data.loc[
        #     no_noise_data["name"] == Y_BPM_START
        # ]
        # del no_noise_data  # Free memory

        start_bpm_df_x = self.comparison_data.loc[
            self.comparison_data["name"] == X_BPM_START
        ]
        start_bpm_df_y = self.comparison_data.loc[
            self.comparison_data["name"] == Y_BPM_START
        ]
        self.start_data_dict = {
            X_BPM_START: select_marker(start_bpm_df_x, X_BPM_START),
            Y_BPM_START: select_marker(start_bpm_df_y, Y_BPM_START)
        }

        def _cut_coordinates(df: pd.DataFrame, coord: str) -> list[int]:
            x_min = XY_MIN if coord == "x" else XY_MIN / 2
            y_min = XY_MIN if coord == "y" else XY_MIN / 2
            px_min = PXPY_MIN if coord == "x" else PXPY_MIN / 2
            py_min = PXPY_MIN if coord == "y" else PXPY_MIN / 2
            return df.loc[
                (abs(df["x"]) > x_min)
                & (abs(df["y"]) > y_min)
                & (abs(df["px"]) > px_min)
                & (abs(df["py"]) > py_min),
                "turn",
            ].tolist()

        self.available_x = _cut_coordinates(start_bpm_df_x, "x")
        self.available_y = _cut_coordinates(start_bpm_df_y, "y")

        # Convert every entry to an integer
        self.available_x = [int(turn) for turn in self.available_x]
        self.available_y = [int(turn) for turn in self.available_y]

        num_turns_needed = NUM_WORKERS * TRACKS_PER_WORKER
        if (
            len(self.available_x) < num_turns_needed
            and len(self.available_y) < num_turns_needed
        ):
            raise ValueError(
                f"Not enough available turns for x or y BPMs. "
                f"Need {num_turns_needed}, but found {len(self.available_x)} for x and {len(self.available_y)} for y."
            )
        print(
            f"Available x turns: {len(self.available_x)}, from {len(start_bpm_df_x)} total x turns.\n"
            f"Available y turns: {len(self.available_y)}, from {len(start_bpm_df_y)} total y turns."
        )

        # Remove turns < RAMP_UP_TURNS if ACD_ON
        if ACD_ON:
            self.available_x = [
                turn for turn in self.available_x if turn >= RAMP_UP_TURNS
            ]
            self.available_y = [
                turn for turn in self.available_y if turn >= RAMP_UP_TURNS
            ]

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

        # Write the markdown table with an extra index column
        with open(KNOB_TABLE, "w") as f:
            f.write(
                "| Index |   Knob   |   True   |   Diff   | Uncertainty | Relative Diff | Relative Uncertainty |\n"
            )
            f.write(
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
        print("Optimization complete.")

    def run(self) -> None:
        """Execute the optimisation loop."""
        run_start = time.time()  # start total timing
        x_tracks_per_worker = min(len(self.available_x) // NUM_WORKERS, TRACKS_PER_WORKER)
        y_tracks_per_worker = min(len(self.available_y) // NUM_WORKERS, TRACKS_PER_WORKER)
        x_batches = [
            self.available_x[i * x_tracks_per_worker : (i + 1) * x_tracks_per_worker]
            for i in range(NUM_WORKERS)
        ]
        y_batches = [
            self.available_y[i * y_tracks_per_worker : (i + 1) * y_tracks_per_worker]
            for i in range(NUM_WORKERS)
        ]
        for i, batch in enumerate(x_batches + y_batches):
            assert batch, (
                f"Batch {i} is empty. Check TRACKS_PER_WORKER and NUM_WORKERS."
            )
        total_tracks = NUM_WORKERS * (x_tracks_per_worker + y_tracks_per_worker) / 2

        # Start worker processes
        parent_conns: list[Connection] = []
        workers: list[mp.Process] = []
        for i in range(NUM_WORKERS):
            batch_x = x_batches[i]
            batch_y = y_batches[i]
            parent, child = mp.Pipe()
            w = Worker(
                child,
                i,
                batch_x,
                batch_y,
                self.comparison_data,
                self.start_data_dict,
                self.window_slices,
            )
            w.start()
            parent_conns.append(parent)
            workers.append(w)
            parent.send((self.var_x, self.var_y))

        # TensorBoard logging
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(log_dir=f"runs/{ts}_opt")

        # Initialize live plot for relative differences (log scale)
        # import matplotlib.pyplot as plt

        # plt.ion()
        # rel_diff_history = {k: [] for k in self.knob_names}
        # fig, ax = plt.subplots()
        # Initialize new figure for consecutive differences (delta)
        # fig_delta, ax_delta = plt.subplots()

        # current_cell = 0
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
                total_loss /= total_tracks

                agg_grad = agg_grad.flatten() / total_tracks

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
                # ----- Live plot update (non-blocking) -----
                # for i, k in enumerate(self.knob_names):
                #     rel_diff_history[k].append(rel_diff[i])
                # if (epoch + 1) % 10 == 0:
                #     # Update log scale figure
                #     ax.clear()
                #     for i, k in enumerate(self.knob_names):
                #         ax.plot(rel_diff_history[k], label=str(k))
                #         if rel_diff_history[k]:
                #             x_val = len(rel_diff_history[k]) - 1
                #             ax.text(
                #                 x_val,
                #                 rel_diff_history[k][-1],
                #                 str(i),
                #                 ha="left",
                #                 va="center",
                #                 fontsize="small",
                #             )
                #     ax.set_xlabel("Epoch")
                #     ax.set_ylabel("Relative Difference")
                #     ax.set_yscale("log", nonpositive="clip")
                #     ax.set_ylim(1e-6, 1e-3)

                #     # Update delta (difference) plot to show only the last 300 results
                #     ax_delta.clear()
                #     for i, k in enumerate(self.knob_names):
                #         # Use only last 300 results, if available
                #         history = rel_diff_history[k]
                #         if len(history) > 300:
                #             history_tail = history[-300:]
                #             x_vals = range(len(history) - 300, len(history) - 1)
                #         else:
                #             history_tail = history
                #             x_vals = range(1, len(history))
                #         # Calculate differences between consecutive elements.
                #         diffs = np.diff(history_tail)
                #         if diffs.size > 0:
                #             ax_delta.plot(x_vals, diffs)
                #             ax_delta.text(
                #                 x_vals[-1],
                #                 diffs[-1],
                #                 str(i),
                #                 ha="left",
                #                 va="center",
                #                 fontsize="small",
                #             )
                #     # Draw a horizontal line at y=0
                #     ax_delta.axhline(0, color="red", linestyle="--")
                #     ax_delta.set_xlabel("Epoch (delta index)")

                # plt.pause(0.001)

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

                print(
                    f"\rEpoch {epoch}: "
                    f"loss={total_loss:.3e}, "
                    f"grad_norm={grad_norm:.3e}, "
                    f"true_diff={true_diff:.3e}, "
                    f"rel_diff={rel_diff:.3e}, "
                    f"lr={lr:.3e}, "
                    f"epoch_time={epoch_time:.3f}s, "
                    f"total_time={total_time:.3f}s",
                    end="\n",
                )

                if self.smoothed_grad_norm < 5e-7:
                    print(
                        f"\nGradient norm below threshold: "
                        f"{self.smoothed_grad_norm:.3e}. "
                        f"Stopping early at epoch {epoch}."
                    )
                    break

        except KeyboardInterrupt:
            print(
                "\nKeyboardInterrupt detected. Terminating early and writing results."
            )
        else:
            # fig.savefig(f"runs/{ts}_relative_difference.png")
            print("\nTerminating workers...")
            H_global = np.zeros((len(self.knob_names), len(self.knob_names)))
            for conn in parent_conns:
                conn.send(None)  # Signal workers to stop the training loop

            for conn in parent_conns:
                H_global += conn.recv()

            cov = np.linalg.inv(H_global)
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
        final_vals = np.array([self.current_knobs[k] for k in knob_names])
        true_vals = np.array([self.true_strengths[k] for k in knob_names])
        uncertainties = np.array(uncertainties)

        # Calculate the relative difference between final and true values
        # and scale uncertainties accordingly (avoid division by zero)
        relative_diff = np.abs(final_vals - true_vals) / np.abs(true_vals)
        relative_uncertainties = np.abs(uncertainties) / np.abs(true_vals)

        x = np.arange(len(knob_names))
        width = 0.5  # width of the bars

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the relative difference as a single bar with error bars set to relative uncertainties
        ax.bar(
            x,
            relative_diff,
            width,
            color="mediumpurple",
            yerr=relative_uncertainties,
            capsize=5,
        )

        ax.set_xlabel("Knob Names")
        ax.set_ylabel("Relative Difference")
        ax.set_title("Relative Difference between Final and True Knob Strengths")
        ax.set_xticks(x)
        ax.set_xticklabels(knob_names, rotation=45, ha="right")

        plt.figure()
        # plt.plot(self.elem_pos, relative_uncertainties, "o", label="Uncertainty")
        plt.plot(self.elem_pos, relative_diff, "o", label="Relative Difference")
        plt.xlabel("Element Position")
        plt.ylabel("Value")
        # plt.title("Uncertainty and Relative Difference vs Element Position")
        plt.title("Relative Difference vs Element Position")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    ctrl = Controller()
    ctrl.run()
