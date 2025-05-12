import datetime
import time
import multiprocessing as mp
from multiprocessing.connection import Connection

import numpy as np
import tfs
from tensorboardX import SummaryWriter

from aba_optimiser.config import (
    BPM_RANGE,
    DECAY_EPOCHS,
    ELEM_NAMES_FILE,
    GRAD_NORM_ALPHA,
    KNOB_TABLE,
    MAX_EPOCHS,
    MAX_LR,
    MIN_LR,
    NUM_WORKERS,
    OUTPUT_KNOBS,
    RAMP_UP_TURNS,
    SEQUENCE_FILE,
    SEQ_NAME,
    TOTAL_TRACKS,
    # TRACK_DATA_FILE,
    NOISE_FILE,
    TRACKS_PER_WORKER,
    TRUE_STRENGTHS,
    WARMUP_EPOCHS,
    WARMUP_LR_START,
    WINDOWS,
)
from aba_optimiser.mad_interface import MadInterface
from aba_optimiser.optimiser import AdamOptimiser
from aba_optimiser.scheduler import LRScheduler
from aba_optimiser.utils import (
    read_elem_names,
    read_knobs,
    save_results,
    scientific_notation,
    select_marker
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
        self.mad_iface.mad.send(f"""
tws = twiss{{sequence=MADX.{SEQ_NAME}, observe=1}}
        """)
        tws: tfs.TfsDataFrame = self.mad_iface.mad.tws.to_df().set_index("name")
        print("Found tunes:", tws.q1, tws.q2)
        
        # Remove all rows that are before the start BPM and after the end BPM
        start_bpm, end_bpm = BPM_RANGE.split("/")
        start_bpm_idx = tws.index.get_loc(start_bpm)
        end_bpm_idx = tws.index.get_loc(end_bpm)
        tws = tws.iloc[start_bpm_idx:end_bpm_idx + 1]

        self.all_bpms = tws.index.tolist()
        self.beta_x = np.sqrt(tws["beta11"].to_numpy())
        self.beta_y = np.sqrt(tws["beta22"].to_numpy())

        assert len(self.beta_x) == len(self.beta_y), "Beta functions are not the same length"
        assert len(self.beta_x) == self.mad_iface.nbpms, "Beta functions do not match the number of BPMs"

        init_vals = self.mad_iface.receive_knob_values()
        self.current_knobs = dict(zip(self.knob_names, init_vals))

        # Validate
        missing = set(self.knob_names) ^ set(self.true_strengths)
        if missing:
            raise ValueError(
                f"Mismatch between model knobs and true strengths: {missing}"
            )

        # Set up optimiser and scheduler
        self.optimiser = AdamOptimiser(
            shape=init_vals.shape,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.0,
        )
        self.scheduler = LRScheduler(
            warmup_epochs=WARMUP_EPOCHS,
            decay_epochs=DECAY_EPOCHS,
            start_lr=WARMUP_LR_START,
            max_lr=MAX_LR,
            min_lr=MIN_LR,
        )

        # track_data = tfs.read(TRACK_DATA_FILE)
        track_data = tfs.read(NOISE_FILE)

        self.start_data = select_marker(track_data, start_bpm)
        # Get indices of the start and end markers from the full list.
        try:
            start_idx = self.all_bpms.index(start_bpm)
            end_idx = self.all_bpms.index(end_bpm)
        except ValueError:
            raise ValueError("Start or end BPM not found in the BPM list.")

        # Get the markers to keep
        markers_to_keep = self.all_bpms[start_idx+1:end_idx]
        if "BPM" in start_bpm:
            markers_to_keep = [start_bpm] + markers_to_keep
        if "BPM" in end_bpm:
            markers_to_keep = markers_to_keep + [end_bpm]

        # Filter data using pandas' vectorised operation
        self.comparison_data = track_data[track_data["name"].isin(markers_to_keep)].copy()
        self.bpm_names = list(self.comparison_data["name"].unique())
        assert len(self.bpm_names) == self.mad_iface.nbpms, "BPM names from track data do not match the number of BPMs in the sequence"

        self.bpm_idx = {name: i for i, name in enumerate(self.bpm_names)}
        self.window_slices = []
        for start, end in WINDOWS:
            i0 = self.bpm_idx[start]
            i1 = self.bpm_idx[end]
            if i1 < i0:
                i0, i1 = i1, i0
            self.window_slices.append(slice(i0, i1 + 1))

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
        
        # Prepare worker batches
        indices = list(range(RAMP_UP_TURNS, TOTAL_TRACKS + RAMP_UP_TURNS + 1))
        batches = [
            indices[i * TRACKS_PER_WORKER : (i + 1) * TRACKS_PER_WORKER]
            for i in range(NUM_WORKERS)
        ]
        for i, batch in enumerate(batches):
            assert batch, f"Batch {i} is empty. Check TRACKS_PER_WORKER and NUM_WORKERS."

        # Start worker processes
        parent_conns: list[Connection] = []
        workers: list[mp.Process] = []
        for i, batch in enumerate(batches):
            parent, child = mp.Pipe()
            w = Worker(i, batch, child, self.comparison_data, self.start_data, self.beta_x, self.beta_y)
            w.start()
            parent_conns.append(parent)
            workers.append(w)

        # TensorBoard logging
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(log_dir=f"runs/{ts}_opt")
        # previous_loss = 1e100  # Initialize previous loss tracker
        try:
            # Main optimisation
            for epoch in range(MAX_EPOCHS):
                epoch_start = time.time()  # start epoch timing

                # choose sub-arc for this epoch
                win = epoch % len(self.window_slices)
                slc = self.window_slices[win]

                # Broadcast updated knob values, and window range to workers
                for conn in parent_conns:
                    conn.send((self.current_knobs, slc))

                lr = self.scheduler(epoch)

                total_loss = 0.0
                agg_grad = None

                # Collect raw per-BPM data from each worker.
                for conn in parent_conns:
                    _, grad, loss = conn.recv()
                    agg_grad = grad if agg_grad is None else agg_grad + grad
                    total_loss += loss

                agg_grad = agg_grad.flatten() / TOTAL_TRACKS

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
                    end="",
                )

                if self.smoothed_grad_norm < 1e-7:
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
            print("\nTerminating workers...")
            H_global = np.zeros((len(self.knob_names), len(self.knob_names)))
            for conn in parent_conns:
                conn.send((None, None))  # Signal workers to stop
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
