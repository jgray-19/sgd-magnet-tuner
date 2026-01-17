"""Beta function matcher for minimising difference between model and computed betas.

This module provides the main entry point for matching beta functions computed
from estimated magnet strengths to a target model by adjusting knob strengths.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import tfs

    from aba_optimiser.matching.matcher_config import MatcherConfig

logger = logging.getLogger(__name__)


class BetaMatcher:
    """
    Matches computed beta functions to a target model by adjusting knob strengths.

    This class is designed to be run after the Controller has estimated the main
    quadrupole strengths from measurement. It takes:
    - A target model twiss (the betas we want to achieve)
    - The estimated quadrupole strengths from the Controller
    - A list of knobs that can be adjusted

    The workflow is:
    1. Apply estimated strengths to compute current betas
    2. Adjust knobs to minimise difference between computed betas and target model betas
    """

    def __init__(
        self,
        config: MatcherConfig,
        show_plots: bool = True,
    ):
        """
        Initialise the BetaMatcher with configuration.

        Args:
            config: Matcher configuration containing:
                - model_twiss_file: Path to target model twiss TFS file
                - estimated_strengths_file: Path to JSON with controller's estimated strengths
                - knobs_file: Path to JSON listing knobs to adjust
                - sequence_file_path: Path to MAD-NG sequence file
                - magnet_range: Range of magnets for matching
                - beam_energy: Beam energy in GeV
                - output_dir: Directory to save results
            show_plots: Whether to display plots during matching
        """
        self.config = config
        self.show_plots = show_plots

        # Validate configuration
        self.config.validate()

        # Load target model twiss (contains BETX, BETY columns) from TFS file
        self.model_twiss = self._load_model_twiss()

        # Load the list of knobs to be adjusted
        self.knobs = self.config.knobs_list
        self.tune_knobs = self.config.tune_knobs.copy()

        # Load the estimated quadrupole strengths using config.get_estimated_strengths()
        self.estimated_strengths = self.config.get_estimated_strengths()

        logger.info("Initialising BetaMatcher")
        logger.info(f"  Target model twiss: {config.model_twiss_file}")
        logger.info(
            f"  Estimated strengths: {'dict in memory' if isinstance(config.estimated_strengths, dict) else config.estimated_strengths}"
        )
        logger.info(f"  Beta knobs: {len(self.knobs)} knobs")
        logger.info(f"  Tune knobs: {len(self.tune_knobs)} knobs")
        logger.info(f"  Sequence file: {config.sequence_file_path}")
        logger.info(f"  Magnet range: {config.magnet_range}")

        # Initialise the MAD-NG interface for computing betas
        self._init_mad_interface()

    def run(self) -> tuple[dict[str, float], dict[str, float]]:
        """
        Execute the beta matching procedure using MAD-NG match command.

        Returns:
            Tuple of (final_knob_values, uncertainties) where:
                - final_knob_values: Dictionary mapping knob names to optimised values
                - uncertainties: Dictionary mapping knob names to their uncertainties
        """
        logger.info("Starting beta matching procedure")

        # Get list of BPMs in the magnet range
        bpm_names = self._get_bpm_list()
        logger.info(f"Found {len(bpm_names)} BPMs in range {self.config.magnet_range}")

        # Prepare variables (beta knobs + tune knobs)
        variables = []
        # Add knobs
        for knob_name in self.knobs + list(self.tune_knobs.keys()):
            variables.append({"var": f"\"MADX['{knob_name}']\"", "name": f"'{knob_name}'"})

        # Prepare equalities (beta constraints for all BPMs)
        equalities = []
        for bpm_name in bpm_names:
            # Get target betas from model twiss
            if bpm_name not in self.model_twiss.index:
                logger.warning(f"BPM {bpm_name} not found in model twiss, skipping")
                continue

            target_betx = self.model_twiss.loc[bpm_name, "beta11"]
            target_bety = self.model_twiss.loc[bpm_name, "beta22"]

            # Add BETX constraint with 1 % relative tolerance
            equalities.append(
                {
                    "expr": f"\\t -> t['{bpm_name}'].beta11 - {target_betx:.16e}",
                    "name": f"'betx_{bpm_name}'",
                    "tol": 1e-3 * target_betx,
                    "kind": "'beta'",
                }
            )

            # Add BETY constraint
            equalities.append(
                {
                    "expr": f"\\t -> t['{bpm_name}'].beta22 - {target_bety:.16e}",
                    "name": f"'bety_{bpm_name}'",
                    "tol": 1e-3 * target_bety,
                    "kind": "'beta'",
                }
            )

        # Add tune constraints
        equalities.append(
            {
                "expr": f"\\t -> t.q1 - {self.target_q1:.16e}",
                "name": "'q1'",
                "tol": 1e-3,  # 0.1% tolerance on tune
                "kind": "'mu'",
            }
        )
        equalities.append(
            {
                "expr": f"\\t -> t.q2 - {self.target_q2:.16e}",
                "name": "'q2'",
                "tol": 1e-3,  # 0.1% tolerance on tune
                "kind": "'mu'",
            }
        )

        logger.info(
            f"Matching {len(variables)} knobs ({len(self.knobs)} beta + {len(self.tune_knobs)} tune) to {len(equalities)} constraints ({len(equalities) - 2} beta + 2 tune)"
        )

        # Create the twiss and send to MAD-NG
        self.mad_interface.mad.send(r"""
local total_fstp = 0.0
function twiss_and_send(cmd)
    local tbl, flow = twiss {sequence=loaded_sequence, observe=1}
    local fstp = cmd["__var"]["fstp"]
    if fstp > 0 then
        total_fstp = total_fstp + fstp
    end

    if total_fstp > 1e-4 then
        py:send({tbl.s, tbl.beta11, tbl.beta22}, true)
        total_fstp = 0.0
    end
    return tbl, flow
end
    """)
        # Run the match
        self.mad_interface.mad.match(
            command=self.mad_interface.mad.twiss_and_send,
            variables=variables,
            equalities=equalities,
            info=2,
            maxcall=300,
        )
        self.mad_interface.mad.send("py:send('match complete')")
        target_betx = self.model_twiss["beta11"]
        target_bety = self.model_twiss["beta22"]

        res = self._flatten_all_vectors(self.mad_interface.mad.recv())
        # Start a plot that updates with each iteration
        from matplotlib import pyplot as plt

        assert np.allclose(res[0], self.model_twiss["s"]), "s positions do not match"
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        plt.ion()
        plt.show(block=False)
        while isinstance(res, list):
            res = self._flatten_all_vectors(res)
            s_positions, betax_vals, betay_vals = res
            betax_rel_diff = (betax_vals - target_betx) / target_betx * 100
            betay_rel_diff = (betay_vals - target_bety) / target_bety * 100
            ax.clear()
            ax.plot(s_positions, betax_rel_diff, label="BETX relative error [%]")
            ax.plot(s_positions, betay_rel_diff, label="BETY relative error [%]")
            ax.set_ylabel("Relative error [%]")
            ax.set_xlabel("s position [m]")
            ax.legend()
            ax.set_title("Beta Matching Progress")
            plt.pause(0.01)
            res = self.mad_interface.mad.recv()
        plt.ioff()

        assert res == "match complete"
        logger.info("Beta matching completed successfully")

        # Extract results
        final_knobs = {knob: self.mad_interface.mad[f"MADX['{knob}']"] for knob in self.knobs}
        for knob_name in self.tune_knobs:
            final_knobs[knob_name] = self.mad_interface.mad[f"MADX['{knob_name}']"]
        uncertainties = {}  # TODO: Extract uncertainties from match result

        logger.info("Beta matching completed")
        return final_knobs, uncertainties

    def run_lbfgs_match(self) -> tuple[dict[str, float], dict[str, float]]:
        """
        Execute the beta matching procedure using LBFGS optimisation.

        This method uses the LBFGS optimiser to minimise the sum of RMS differences
        of beta functions from the target model by adjusting knob strengths.

        Returns:
            Tuple of (final_knob_values, uncertainties) where:
                - final_knob_values: Dictionary mapping knob names to optimised values
                - uncertainties: Dictionary mapping knob names to their uncertainties
        """
        logger.info("Starting LBFGS beta matching procedure")

        # Get BPMs in range and filter to those in model twiss
        bpm_names = self._get_bpm_list()
        logger.info(f"Found {len(bpm_names)} BPMs in range {self.config.magnet_range}")
        bpm_names_filtered = [bpm for bpm in bpm_names if bpm in self.model_twiss.index]

        # Prepare knob lists and initial values
        all_knobs = self.knobs + list(self.tune_knobs.keys())
        knobs_list = all_knobs
        initial_values = np.array(
            [self.mad_interface.mad[f"MADX['{knob}']"] for knob in knobs_list]
        )

        # Extract target betas for filtered BPMs
        target_betax = np.array([self.model_twiss.loc[bpm, "beta11"] for bpm in bpm_names_filtered])
        target_betay = np.array([self.model_twiss.loc[bpm, "beta22"] for bpm in bpm_names_filtered])
        n = len(bpm_names_filtered)

        def objective_and_grad(x):
            """Compute objective (sum of RMS beta differences + tune penalty) and its gradient."""
            # Set knob values in MAD-NG
            for i, knob in enumerate(knobs_list):
                self.mad_interface.mad[f"MADX['{knob}']"] = x[i]

            # Compute twiss without derivatives first
            twiss_df_no_deriv = self._compute_twiss_without_derivatives()
            bpm_indices = [twiss_df_no_deriv.index.get_loc(bpm) for bpm in bpm_names_filtered]

            # Extract computed betas
            betax = twiss_df_no_deriv["beta11"].values[bpm_indices]
            betay = twiss_df_no_deriv["beta22"].values[bpm_indices]

            # Compute RMS differences
            diffx = betax - target_betax
            diffy = betay - target_betay
            rms_x = np.sqrt(np.mean(diffx**2))
            rms_y = np.sqrt(np.mean(diffy**2))
            f = rms_x + rms_y

            # Add tune penalty
            current_q1 = twiss_df_no_deriv.headers["q1"]
            current_q2 = twiss_df_no_deriv.headers["q2"]
            tune_penalty = 10 * (
                abs(current_q1 - self.target_q1) + abs(current_q2 - self.target_q2)
            )
            f += tune_penalty

            # Check if to recompute derivatives
            if self.cached_loss is None or abs(f - self.cached_loss) >= 0.1:
                # Recompute with derivatives
                twiss_df, kopt_list = self._compute_twiss_with_derivatives(knobs_list)
                self.cached_derivatives = (twiss_df, kopt_list)
                self.cached_loss = f
            else:
                # Use cached derivatives
                twiss_df, kopt_list = self.cached_derivatives

            # Now compute gradient using twiss_df
            bpm_indices = [twiss_df.index.get_loc(bpm) for bpm in bpm_names_filtered]
            nknobs = len(knobs_list)

            # Vectorized computation for beta gradients
            derivatives_beta_x = np.array(
                [twiss_df[kopt_list[0][j]].values[bpm_indices] for j in range(nknobs)]
            ).T  # (N, nknobs)
            derivatives_beta_y = np.array(
                [twiss_df[kopt_list[1][j]].values[bpm_indices] for j in range(nknobs)]
            ).T  # (N, nknobs)

            if rms_x > 0:
                d_rms_x_all = (1 / (n * rms_x)) * np.sum(
                    diffx[:, None] * derivatives_beta_x, axis=0
                )
            else:
                d_rms_x_all = np.zeros(nknobs)

            if rms_y > 0:
                d_rms_y_all = (1 / (n * rms_y)) * np.sum(
                    diffy[:, None] * derivatives_beta_y, axis=0
                )
            else:
                d_rms_y_all = np.zeros(nknobs)

            grad = d_rms_x_all + d_rms_y_all

            # Add gradient from tune penalty (scalar, so loop)
            for j in range(len(knobs_list)):
                # Add gradient from tune penalty (using phase advance at sequence end)
                last_element = twiss_df.index[-1]  # Should be the end marker (e.g., "$end")
                assert last_element == "$end", "Last element should be $end for tune derivative"
                d_mu1_end_d_knob = twiss_df[kopt_list[0][nknobs + j]].loc[last_element]
                d_mu2_end_d_knob = twiss_df[kopt_list[1][nknobs + j]].loc[last_element]
                d_q1_d_knob = d_mu1_end_d_knob / (2 * np.pi)
                d_q2_d_knob = d_mu2_end_d_knob / (2 * np.pi)
                d_tune_penalty_d_knob = 10 * (
                    np.sign(current_q1 - self.target_q1) * d_q1_d_knob
                    + np.sign(current_q2 - self.target_q2) * d_q2_d_knob
                )
                grad[j] += d_tune_penalty_d_knob

            return f, grad

        # Run custom LBFGS with learning rate scheduler
        from aba_optimiser.optimisers.lbfgs import LBFGSOptimiser
        from aba_optimiser.training.scheduler import LRScheduler

        optimiser = LBFGSOptimiser(history_size=20, use_adaptive_lr=True)
        scheduler = LRScheduler(
            warmup_epochs=10,
            decay_epochs=0,
            start_lr=1e-13,
            max_lr=3,
            min_lr=3,
        )
        max_iter = 200
        x = initial_values.copy()
        start_time = time.time()
        loss = np.inf
        for iteration in range(max_iter):
            iter_start = time.time()
            f, grad = objective_and_grad(x)
            lr = scheduler(iteration)
            x = optimiser.step(x, grad, lr)
            iter_time = time.time() - iter_start
            total_time = time.time() - start_time
            logger.info(
                f"Iteration {iteration + 1}: loss={f:.5f}, lr={lr:.2e}, param_norm={np.linalg.norm(x):.4e}, itertime={iter_time:.2f}s, totaltime={total_time:.1g}s"
            )
            if f < 0.1 or abs(loss - f) < 1e-4:
                logger.info("Convergence achieved")
                break
            loss = f
        logger.info(f"LBFGS optimisation completed in {iteration + 1} iterations")

        # Use final x
        result = type("Result", (), {"x": x, "success": True, "message": "Converged"})()

        # Set and return final knob values
        final_knobs = {knobs_list[i]: result.x[i] for i in range(len(knobs_list))}
        for knob, value in final_knobs.items():
            self.mad_interface.mad[f"MADX['{knob}']"] = value

        uncertainties = {}  # LBFGS doesn't provide uncertainties directly
        logger.info("LBFGS beta matching completed")
        return final_knobs, uncertainties

    def run_linear_match(self, n_steps: int = 1, svd_cutoff: float | None = None) -> tuple[dict[str, float], dict[str, float]]:
        """
        Execute the beta matching procedure using linear response matrix inversion.

        This method assumes the problem is approximately linear and computes the response
        matrix of beta functions and tunes with respect to knobs. It then inverts the
        response matrix to solve for the knob adjustments in multiple steps if specified.

        Args:
            n_steps: Number of iterative linear correction steps to perform.
            svd_cutoff: If provided, use SVD with singular value cutoff for regularization.
                        Singular values below svd_cutoff * max(singular_value) are set to zero.

        Returns:
            Tuple of (final_knob_values, uncertainties) where:
                - final_knob_values: Dictionary mapping knob names to optimised values
                - uncertainties: Dictionary mapping knob names to their uncertainties
        """
        logger.info(f"Starting linear beta matching procedure with {n_steps} steps")

        # Get BPMs in range and filter to those in model twiss
        bpm_names = self._get_bpm_list()
        logger.info(f"Found {len(bpm_names)} BPMs in range {self.config.magnet_range}")
        bpm_names_filtered = [bpm for bpm in bpm_names if bpm in self.model_twiss.index]

        # Prepare knob lists
        all_knobs = self.knobs + list(self.tune_knobs.keys())
        knobs_list = all_knobs
        nknobs = len(knobs_list)

        # Extract target betas for filtered BPMs
        target_betax = np.array([self.model_twiss.loc[bpm, "beta11"] for bpm in bpm_names_filtered])
        target_betay = np.array([self.model_twiss.loc[bpm, "beta22"] for bpm in bpm_names_filtered])

        for step in range(n_steps):
            logger.info(f"Linear matching step {step + 1}/{n_steps}")

            # Compute twiss with derivatives at current point
            twiss_df, kopt_list = self._compute_twiss_with_derivatives(knobs_list)
            bpm_indices = [twiss_df.index.get_loc(bpm) for bpm in bpm_names_filtered]

            # Extract current betas and tunes
            betax_current = twiss_df["beta11"].values[bpm_indices]
            betay_current = twiss_df["beta22"].values[bpm_indices]
            current_q1 = twiss_df.headers["q1"]
            current_q2 = twiss_df.headers["q2"]

            # Build response matrix R: rows for beta_x, beta_y, q1, q2; columns for knobs
            derivatives_beta_x = np.array(
                [twiss_df[kopt_list[0][j]].values[bpm_indices] for j in range(nknobs)]
            ).T  # (N, nknobs)
            derivatives_beta_y = np.array(
                [twiss_df[kopt_list[1][j]].values[bpm_indices] for j in range(nknobs)]
            ).T  # (N, nknobs)

            # Derivatives for tunes
            last_element = twiss_df.index[-1]
            d_mu1_end = np.array([twiss_df[kopt_list[0][nknobs + j]].loc[last_element] for j in range(nknobs)])
            d_mu2_end = np.array([twiss_df[kopt_list[1][nknobs + j]].loc[last_element] for j in range(nknobs)])
            d_q1_d_knob = d_mu1_end / (2 * np.pi)
            d_q2_d_knob = d_mu2_end / (2 * np.pi)

            # Stack response matrix
            r = np.vstack([derivatives_beta_x, derivatives_beta_y, d_q1_d_knob.reshape(1, -1), d_q2_d_knob.reshape(1, -1)])

            # Build delta vector
            delta_beta_x = target_betax - betax_current
            delta_beta_y = target_betay - betay_current
            delta_q1 = self.target_q1 - current_q1
            delta_q2 = self.target_q2 - current_q2
            delta = np.concatenate([delta_beta_x, delta_beta_y, [delta_q1], [delta_q2]])

            # Solve for delta_knob
            if svd_cutoff is not None:
                # Use SVD with cutoff for regularization
                u, s, vt = np.linalg.svd(r, full_matrices=False)
                s_max = s.max()
                s_cutoff = s > svd_cutoff * s_max
                s_inv = np.where(s_cutoff, 1 / s, 0)
                delta_knob = vt.T @ (s_inv * (u.T @ delta))
            else:
                # Use pseudoinverse
                delta_knob = np.linalg.pinv(r) @ delta

            # Get current knob values
            current_values = np.array([self.mad_interface.mad[f"MADX['{knob}']"] for knob in knobs_list])

            # Compute new knob values
            new_values = current_values + delta_knob

            # Set new knob values
            for i, knob in enumerate(knobs_list):
                self.mad_interface.mad[f"MADX['{knob}']"] = new_values[i]

        # Final knob values
        final_knobs = {knob: self.mad_interface.mad[f"MADX['{knob}']"] for knob in knobs_list}

        uncertainties = {}  # Linear method doesn't provide uncertainties
        logger.info("Linear beta matching completed")
        return final_knobs, uncertainties

    def _load_model_twiss(self):
        """Load target model twiss from TFS file.

        Returns:
            DataFrame containing model twiss data with BETX, BETY columns.
        """
        import tfs

        logger.info(f"Loading model twiss from {self.config.model_twiss_file}")
        model_twiss = tfs.read(self.config.model_twiss_file)
        logger.debug(f"Loaded twiss data with {len(model_twiss)} elements")
        return model_twiss

    def _init_mad_interface(self) -> None:
        """Initialise the MAD-NG interface for beta computations.

        Sets up the MAD-NG interface with the sequence file and applies
        the estimated quadrupole strengths from the controller.
        """
        from aba_optimiser.mad.base_mad_interface import BaseMadInterface

        logger.info("Initialising MAD-NG interface for beta matching")

        # Create MAD interface
        self.mad_interface = BaseMadInterface()

        # Load sequence
        self.mad_interface.load_sequence(self.config.sequence_file_path, self.config.seq_name)

        # Set up beam
        self.mad_interface.setup_beam(self.config.beam_energy)

        # Observe the bpms
        self.mad_interface.observe_elements()

        # Apply estimated quadrupole strengths
        logger.info(f"Applying {len(self.estimated_strengths)} estimated quadrupole strengths")
        self.mad_interface.set_magnet_strengths(self.estimated_strengths)
        self.mad_interface.set_madx_variables(**self.tune_knobs)

        # Compute twiss to get target tunes after applying estimated strengths
        twiss_result = self.mad_interface.run_twiss()
        self.target_q1 = twiss_result.headers["q1"]
        self.target_q2 = twiss_result.headers["q2"]
        logger.info(
            f"Target tunes after applying estimated strengths: Q1={self.target_q1}, Q2={self.target_q2}"
        )

        logger.debug("MAD-NG interface initialized successfully")
        self.cached_derivatives = None
        self.cached_loss = None


    def _get_bpm_list(self) -> list[str]:
        """Get list of BPM names in the magnet range.

        Returns:
            List of BPM names within the specified magnet range.
        """
        return self.mad_interface.get_bpm_list(self.config.magnet_range)

    def _flatten_all_vectors(self, vectors: list[np.ndarray]) -> list[np.ndarray]:
        """Flatten all input vectors to 1D arrays.

        Args:
            vectors: List of numpy arrays to flatten.

        Returns:
            List of flattened numpy arrays.
        """
        return [vec.flatten() for vec in vectors]

    def _compute_twiss_without_derivatives(self) -> tfs.TfsDataFrame:
        """Compute twiss without derivatives.

        Returns:
            Twiss DataFrame without derivatives.
        """

        self.mad_interface.mad.send("""
--start-mad
local observed in MAD.element.flags
loaded_sequence:select(observed, {pattern="$end"})
tws, _ = twiss {sequence=loaded_sequence, observe=1}
loaded_sequence:deselect(observed, {pattern="$end"})
--end-mad
    """)
        all_cols = ["name", "beta11", "beta22"]
        return self.mad_interface.mad.tws.to_df(columns=all_cols).set_index("name")



    def _compute_twiss_with_derivatives(
        self,
        knobs: list[str] | None = None,
    ) -> tuple[tfs.TfsDataFrame, list[list[str]]]:
        """Compute derivatives of beta functions with respect to knobs.

        Args:
            knobs: List of knob names to differentiate. If None, uses ``self.knobs``

        Returns:
            Tuple of (twiss DataFrame, kopt_list) where:
                - twiss DataFrame: Computed twiss with derivatives, including all optical functions
                - kopt_list: List of lists of knob derivative identifiers for betx and bety. First list is for betx, second for bety.
        """
        if knobs is None:
            knobs = list(self.knobs) + list(self.tune_knobs.keys())

        nknobs = len(knobs)

        # Helper function to generate binary mask string for derivative w.r.t. knob j
        def make_binary_mask(j: int, nknobs: int) -> str:
            return "".join("1" if pos == j else "0" for pos in range(nknobs))

        # Build list of derivative identifiers for DA (Differential Algebra)
        # For each plane (x=1, y=2), create derivative names for beta and phase advance (mu)
        # Each derivative name is like "beta11_0100" where the binary string indicates
        # which knob (by position) is being differentiated (1 = differentiate, 0 = not)
        kopt_list = [
            [f"beta{i}{i}_{make_binary_mask(j, nknobs)}" for j in range(nknobs)]
            + [f"mu{i}_{make_binary_mask(j, nknobs)}" for j in range(nknobs)]
            for i in (1, 2)
        ]
        # but prefer the 1-based version for clarity and to avoid negative multipliers
        self.mad_interface.mad.send("""
--start-mad
local knob_list = py:recv()
local num_k = #knob_list
local k_ord = 2
local x0 = MAD.damap { nv = 6, np = num_k, no = {k_ord, k_ord, k_ord, k_ord, 1, 1}, po=1, pn=knob_list}
for i, knob in ipairs(knob_list) do
!   TPSA       = Number     + TPSA (0)
    MADX[knob] = MADX[knob] + x0[knob]
end

local opt_list = py:recv()
local observed in MAD.element.flags
loaded_sequence:select(observed, {pattern="$end"})
tws, _ = twiss {sequence=loaded_sequence, observe=1, X0=x0, trkopt=opt_list}
loaded_sequence:deselect(observed, {pattern="$end"})

-- Reset the knobs after
for i, knob in ipairs(knob_list) do
    MADX[knob] = MADX[knob]:get0()
end
--end-mad
""")
        self.mad_interface.mad.send(knobs)
        flat_opt_list = [item for sublist in kopt_list for item in sublist]
        self.mad_interface.mad.send(flat_opt_list)
        all_cols = ["name", "beta11", "beta22"] + flat_opt_list
        twiss_result = self.mad_interface.mad.tws.to_df(columns=all_cols).set_index("name")
        return twiss_result, kopt_list

    def _run_beta_match(self, bpm_names: list[str]) -> None:
        """Run the massive beta matching using MAD-NG match command.

        Args:
            bpm_names: List of BPM names to match betas at.
        """
        # Prepare variables (beta knobs + tune knobs)
        variables = []
        # Add knobs
        for knob_name in self.knobs + list(self.tune_knobs.keys()):
            variables.append({"var": f"\"MADX['{knob_name}']\"", "name": f"'{knob_name}'"})

        # Prepare equalities (beta constraints for all BPMs)
        equalities = []
        for bpm_name in bpm_names:
            # Get target betas from model twiss
            if bpm_name not in self.model_twiss.index:
                logger.warning(f"BPM {bpm_name} not found in model twiss, skipping")
                continue

            target_betx = self.model_twiss.loc[bpm_name, "beta11"]
            target_bety = self.model_twiss.loc[bpm_name, "beta22"]

            # Add BETX constraint with 1 % relative tolerance
            equalities.append(
                {
                    "expr": f"\\t -> t['{bpm_name}'].beta11 - {target_betx:.16e}",
                    "name": f"'betx_{bpm_name}'",
                    "tol": 1e-3 * target_betx,
                    "kind": "'beta'",
                }
            )

            # Add BETY constraint
            equalities.append(
                {
                    "expr": f"\\t -> t['{bpm_name}'].beta22 - {target_bety:.16e}",
                    "name": f"'bety_{bpm_name}'",
                    "tol": 1e-3 * target_bety,
                    "kind": "'beta'",
                }
            )

        # Add tune constraints
        equalities.append(
            {
                "expr": f"\\t -> t.q1 - {self.target_q1:.16e}",
                "name": "'q1'",
                "tol": 1e-3,  # 0.1% tolerance on tune
                "kind": "'mu'",
            }
        )
        equalities.append(
            {
                "expr": f"\\t -> t.q2 - {self.target_q2:.16e}",
                "name": "'q2'",
                "tol": 1e-3,  # 0.1% tolerance on tune
                "kind": "'mu'",
            }
        )

        logger.info(
            f"Matching {len(variables)} knobs ({len(self.knobs)} beta + {len(self.tune_knobs)} tune) to {len(equalities)} constraints ({len(equalities) - 2} beta + 2 tune)"
        )

        # Create the twiss and send to MAD-NG
        self.mad_interface.mad.send(r"""
local total_fstp = 0.0
function twiss_and_send(cmd)
    local tbl, flow = twiss {sequence=loaded_sequence, observe=1}
    local fstp = cmd["__var"]["fstp"]
    if fstp > 0 then
        total_fstp = total_fstp + fstp
    end

    if total_fstp > 1e-4 then
        py:send({tbl.s, tbl.beta11, tbl.beta22}, true)
        total_fstp = 0.0
    end
    return tbl, flow
end
    """)
        # Run the match
        self.mad_interface.mad.match(
            command=self.mad_interface.mad.twiss_and_send,
            variables=variables,
            equalities=equalities,
            info=2,
            maxcall=300,
        )
        self.mad_interface.mad.send("py:send('match complete')")
        target_betx = self.model_twiss["beta11"]
        target_bety = self.model_twiss["beta22"]

        res = self._flatten_all_vectors(self.mad_interface.mad.recv())
        # Start a plot that updates with each iteration
        from matplotlib import pyplot as plt

        assert np.allclose(res[0], self.model_twiss["s"]), "s positions do not match"
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        plt.ion()
        plt.show(block=False)
        while isinstance(res, list):
            res = self._flatten_all_vectors(res)
            s_positions, betax_vals, betay_vals = res
            betax_rel_diff = (betax_vals - target_betx) / target_betx * 100
            betay_rel_diff = (betay_vals - target_bety) / target_bety * 100
            ax.clear()
            ax.plot(s_positions, betax_rel_diff, label="BETX relative error [%]")
            ax.plot(s_positions, betay_rel_diff, label="BETY relative error [%]")
            ax.set_ylabel("Relative error [%]")
            ax.set_xlabel("s position [m]")
            ax.legend()
            ax.set_title("Beta Matching Progress")
            plt.pause(0.01)
            res = self.mad_interface.mad.recv()
        plt.ioff()

        assert res == "match complete"
        logger.info("Beta matching completed successfully")

    def _compute_betas_with_knobs(self, knob_values: dict[str, float]) -> dict:
        """Compute beta functions with given knob settings.

        This method is not used in the current MAD-NG based matching approach.
        """
        raise NotImplementedError("Using MAD-NG match command instead of external optimisation")

    def _compute_objective(self, knob_values: dict[str, float]) -> float:
        """Compute the objective function value (difference from model).

        This method is not used in the current MAD-NG based matching approach.
        """
        raise NotImplementedError("Using MAD-NG match command instead of external optimisation")

    def _save_results(
        self,
        final_knobs: dict[str, float],
        uncertainties: dict[str, float],
    ) -> None:
        """Save matching results to output directory.

        Args:
            final_knobs: Optimised knob values.
            uncertainties: Uncertainties on knob values.
        """
        # TODO: Save results
        # - Write JSON file with final knob values
        # - Write JSON file with uncertainties
        # - Generate and save comparison plots
        raise NotImplementedError
