"""Beta function matcher for minimising difference between model and computed betas.

This module provides the main entry point for matching beta functions computed
from estimated magnet strengths to a target model by adjusting knob strengths.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.accelerators import LHC, SPS
from aba_optimiser.mad.optimising_mad_interface import GradientDescentMadInterface

if TYPE_CHECKING:
    import pandas as pd

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
    ):
        self.config = config
        self.config.validate()

        self.model_twiss = self._load_model_twiss()
        self.knobs = self.config.knobs_list
        self.tune_knobs = self.config.tune_knobs.copy()
        self.estimated_strengths = self.config.get_estimated_strengths()
        self.accelerator = self._build_accelerator_from_config()

        logger.info("Initialising BetaMatcher")
        logger.info(f"  Target model twiss: {config.model_twiss_file}")
        logger.info(f"  Beta knobs: {len(self.knobs)} knobs")
        logger.info(f"  Tune knobs: {len(self.tune_knobs)} knobs")
        logger.info(f"  Sequence file: {config.sequence_file_path}")
        logger.info(f"  Magnet range: {config.magnet_range}")

        self._init_mad_interface()

    def run_lbfgs_match(self) -> tuple[dict[str, float], dict[str, float]]:
        """Execute beta matching using LBFGS optimisation.

        Returns:
            Tuple of (final_knob_values, uncertainties).
        """
        logger.info("Starting LBFGS beta matching procedure")

        bpm_names = self._get_bpm_list()
        logger.info(f"Found {len(bpm_names)} BPMs in range {self.config.magnet_range}")
        bpm_names_filtered = [bpm for bpm in bpm_names if bpm in self.model_twiss.index]

        all_knobs = self.knobs + list(self.tune_knobs.keys())
        knobs_list = all_knobs
        initial_values = np.array(
            [self.mad_interface.mad[f"MADX['{knob}']"] for knob in knobs_list]
        )

        target_betax = np.array([self.model_twiss.loc[bpm, "beta11"] for bpm in bpm_names_filtered])
        target_betay = np.array([self.model_twiss.loc[bpm, "beta22"] for bpm in bpm_names_filtered])
        n = len(bpm_names_filtered)

        def objective_and_grad(x):
            for i, knob in enumerate(knobs_list):
                self.mad_interface.mad[f"MADX['{knob}']"] = x[i]

            twiss_df_no_deriv = self._compute_twiss_without_derivatives()
            bpm_indices = [twiss_df_no_deriv.index.get_loc(bpm) for bpm in bpm_names_filtered]

            betax = twiss_df_no_deriv["beta11"].values[bpm_indices]
            betay = twiss_df_no_deriv["beta22"].values[bpm_indices]

            diffx = betax - target_betax
            diffy = betay - target_betay
            rms_x = np.sqrt(np.mean(diffx**2))
            rms_y = np.sqrt(np.mean(diffy**2))
            f = rms_x + rms_y

            current_q1 = twiss_df_no_deriv.headers["q1"]
            current_q2 = twiss_df_no_deriv.headers["q2"]
            tune_penalty = 10 * (
                abs(current_q1 - self.target_q1) + abs(current_q2 - self.target_q2)
            )
            f += tune_penalty

            if self._cached_loss is None or abs(f - self._cached_loss) >= 0.1:
                twiss_df, kopt_list = self._compute_twiss_with_derivatives(knobs_list)
                self._cached_derivatives = (twiss_df, kopt_list)
                self._cached_loss = f
            else:
                assert self._cached_derivatives is not None
                twiss_df, kopt_list = self._cached_derivatives

            bpm_indices_d = [twiss_df.index.get_loc(bpm) for bpm in bpm_names_filtered]
            nknobs = len(knobs_list)

            derivatives_beta_x = np.array(
                [twiss_df[kopt_list[0][j]].values[bpm_indices_d] for j in range(nknobs)]
            ).T
            derivatives_beta_y = np.array(
                [twiss_df[kopt_list[1][j]].values[bpm_indices_d] for j in range(nknobs)]
            ).T

            d_rms_x_all = (
                (1 / (n * rms_x)) * np.sum(diffx[:, None] * derivatives_beta_x, axis=0)
                if rms_x > 0
                else np.zeros(nknobs)
            )
            d_rms_y_all = (
                (1 / (n * rms_y)) * np.sum(diffy[:, None] * derivatives_beta_y, axis=0)
                if rms_y > 0
                else np.zeros(nknobs)
            )
            grad = d_rms_x_all + d_rms_y_all

            last_element = twiss_df.index[-1]
            for j in range(nknobs):
                d_mu1 = twiss_df[kopt_list[0][nknobs + j]].loc[last_element]
                d_mu2 = twiss_df[kopt_list[1][nknobs + j]].loc[last_element]
                grad[j] += 10 * (
                    np.sign(current_q1 - self.target_q1) * d_mu1 / (2 * np.pi)
                    + np.sign(current_q2 - self.target_q2) * d_mu2 / (2 * np.pi)
                )

            return f, grad

        from aba_optimiser.optimisers.lbfgs import LBFGSOptimiser
        from aba_optimiser.training.scheduler import LRScheduler

        optimiser = LBFGSOptimiser(history_size=20, use_adaptive_lr=True)
        scheduler = LRScheduler(warmup_epochs=10, decay_epochs=0, start_lr=1e-13, max_lr=3, min_lr=3)
        x = initial_values.copy()
        loss = np.inf
        start_time = time.time()
        for iteration in range(200):
            f, grad = objective_and_grad(x)
            lr = scheduler(iteration)
            x = optimiser.step(x, grad, lr)
            logger.info(
                f"Iteration {iteration + 1}: loss={f:.5f}, lr={lr:.2e}, time={time.time() - start_time:.1f}s"
            )
            if f < 0.1 or abs(loss - f) < 1e-4:
                logger.info("Convergence achieved")
                break
            loss = f

        final_knobs = {knobs_list[i]: x[i] for i in range(len(knobs_list))}
        for knob, value in final_knobs.items():
            self.mad_interface.mad[f"MADX['{knob}']"] = value

        logger.info("LBFGS beta matching completed")
        return final_knobs, {}

    def _load_model_twiss(self):
        import tfs

        logger.info(f"Loading model twiss from {self.config.model_twiss_file}")
        return tfs.read(self.config.model_twiss_file)

    def _init_mad_interface(self) -> None:
        logger.info("Initialising MAD-NG interface for beta matching")

        self.mad_interface = GradientDescentMadInterface(accelerator=self.accelerator)
        self.mad_interface.observe()
        self._apply_estimated_strengths()
        self.mad_interface.set_madx_variables(**self.tune_knobs)

        twiss_result = self.mad_interface.run_twiss()
        self.target_q1 = twiss_result.headers["q1"]
        self.target_q2 = twiss_result.headers["q2"]
        logger.info(f"Target tunes: Q1={self.target_q1}, Q2={self.target_q2}")

        self._cached_derivatives: tuple[pd.DataFrame, list[list[str]]] | None = None
        self._cached_loss: float | None = None

    def _apply_estimated_strengths(self) -> None:
        """Set estimated magnet strengths directly on sequence elements, bypassing knob validation."""
        commands = [
            f"loaded_sequence['{name}'] = {value}"
            for name, value in self.estimated_strengths.items()
        ]
        if commands:
            self.mad_interface.mad.send("\n".join(commands))
        logger.info(f"Applied {len(commands)} estimated magnet strengths to sequence")

    def _build_accelerator_from_config(self) -> LHC | SPS:
        seq_name = str(getattr(self.config, "seq_name", "") or "").lower()
        seq_file = self.config.sequence_file_path
        kinetic_energy = self.config.kinetic_energy

        knobs = list(self.estimated_strengths)
        if "sps" in seq_name or "sps" in seq_file.stem.lower():
            return SPS(
                sequence_file=seq_file,
                kinetic_energy=kinetic_energy,
                custom_knobs_to_optimise=knobs,
            )

        beam = 2 if "b2" in seq_name else 1
        return LHC(
            beam=beam,
            sequence_file=seq_file,
            kinetic_energy=kinetic_energy,
            # Enable all quad types so every perturbed magnet is discovered,
            # then custom_knobs_to_optimise restricts to only the ones we need.
            optimise_quadrupoles=True,
            optimise_other_quadrupoles=True,
            custom_knobs_to_optimise=knobs,
        )

    def _get_bpm_list(self) -> list[str]:
        _, bpms_in_range = self.mad_interface.get_bpm_list(self.config.magnet_range)
        return bpms_in_range

    def _compute_twiss_without_derivatives(self) -> pd.DataFrame:
        self.mad_interface.mad.send("""
local observed in MAD.element.flags
loaded_sequence:select(observed, {pattern="$end"})
tws, _ = twiss {sequence=loaded_sequence, observe=1}
loaded_sequence:deselect(observed, {pattern="$end"})
""")
        return self.mad_interface.mad.tws.to_df(columns=["name", "beta11", "beta22"]).set_index("name")

    def _compute_twiss_with_derivatives(
        self,
        knobs: list[str] | None = None,
    ) -> tuple[pd.DataFrame, list[list[str]]]:
        if knobs is None:
            knobs = list(self.knobs) + list(self.tune_knobs.keys())

        nknobs = len(knobs)

        def make_binary_mask(j: int) -> str:
            return "".join("1" if pos == j else "0" for pos in range(nknobs))

        kopt_list = [
            [f"beta{i}{i}_{make_binary_mask(j)}" for j in range(nknobs)]
            + [f"mu{i}_{make_binary_mask(j)}" for j in range(nknobs)]
            for i in (1, 2)
        ]
        self.mad_interface.mad.send("""
local knob_list = py:recv()
local num_k = #knob_list
local k_ord = 2
local x0 = MAD.damap { nv = 6, np = num_k, no = {k_ord, k_ord, k_ord, k_ord, 1, 1}, po=1, pn=knob_list}
for i, knob in ipairs(knob_list) do
    MADX[knob] = MADX[knob] + x0[knob]
end
local opt_list = py:recv()
local observed in MAD.element.flags
loaded_sequence:select(observed, {pattern="$end"})
tws, _ = twiss {sequence=loaded_sequence, observe=1, X0=x0, trkopt=opt_list}
loaded_sequence:deselect(observed, {pattern="$end"})
for i, knob in ipairs(knob_list) do
    MADX[knob] = MADX[knob]:get0()
end
""")
        self.mad_interface.mad.send(knobs)
        flat_opt_list = [item for sublist in kopt_list for item in sublist]
        self.mad_interface.mad.send(flat_opt_list)
        all_cols = ["name", "beta11", "beta22"] + flat_opt_list
        twiss_result = self.mad_interface.mad.tws.to_df(columns=all_cols).set_index("name")
        return twiss_result, kopt_list
