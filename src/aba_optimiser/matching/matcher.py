"""Beta function matcher for minimising difference between model and computed betas.

This module provides the main entry point for matching beta functions computed
from estimated magnet strengths to a target model by adjusting knob strengths.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import tfs

from aba_optimiser.accelerators import LHC, SPS
from aba_optimiser.mad.optimising_mad_interface import GradientDescentMadInterface
from aba_optimiser.optimisers import (
    LevenbergMarquardtConfig,
    LevenbergMarquardtOptimiser,
)
from aba_optimiser.optimisers.lbfgs import LBFGSOptimiser
from aba_optimiser.training.optimisation.scheduler import LRScheduler

if TYPE_CHECKING:
    import pandas as pd

    from aba_optimiser.matching.matcher_config import MatcherConfig

logger = logging.getLogger(__name__)


class BetaMatcher:
    """
    Matches computed beta functions to a target model by adjusting knob strengths.

    This class is designed to be run after a tracking fitter has estimated the main
    quadrupole strengths from measurement. It takes:
    - A target model twiss (the betas we want to achieve)
    - The estimated quadrupole strengths from the fitter
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

    def run_match(
        self,
        optimiser_type: str = "lbfgs",
        max_iterations: int = 100,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Run beta matching with the selected optimiser.

        Supported optimisers:
            - ``"lbfgs"``
            - ``"lm"`` (Levenberg-Marquardt)
        """
        optimiser_type = self._validate_optimiser_type(optimiser_type)

        logger.info("Starting %s beta matching", optimiser_type.upper())

        knobs = self.knobs + list(self.tune_knobs)
        current = self._get_current_knob_values(knobs)
        if optimiser_type == "lbfgs":
            step, state = self._build_lbfgs_step(current)
        else:
            step, state = self._build_lm_step(current, max_iterations)
        best = current.copy()
        best_loss = np.inf

        for iteration in range(max_iterations):
            residual, jacobian = self._match_residual_and_jacobian(current, knobs)
            loss = 0.5 * float(residual @ residual)
            gradient = jacobian.T @ residual

            if optimiser_type == "lbfgs" and loss < best_loss:
                best_loss = loss
                best = current.copy()

            if step(iteration, loss, gradient, jacobian):
                logger.info("Convergence achieved")
                break

        final_values = best if optimiser_type == "lbfgs" else state.best_params
        final_knobs = self._set_final_knob_values(knobs, final_values)
        logger.info(
            "%s beta matching completed after %d iterations",
            optimiser_type.upper(),
            iteration + 1,
        )
        return final_knobs, {}

    @staticmethod
    def _validate_optimiser_type(optimiser_type: str) -> str:
        optimiser_type = optimiser_type.lower()
        if optimiser_type not in {"lbfgs", "lm"}:
            raise ValueError(
                f"Unknown beta-matching optimiser: {optimiser_type}. Choose 'lbfgs' or 'lm'."
            )
        return optimiser_type

    def _get_current_knob_values(self, knobs: list[str]) -> np.ndarray:
        return np.array(
            [self.mad_interface.mad[f"MADX['{knob}']"] for knob in knobs], dtype=float
        )

    def _build_lbfgs_step(self, current: np.ndarray):
        optimiser = LBFGSOptimiser(history_size=20, use_adaptive_lr=True)
        scheduler = LRScheduler(
            warmup_epochs=1, decay_epochs=0, start_lr=1e-4, max_lr=1e-4, min_lr=1e-4
        )
        previous_loss = np.inf
        start_time = time.time()

        def step(iteration: int, loss: float, gradient: np.ndarray, jacobian: np.ndarray) -> bool:
            nonlocal previous_loss
            lr = scheduler(iteration)
            logger.info(
                "Iteration %d: loss=%.5f, lr=%.2e, time=%.1fs",
                iteration + 1,
                loss,
                lr,
                time.time() - start_time,
            )
            converged = loss < 0.1 or abs(previous_loss - loss) < 1e-4
            previous_loss = loss
            if not converged:
                current[:] = optimiser.step(current, gradient, lr)
            return converged

        return step, optimiser

    def _build_lm_step(self, current: np.ndarray, max_iterations: int):
        optimiser = LevenbergMarquardtOptimiser(
            LevenbergMarquardtConfig(max_iterations=max_iterations), initial_params=current
        )

        def step(iteration: int, loss: float, gradient: np.ndarray, jacobian: np.ndarray) -> bool:
            update = optimiser.update(current, loss, gradient, jacobian.T @ jacobian)
            current[:] = update.next_params
            logger.info("Iteration %d: loss=%.5f", iteration + 1, loss)
            return update.converged

        return step, optimiser

    def _set_final_knob_values(self, knobs: list[str], values: np.ndarray) -> dict[str, float]:
        final_knobs = {knob: float(value) for knob, value in zip(knobs, values, strict=True)}
        for knob, value in final_knobs.items():
            self.mad_interface.mad[f"MADX['{knob}']"] = value
        return final_knobs

    def _match_residual_and_jacobian(
        self, x: np.ndarray, knobs: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the common relative-beta residual and its analytic Jacobian."""
        for knob, value in zip(knobs, x, strict=True):
            self.mad_interface.mad[f"MADX['{knob}']"] = value
        bpm_names = [bpm for bpm in self._get_bpm_list() if bpm in self.model_twiss.index]
        target_x = self.model_twiss.loc[bpm_names, "beta11"].to_numpy()
        target_y = self.model_twiss.loc[bpm_names, "beta22"].to_numpy()
        twiss, kopt = self._compute_twiss_with_derivatives(knobs)
        indices = [twiss.index.get_loc(bpm) for bpm in bpm_names]
        nknobs = len(knobs)
        rx = (twiss["beta11"].to_numpy()[indices] - target_x) / target_x
        ry = (twiss["beta22"].to_numpy()[indices] - target_y) / target_y
        jx = np.array([twiss[kopt[0][j]].to_numpy()[indices] / target_x for j in range(nknobs)]).T
        jy = np.array([twiss[kopt[1][j]].to_numpy()[indices] / target_y for j in range(nknobs)]).T
        weight = np.sqrt(10.0)
        rq = weight * np.array(
            [twiss.headers["q1"] - self.target_q1, twiss.headers["q2"] - self.target_q2]
        )
        jq = weight * np.array(
            [
                [twiss[kopt[0][nknobs + j]].iloc[-1] / (2 * np.pi) for j in range(nknobs)],
                [twiss[kopt[1][nknobs + j]].iloc[-1] / (2 * np.pi) for j in range(nknobs)],
            ]
        )
        return np.concatenate((rx, ry, rq)), np.vstack((jx, jy, jq))

    def _load_model_twiss(self):
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
