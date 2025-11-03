from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tfs

from aba_optimiser.config import (
    module_path,
)
from aba_optimiser.io.utils import get_lhc_file_path
from aba_optimiser.mad.optimising_mad_interface import OptimisationMadInterface

LOGGER = logging.getLogger(__name__)


class PhaseSpaceDiagnostics:
    def __init__(
        self,
        bpm,
        x_data,
        px_data,
        y_data,
        py_data,
        info=None,
        tws: None | tfs.TfsDataFrame = None,
        analysis_dir: Path | str = module_path / "analysis",
    ):
        LOGGER.debug(
            f"Initialising phase space diagnostics for BPM {bpm} with {len(x_data)} data points"
        )
        self.bpm = bpm
        self.x0 = np.mean(x_data)
        self.px0 = np.mean(px_data)
        self.y0 = np.mean(y_data)
        self.py0 = np.mean(py_data)

        self.dx = x_data - self.x0
        self.dpx = px_data - self.px0
        self.dy = y_data - self.y0
        self.dpy = py_data - self.py0

        # Get Twiss parameters
        if tws is None:
            self._load_twiss(Path(analysis_dir))
        else:
            self.betax = tws.loc[self.bpm, "beta11"]
            self.alfax = tws.loc[self.bpm, "alfa11"]
            self.betay = tws.loc[self.bpm, "beta22"]
            self.alfay = tws.loc[self.bpm, "alfa22"]

        # Compute emittances
        self._compute_emittances()

        # Precompute gamma
        self.gammax = (1 + self.alfax**2) / self.betax
        self.gammay = (1 + self.alfay**2) / self.betay

        # self.refine_optics_with_initial_guess(info)

    def _load_twiss(self, analysis_dir: Path) -> None:
        try:
            beta_phase_x = tfs.read(analysis_dir / "beta_phase_x.tfs", index="NAME")
            beta_phase_y = tfs.read(analysis_dir / "beta_phase_y.tfs", index="NAME")
            use_beta_phase = (
                self.bpm in beta_phase_x.index and self.bpm in beta_phase_y.index
            )
        except FileNotFoundError:
            use_beta_phase = False
        if use_beta_phase:
            self.betax = beta_phase_x.loc[self.bpm, "BETX"]
            self.alfax = beta_phase_x.loc[self.bpm, "ALFX"]
            self.betay = beta_phase_y.loc[self.bpm, "BETY"]
            self.alfay = beta_phase_y.loc[self.bpm, "ALFY"]
        else:
            mad_iface = OptimisationMadInterface(
                get_lhc_file_path(beam=1),
                bpm_pattern="BPM",
                use_real_strengths=False,
            )
            tws = mad_iface.run_twiss()
            self.betax = tws.loc[self.bpm, "beta11"]
            self.alfax = tws.loc[self.bpm, "alfa11"]
            self.betay = tws.loc[self.bpm, "beta22"]
            self.alfay = tws.loc[self.bpm, "alfa22"]

    def _compute_emittances(self) -> None:
        """Compute the emittances based on the provided data."""
        cov_x = np.cov(self.dx, self.dpx)
        self.emit_x = np.sqrt(cov_x[0, 0] * cov_x[1, 1] - cov_x[0, 1] ** 2)

        cov_y = np.cov(self.dy, self.dpy)
        self.emit_y = np.sqrt(cov_y[0, 0] * cov_y[1, 1] - cov_y[0, 1] ** 2)

    def compute_residuals(self) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Compute the residuals of the phase space ellipse fit.
        Returns
        -------
        residuals_x: np.ndarray
            Residuals for the x plane.
        residuals_y: np.ndarray
            Residuals for the y plane.
        std_x: float
            Standard deviation of the residuals for the x plane.
        std_y: float
            Standard deviation of the residuals for the y plane.
        """
        jx2 = (
            self.gammax * self.dx**2
            + 2 * self.alfax * self.dx * self.dpx
            + self.betax * self.dpx**2
        )
        jy2 = (
            self.gammay * self.dy**2
            + 2 * self.alfay * self.dy * self.dpy
            + self.betay * self.dpy**2
        )

        residuals_x = (jx2 / 2 - self.emit_x) / self.emit_x
        residuals_y = (jy2 / 2 - self.emit_y) / self.emit_y

        std_x = np.std(residuals_x)
        std_y = np.std(residuals_y)

        return residuals_x, residuals_y, std_x, std_y

    def ellipse_points(
        self, num_points: int = 50
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate points on the phase space ellipse for the current optics.
        Returns
        -------
        x: np.ndarray
            x-coordinates of the ellipse points.
        px: np.ndarray
            px-coordinates of the ellipse points.
        y: np.ndarray
            y-coordinates of the ellipse points.
        py: np.ndarray
            py-coordinates of the ellipse points.
        """
        theta = np.linspace(0, 2 * np.pi, num_points)

        x = np.sqrt(2 * self.emit_x * self.betax) * np.cos(theta)
        px = -np.sqrt(2 * self.emit_x / self.betax) * (
            self.alfax * np.cos(theta) + np.sin(theta)
        )

        y = np.sqrt(2 * self.emit_y * self.betay) * np.cos(theta)
        py = -np.sqrt(2 * self.emit_y / self.betay) * (
            self.alfay * np.cos(theta) + np.sin(theta)
        )

        return (
            x + self.x0,
            px + self.px0,
            y + self.y0,
            py + self.py0,
        )

    def ellipse_sigma(
        self, num_points: int = 50, sigma_level: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        _, _, std_x, std_y = self.compute_residuals()

        emit_x_scaled = self.emit_x * (1 + sigma_level * std_x)
        emit_y_scaled = self.emit_y * (1 + sigma_level * std_y)

        theta = np.linspace(0, 2 * np.pi, num_points)

        x_upper = np.sqrt(2 * emit_x_scaled * self.betax) * np.cos(theta)
        px_upper = -np.sqrt(2 * emit_x_scaled / self.betax) * (
            self.alfax * np.cos(theta) + np.sin(theta)
        )

        y_upper = np.sqrt(2 * emit_y_scaled * self.betay) * np.cos(theta)
        py_upper = -np.sqrt(2 * emit_y_scaled / self.betay) * (
            self.alfay * np.cos(theta) + np.sin(theta)
        )
        return (
            x_upper + self.x0,
            px_upper + self.px0,
            y_upper + self.y0,
            py_upper + self.py0,
        )

    def refine_optics_with_initial_guess(self, info=None):
        """
        Refine β, α, ε by non-linear least-squares *starting* from
        the current (design) β, α, ε.  Keeps β > 0 automatically.

        Returns
        -------
        (βx, αx, εx, βy, αy, εy)
        """
        from scipy.optimise import least_squares

        old_betax = self.betax
        old_betay = self.betay
        old_alfax = self.alfax
        old_alfay = self.alfay
        old_emit_x = self.emit_x
        old_emit_y = self.emit_y

        # helper: residuals for one plane --------------------------
        def _residual_plane(params, dx, dpx):
            log_beta, alpha, log_emit = params
            beta = np.exp(log_beta)  # guarantees β>0
            emit = np.exp(log_emit)  # guarantees ε>0
            gamma = (1 + alpha**2) / beta
            ji2 = gamma * dx**2 + 2 * alpha * dx * dpx + beta * dpx**2
            return (ji2 / 2 - emit) / emit  # fractional residual

        # current optics as starting point (log to keep positivity)
        x0 = [np.log(self.betax), self.alfax, np.log(self.emit_x)]
        y0 = [np.log(self.betay), self.alfay, np.log(self.emit_y)]

        # solve
        sol_x = least_squares(
            _residual_plane, x0, args=(self.dx, self.dpx), method="lm"
        )
        sol_y = least_squares(
            _residual_plane, y0, args=(self.dy, self.dpy), method="lm"
        )

        # unpack
        self.betax = np.exp(sol_x.x[0])
        self.alfax = sol_x.x[1]
        self.emit_x = np.exp(sol_x.x[2])
        self.betay = np.exp(sol_y.x[0])
        self.alfay = sol_y.x[1]
        self.emit_y = np.exp(sol_y.x[2])

        if info is not None:
            print(
                f"Refined optics for {self.bpm}: "
                f"βx={self.betax:.3f}, αx={self.alfax:.3f}, εx={self.emit_x:.3e}, "
                f"βy={self.betay:.3f}, αy={self.alfay:.3f}, εy={self.emit_y:.3e}"
            )
            print(
                f"  (old: βx={old_betax:.3f}, αx={old_alfax:.3f}, εx={old_emit_x:.3e}, "
                f"βy={old_betay:.3f}, αy={old_alfay:.3f}, εy={old_emit_y:.3e})"
            )
            print(
                f"  relative changes: "
                f"βx={self.betax / old_betax:.3f}, αx={self.alfax / old_alfax:.3f}, εx={self.emit_x / old_emit_x:.3f}, "
                f"βy={self.betay / old_betay:.3f}, αy={self.alfay / old_alfay:.3f}, εy={self.emit_y / old_emit_y:.3f}"
            )

        # update gammas
        self.gammax = (1 + self.alfax**2) / self.betax
        self.gammay = (1 + self.alfay**2) / self.betay

        return (
            self.betax,
            self.alfax,
            self.emit_x,
            self.betay,
            self.alfay,
            self.emit_y,
        )
