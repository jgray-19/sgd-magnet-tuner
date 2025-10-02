"""
Tracking interface for beam dynamics simulations.

This module provides a specialized interface for tracking simulations that builds
on the base MAD interface without unnecessary optimization setup.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .base_mad_interface import BaseMadInterface

if TYPE_CHECKING:
    from pathlib import Path

    import tfs

LOGGER = logging.getLogger(__name__)


class TrackingMadInterface(BaseMadInterface):
    """
    Specialized MAD interface for tracking simulations.

    This class provides tracking-specific functionality while keeping
    the interface lightweight and focused.
    """

    def __init__(self, enable_logging: bool = True, **mad_kwargs):
        """
        Initialize tracking MAD interface.

        Args:
            enable_logging: Whether to enable MAD output logging
            **mad_kwargs: Additional keyword arguments for MAD
        """
        if enable_logging:
            mad_kwargs.setdefault("stdout", "mad_stdout.log")
            mad_kwargs.setdefault("redirect_stderr", True)
            mad_kwargs.setdefault("debug", True)
        else:
            mad_kwargs.setdefault("stdout", "/dev/null")
            mad_kwargs.setdefault("redirect_stderr", True)

        super().__init__(**mad_kwargs)
        LOGGER.debug("Initialized tracking MAD interface")

    def setup_for_tracking(
        self,
        sequence_file: str | Path,
        seq_name: str,
        beam_energy: float,
        element_range: str = None,
        cycle_to_start: bool = True,
    ) -> None:
        """
        Complete setup for tracking simulations.

        Args:
            sequence_file: Path to sequence file
            seq_name: Sequence name
            beam_energy: Beam energy in GeV
            element_range: Element range for cycling (optional)
            cycle_to_start: Whether to cycle to start of range
        """
        LOGGER.info(f"Setting up tracking for {seq_name}")

        # Load sequence and set beam
        self.load_sequence(sequence_file, seq_name)
        self.setup_beam(beam_energy)

        # Configure observation for BPMs
        self.observe_elements("BPM")

        # Cycle to start of range if requested
        if cycle_to_start and element_range:
            start_element = element_range.split("/")[0]
            marker_name = f"{start_element}_marker"
            self.install_marker(start_element, marker_name)
            self.cycle_sequence(marker_name)

    def calculate_initial_coordinates(
        self,
        df_twiss: tfs.TfsDataFrame,
        action: float,
        angle: float,
        start_element: str = None,
    ) -> dict[str, float]:
        """
        Calculate initial coordinates from action-angle variables.

        Args:
            df_twiss: Twiss dataframe with optical functions
            action: Initial action [m·rad]
            angle: Initial angle [rad]
            start_element: Element for coordinate calculation (uses first BPM if None)

        Returns:
            Dictionary with initial coordinates
        """
        if start_element is None:
            # Find first BPM in the dataframe
            bpm_mask = df_twiss.index.str.match(r"^BPM\.\d\d.*")
            if not bpm_mask.any():
                raise ValueError("No BPMs found in twiss data")
            start_element = df_twiss[bpm_mask].index[0]

        # Get optical functions at start element
        beta11 = df_twiss.loc[start_element, "beta11"]
        beta22 = df_twiss.loc[start_element, "beta22"]
        alfa11 = df_twiss.loc[start_element, "alfa11"]
        alfa22 = df_twiss.loc[start_element, "alfa22"]

        # Convert action-angle to physical coordinates
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        x0 = np.sqrt(action * beta11) * cos_angle
        px0 = -np.sqrt(action / beta11) * (sin_angle + alfa11 * cos_angle)
        y0 = np.sqrt(action * beta22) * cos_angle
        py0 = -np.sqrt(action / beta22) * (sin_angle + alfa22 * cos_angle)

        return {"x": x0, "px": px0, "y": y0, "py": py0, "t": 0.0, "pt": 0.0}

    def apply_coordinate_errors(
        self,
        coordinates: dict[str, float],
        xy_std: float = 1e-6,
        pxpy_std: float = 1e-6,
        seed: int = None,
    ) -> dict[str, float]:
        """
        Apply random errors to initial coordinates.

        Args:
            coordinates: Initial coordinates dictionary
            xy_std: Standard deviation for position errors [m]
            pxpy_std: Standard deviation for momentum errors [rad]
            seed: Random seed for reproducibility

        Returns:
            Coordinates with applied errors
        """
        rng = np.random.default_rng(seed)

        perturbed = coordinates.copy()
        perturbed["x"] += rng.normal(0, xy_std)
        perturbed["y"] += rng.normal(0, xy_std)
        perturbed["px"] += rng.normal(0, pxpy_std)
        perturbed["py"] += rng.normal(0, pxpy_std)

        return perturbed

    def run_tracking_from_coordinates(
        self, coordinates: dict[str, float], nturns: int = 1
    ) -> tfs.TfsDataFrame:
        """
        Run tracking from specified initial coordinates.

        Args:
            coordinates: Initial coordinates dictionary
            nturns: Number of turns to track

        Returns:
            Tracking results dataframe
        """
        self.run_tracking(**coordinates, nturns=nturns)
        return self.get_tracking_data()

    def run_tracking_from_action_angle(
        self,
        df_twiss: tfs.TfsDataFrame,
        action: float,
        angle: float,
        nturns: int = 1,
        start_element: str = None,
        apply_errors: bool = False,
        xy_std: float = 1e-6,
        pxpy_std: float = 1e-6,
        seed: int = None,
    ) -> tfs.TfsDataFrame:
        """
        Run tracking from action-angle initial conditions.

        Args:
            df_twiss: Twiss dataframe
            action: Initial action [m·rad]
            angle: Initial angle [rad]
            nturns: Number of turns
            start_element: Starting element (auto-detected if None)
            apply_errors: Whether to apply coordinate errors
            xy_std: Position error standard deviation
            pxpy_std: Momentum error standard deviation
            seed: Random seed

        Returns:
            Tracking results dataframe
        """
        # Calculate initial coordinates
        coords = self.calculate_initial_coordinates(
            df_twiss, action, angle, start_element
        )

        # Apply errors if requested
        if apply_errors:
            coords = self.apply_coordinate_errors(coords, xy_std, pxpy_std, seed)

        # Run tracking
        return self.run_tracking_from_coordinates(coords, nturns)

    def apply_quadrupole_errors(
        self, quad_names: list[str], rel_std_dev: float, seed: int = None
    ) -> None:
        """
        Apply relative strength errors to quadrupoles.

        Args:
            quad_names: List of quadrupole names
            rel_std_dev: Relative standard deviation for errors
            seed: Random seed for reproducibility
        """
        rng = np.random.default_rng(seed)

        for name in quad_names:
            # Get current strength
            current_k1 = self.get_variable(f"MADX['{name}'].k1")

            # Apply relative error
            noise = rng.normal(scale=rel_std_dev)
            new_k1 = current_k1 * (1 + noise)

            # Set new strength
            self.set_variable(f"MADX['{name}'].k1", new_k1)

        LOGGER.debug(f"Applied errors to {len(quad_names)} quadrupoles")

    def get_quadrupole_names(
        self, element_range: str = None, pattern: str = "MQ%."
    ) -> list[str]:
        """
        Get names of main quadrupoles with non-zero strength.

        Args:
            element_range: Range to search (optional)
            pattern: Pattern for quadrupole names

        Returns:
            List of quadrupole names
        """
        if element_range is None:
            iter_command = "loaded_sequence:iter()"
        else:
            iter_command = f'loaded_sequence:iter("{element_range}")'

        self.execute_command(f"""
local quad_names = {{}}
for i, elm, s, ds in {iter_command} do
    if elm.k1 and elm.k1 ~= 0 and elm.name:match("{pattern}") then
        table.insert(quad_names, elm.name)
    end
end
{self.py_name}:send(quad_names, true)
""")
        return self.mad.recv()


def create_tracking_interface(
    sequence_file: str | Path,
    seq_name: str,
    beam_energy: float,
    element_range: str = None,
    enable_logging: bool = True,
) -> TrackingMadInterface:
    """
    Factory function to create a configured tracking interface.

    Args:
        sequence_file: Path to sequence file
        seq_name: Sequence name
        beam_energy: Beam energy in GeV
        element_range: Element range for setup
        enable_logging: Enable MAD logging

    Returns:
        Configured tracking interface
    """
    interface = TrackingMadInterface(enable_logging=enable_logging)
    interface.setup_for_tracking(
        sequence_file=sequence_file,
        seq_name=seq_name,
        beam_energy=beam_energy,
        element_range=element_range,
    )
    return interface
