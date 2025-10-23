"""
Base MAD-NG interface providing core functionality without automatic initialization.

This module provides a minimal base class for MAD-NG operations that can be
extended for specific use cases without unnecessary automatic setup.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pymadng import MAD

if TYPE_CHECKING:
    from pathlib import Path

    import tfs

logger = logging.getLogger(__name__)


class BaseMadInterface:
    """
    Base class for MAD-NG interfaces providing core functionality.

    This class provides essential MAD-NG operations without automatic
    initialization, allowing subclasses to customize setup as needed.
    """

    def __init__(self, **mad_kwargs):
        """
        Initialize base MAD interface.

        Args:
            **mad_kwargs: Keyword arguments passed to pymadng.MAD()
        """
        self.mad = MAD(**mad_kwargs)
        logger.debug("Initialized base MAD interface")
        self.py_name = self.mad.py_name

    def load_sequence(self, sequence_file: str | Path, seq_name: str) -> None:
        """
        Load a sequence file into MAD-NG.

        Args:
            sequence_file: Path to sequence file
            seq_name: Name of the sequence to load
        """
        logger.info(f"Loading sequence from {sequence_file}")
        self.mad.send(f'MADX:load("{sequence_file}")')
        self.mad.send(f"loaded_sequence = MADX.{seq_name}")
        self.mad["SEQ_NAME"] = seq_name

    def setup_beam(self, beam_energy: float, particle: str = "proton") -> None:
        """
        Set up beam parameters.

        Args:
            beam_energy: Beam energy in GeV
            particle: Particle type (default: proton)
        """
        logger.info(f"Setting beam: particle={particle}, energy={beam_energy}")
        self.mad.send(
            f'loaded_sequence.beam = beam {{ particle = "{particle}", energy = {beam_energy} }}'
        )

    def observe_elements(self, pattern: str = "BPM") -> None:
        """
        Configure element observation for tracking.

        Args:
            pattern: Pattern to match elements for observation
        """
        logger.debug(f"Setting observation pattern: {pattern}")
        self.mad.send(f"""
local observed in MAD.element.flags
loaded_sequence:deselect(observed)
loaded_sequence:select(observed, {{pattern="{pattern}"}})
""")

    def unobserve_elements(self, elements: list[str]) -> None:
        """
        Remove specific elements from observation.

        Args:
            elements: List of element names to unobserve
        """
        logger.debug(f"Unobserving elements: {elements}")
        for elem in elements:
            self.mad.send(f"""
local observed in MAD.element.flags
loaded_sequence:deselect(observed, {{pattern="{elem}"}})
""")

    def cycle_sequence(self, marker_name: str | None = None) -> None:
        """
        Cycle sequence to start from a specific marker.

        Args:
            marker_name: Name of marker to cycle to
        """
        logger.debug(f"Cycling sequence to start from {marker_name}")
        if marker_name is None:
            self.mad.send("loaded_sequence:cycle()")
        else:
            self.mad.send(f"loaded_sequence:cycle('{marker_name}')")

    def install_marker(
        self, element_name: str, marker_name: str = None, offset: float = -1e-10
    ) -> str:
        """
        Install a marker element near an existing element.

        Args:
            element_name: Name of reference element
            marker_name: Name for new marker (auto-generated if None)
            offset: Offset from reference element
        """
        if marker_name is None:
            marker_name = f"{element_name}_marker"

        quoted_marker = self.mad.quote_strings(marker_name)
        logger.debug(f"Installing marker {marker_name} at {element_name}")

        self.mad.send(f"""
loaded_sequence:install{{
MAD.element.marker {quoted_marker} {{ at={offset}, from="{element_name}" }}
}}
""")
        return marker_name

    def run_twiss(self, **twiss_kwargs) -> tfs.TfsDataFrame:
        """
        Run TWISS calculation and return results.

        Args:
            **twiss_kwargs: Additional arguments for twiss calculation

        Returns:
            TFS DataFrame with twiss results
        """
        logger.debug("Running twiss calculation")
        if "observe" not in twiss_kwargs:
            twiss_kwargs["observe"] = 1  # Default to no observation if not set
        tws, _ = self.mad.twiss(sequence="loaded_sequence", **twiss_kwargs)
        df = tws.to_df()
        if "name" in df.columns:
            df.set_index("name", inplace=True)
        return df

    def set_variables(self, **kwargs) -> None:
        """
        Set multiple MAD variables.

        Args:
            **kwargs: Variable names and their values
        """
        self.mad.send_vars(**kwargs)

    def set_madx_variables(self, **kwargs) -> None:
        """
        Set multiple MADX variables.

        Args:
            **kwargs: Variable names and their values
        """
        kwargs = {f"MADX['{key}']": value for key, value in kwargs.items()}
        self.set_variables(**kwargs)

    def get_variables(self, *names: str) -> float:
        """
        Get MAD variable values.

        Args:
            names: Variable names

        Returns:
            Variable values
        """
        return self.mad.recv_vars(*names, shallow_copy=True)

    def set_magnet_strengths(self, strengths: dict) -> None:
        """
        Set magnet strengths using standardized naming conventions.

        Args:
            strengths: Dictionary of magnet strengths with '_k' suffix naming
        """
        logger.debug(f"Setting {len(strengths)} magnet strengths")
        for name, strength in strengths.items():
            if name.endswith("_k"):
                element_name = name[:-2]  # Remove '_k' suffix
                if "MB." in element_name:
                    self.set_variables(**{f"MADX['{element_name}'].k0": strength})
                elif "MQ." in element_name:
                    self.set_variables(**{f"MADX['{element_name}'].k1": strength})
                elif "MS." in element_name:
                    self.set_variables(**{f"MADX['{element_name}'].k2": strength})
                else:
                    logger.warning(f"Unknown magnet type for {element_name}")

    def apply_corrector_strengths(self, corrector_table: tfs.TfsDataFrame) -> None:
        """
        Apply corrector strengths from a table to MAD sequence.

        Args:
            corrector_table: DataFrame with corrector strengths
        """
        logger.debug(f"Applying corrector strengths to {len(corrector_table)} elements")

        # Mapping of element kinds to (attribute, column) pairs
        mappings = {
            "hkicker": [("kick", "hkick")],
            "vkicker": [("kick", "vkick")],
            "tkicker": [("hkick", "hkick"), ("vkick", "vkick")],  # untested
        }

        for _, row in corrector_table.iterrows():
            ename = row["ename"]
            try:
                element = self.mad.loaded_sequence[ename]
                kind = element.kind
            except KeyError:
                logger.warning(f"Element {ename} not found in loaded sequence")
                continue

            if kind in mappings:
                for attr, col in mappings[kind]:
                    if col in row.index:
                        self.mad.send(
                            f"loaded_sequence['{ename}'].{attr} = {self.py_name}:recv()"
                        )
                        self.mad.send(row[col])
                    else:
                        logger.warning(
                            f"Column '{col}' not found in corrector table for element {ename}"
                        )
            else:
                logger.warning(f"Element {ename} has unknown kind '{kind}'")

    def match_tunes(
        self,
        target_qx: float = 0.28,
        target_qy: float = 0.31,
        qx_knob: str = "dqx_b1_op",
        qy_knob: str = "dqy_b1_op",
    ) -> dict[str, float]:
        """
        Match fractional tunes using specified knobs.

        Args:
            target_qx: Target horizontal fractional tune
            target_qy: Target vertical fractional tune
            qx_knob: Horizontal tune knob name
            qy_knob: Vertical tune knob name

        Returns:
            Dictionary of matched knob values
        """
        logger.info(f"Matching tunes to ({target_qx}, {target_qy})")

        self.mad["result"] = self.mad.match(
            command=r"\ -> twiss{sequence=loaded_sequence}",
            variables=[
                {"var": f"'MADX.{qx_knob}'", "name": f"'{qx_knob}'"},
                {"var": f"'MADX.{qy_knob}'", "name": f"'{qy_knob}'"},
            ],
            equalities=[
                {"expr": f"\\t -> math.abs(t.q1)-(62+{target_qx})", "name": "'q1'"},
                {"expr": f"\\t -> math.abs(t.q2)-(60+{target_qy})", "name": "'q2'"},
            ],
            objective={"fmin": 1e-18},
            info=2,
        )

        matched_tunes = {
            qx_knob: self.mad[f"MADX['{qx_knob}']"],
            qy_knob: self.mad[f"MADX['{qy_knob}']"],
        }

        logger.info(f"Matched tune values: {matched_tunes}")
        return matched_tunes

    def apply_tune_values(self, tune_values: dict[str, float]) -> None:
        """
        Apply tune knob values.

        Args:
            tune_values: Dictionary of knob names and values
        """
        for knob, value in tune_values.items():
            self.mad.send(f"MADX['{knob}'] = {value}")

    def run_tracking(
        self,
        x0: float = 0,
        px0: float = 0,
        y0: float = 0,
        py0: float = 0,
        t0: float = 0,
        pt0: float = 0,
        nturns: int = 1,
    ) -> None:
        """
        Run particle tracking.

        Args:
            x0, px0, y0, py0, t0, pt0: Initial coordinates
            nturns: Number of turns to track
        """
        logger.debug(f"Running tracking for {nturns} turns")
        self.mad["trk", "mflw"] = self.mad.track(
            sequence="loaded_sequence",
            X0={"x": x0, "px": px0, "y": y0, "py": py0, "t": t0, "pt": pt0},
            nturn=nturns,
        )

    def get_tracking_data(self) -> tfs.TfsDataFrame:
        """
        Retrieve tracking data.

        Args:
            last_turn_only: If True, return only last turn data

        Returns:
            DataFrame with tracking results
        """
        return self.mad.trk.to_df()

    def pt2dp(self, pt: float) -> float:
        """Convert transverse momentum to delta p/p."""
        self.mad.send(
            f"{self.py_name}:send(MAD.gphys.pt2dp({self.py_name}:recv(), loaded_sequence.beam.beta))"
        )
        self.mad.send(pt)
        return self.mad.recv()

    def dp2pt(self, dp: float) -> float:
        """Convert delta p/p to transverse momentum."""
        self.mad.send(
            f"{self.py_name}:send(MAD.gphys.dp2pt({self.py_name}:recv(), loaded_sequence.beam.beta))"
        )
        self.mad.send(dp)
        return self.mad.recv()

    def __del__(self) -> None:
        """Clean up the MAD-NG session on object destruction."""
        if hasattr(self, "mad"):
            del self.mad
