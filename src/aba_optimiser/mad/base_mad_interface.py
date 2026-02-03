"""
Base MAD-NG interface providing core functionality without automatic initialization.

This module provides a minimal base class for MAD-NG operations that can be
extended for specific use cases without unnecessary automatic setup.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pymadng import MAD

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

shushing_script = """
-- LuaJIT-only: shush()/unshush() redirect fd 1&2 to /dev/null (mutes Lua + C warnings)

local ffi = require("ffi")
local C = ffi.C

ffi.cdef[[
  int open(const char *pathname, int flags, ...);
  int close(int fd);
  int dup(int oldfd);
  int dup2(int oldfd, int newfd);
  int fflush(void *stream);  // fflush(NULL) flushes all streams
]]

local O_WRONLY = 1            -- POSIX
local DEVNULL  = "/dev/null"

local depth = 0
local saved_out_fd, saved_err_fd

function shush()
  if depth == 0 then
    C.fflush(nil)                                  -- flush all stdio first
    local devnull = C.open(DEVNULL, O_WRONLY)
    saved_out_fd = C.dup(1)
    saved_err_fd = C.dup(2)
    C.dup2(devnull, 1)                              -- redirect stdout -> /dev/null
    C.dup2(devnull, 2)                              -- redirect stderr -> /dev/null
    C.close(devnull)
  end
  depth = depth + 1
end

function unshush()
  if depth == 0 then return end
  depth = depth - 1
  if depth == 0 then
    C.dup2(saved_out_fd, 1)
    C.dup2(saved_err_fd, 2)
    C.close(saved_out_fd)
    C.close(saved_err_fd)
    saved_out_fd, saved_err_fd = nil, nil
  end
end
"""


class BaseMadInterface:
    """
    Base class for MAD-NG interfaces providing core functionality.

    This class provides essential MAD-NG operations without automatic
    initialization, allowing subclasses to customise setup as needed.
    """

    def __init__(self, **mad_kwargs):
        """
        Initialise base MAD interface.

        Args:
            **mad_kwargs: Keyword arguments passed to pymadng.MAD()
        """
        self.mad = MAD(**mad_kwargs)
        logger.debug("Initialised base MAD interface")
        self.py_name = self.mad.py_name
        self.mad.send(shushing_script)

    def load_sequence(self, sequence_file: str | Path, seq_name: str) -> None:
        """
        Load a sequence file into MAD-NG.

        Args:
            sequence_file: Path to sequence file
            seq_name: Name of the sequence to load
        """
        logger.debug(f"Loading sequence from {sequence_file}")
        file_path = Path(sequence_file).resolve()
        self.mad.send("shush()")
        self.mad.send(f'MADX:load("{file_path}")')
        if self.mad.MADX[seq_name] == 0:
            raise ValueError(f"Sequence '{seq_name}' not found in MAD file '{sequence_file}'")
        self.mad.send(f"loaded_sequence = MADX.{seq_name}")
        self.mad["SEQ_NAME"] = seq_name
        self.mad.send("unshush()")

    def setup_beam(self, beam_energy: float, particle: str = "proton") -> None:
        """
        Set up beam parameters.

        Args:
            beam_energy: Beam energy in GeV
            particle: Particle type (default: proton)
        """
        logger.debug(f"Setting beam: particle={particle}, energy={beam_energy:.15e} GeV")
        self.mad.send(
            f'loaded_sequence.beam = beam {{ particle = "{particle}", energy = {beam_energy:.15e} }}'
        )

    def get_bpm_list(self, bpm_range: str) -> tuple[list[str], list[str]]:
        """
        Get list of BPM names within a specified range.

        Args:
            bpm_range: Range specification (e.g., "BPM.11R2.B1/BPM.11L3.B1")

        Returns:
            Tuple containing:
            - List of all BPM names in the sequence
            - List of BPM names within the specified range
        """
        logger.debug(f"Getting BPM list for range: {bpm_range}")

        # Use MAD script to collect BPMs in range
        get_bpms_mad = f"""
        local all_bpms = {{}}
        local bpm_in_range = {{}}
        for _, elm in loaded_sequence:iter() do
            if elm:is_observed() then
                table.insert(all_bpms, elm.name)
            end
        end
        for _, elm in loaded_sequence:iter("{bpm_range}") do
            if elm:is_observed() then
                table.insert(bpm_in_range, elm.name)
            end
        end
        {self.py_name}:send(all_bpms, true)
        {self.py_name}:send(bpm_in_range, true)
        """
        self.mad.send(get_bpms_mad)
        all_bpms = self.mad.receive()  # Run the script
        bpms_in_range = self.mad.receive()
        bpms_in_range = [
            bpm for bpm in all_bpms if bpm in bpms_in_range
        ]  # Preserve order of all_bpms
        logger.debug(f"Found {len(bpms_in_range)} BPMs in range {bpm_range}")
        return all_bpms, bpms_in_range

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
        logger.debug(f"Unobserving elements: {', '.join(elements)}")
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
        success_script = f"\n{self.py_name}:send(true)\n"
        if marker_name is None:
            self.mad.send("loaded_sequence:cycle()" + success_script)
        else:
            self.mad.send(f"loaded_sequence:cycle('{marker_name}')" + success_script)
        try:
            assert self.mad.recv(), (
                "Sequence cycling failed, you may have left something in the pipe."
            )
        except RuntimeError as e:
            logger.error(f"Error during sequence cycling: {e}")
            raise RuntimeError("Cycle failed - check MAD output for details") from e

    def install_marker(
        self, element_name: str, marker_name: str | None = None, offset: float = -1e-10
    ) -> str:
        """
        Install a marker element near an existing element.

        Args:
            element_name: Name of reference element
            marker_name: Name for new marker (auto-generated if None)
            offset: Offset from reference element

        Returns:
            Name of the installed marker
        """
        if marker_name is None:
            marker_name = f"{element_name}_marker"

        elm_idx = self.mad.send(
            f"{self.py_name}:send(loaded_sequence:index_of('{element_name}'))"
        ).recv()
        if elm_idx is None:
            raise ValueError(f"Element '{element_name}' not found in loaded sequence")
        if elm_idx <= 2:
            # First index is always $start marker, second is first real element
            offset = 1e-10  # Can't go before the first element -> MAD-NG bug

        quoted_marker = self.mad.quote_strings(marker_name)
        logger.debug(f"Installing marker {marker_name} at {element_name}")

        self.mad.send(f"""
loaded_sequence:install{{
MAD.element.marker {quoted_marker} {{ at={offset}, from="{element_name}" }}
}}
""")
        return marker_name

    def install_ac_dipole(
        self,
        marker_name: str,
        nat_tunes: tuple[float, float],
        drv_tunes: tuple[float, float],
        offset: float = 0.0,
    ) -> None:
        """
        Install AC dipole kickers at a specified marker location.

        The AC dipole consists of horizontal and vertical kicker elements that
        drive the beam at specified tunes. The beta functions at the marker
        location are automatically retrieved from the twiss table.

        Args:
            marker_name: Name of marker where AC dipole will be installed
            nat_tunes: Natural tunes (qx, qy)
            drv_tunes: Driven tunes (qx_drv, qy_drv)
            offset: Offset from marker location (default: 0.0)
        """
        logger.debug(
            f"Installing AC dipole at {marker_name} with natural tunes {nat_tunes} "
            f"and driven tunes {drv_tunes}"
        )

        # Get beta functions at AC marker location
        self.mad.send(f"""
local tws = twiss{{sequence=loaded_sequence}}
local betx = tws['{marker_name}'].beta11
local bety = tws['{marker_name}'].beta22
{self.py_name}:send({{betx, bety}}, true)
""")
        betx, bety = self.mad.recv()

        if betx is None or bety is None:
            raise ValueError(f"Could not retrieve beta functions at marker '{marker_name}'")

        # Install AC kickers
        self.mad.send(f"""
local hackicker, vackicker in MAD.element
loaded_sequence:install{{
    hackicker "hackicker" {{
        at = {offset},
        from = "{marker_name}",
        nat_q = {nat_tunes[0]:.15e},
        drv_q = {drv_tunes[0]:.15e},
        ac_bet = {betx:.15e},
    }},
    vackicker "vackicker" {{
        at = {offset},
        from = "{marker_name}",
        nat_q = {nat_tunes[1]:.15e},
        drv_q = {drv_tunes[1]:.15e},
        ac_bet = {bety:.15e},
    }}
}}
""")

        logger.debug(f"AC dipole installed: betx={betx:.6f}, bety={bety:.6f} at {marker_name}")

    def run_twiss(self, **twiss_kwargs) -> pd.DataFrame:
        """
        Run TWISS calculation and return results. If 'observe' is not specified,
        it defaults to 1 (observing observed elements every turn).

        Args:
            **twiss_kwargs: Additional arguments for twiss calculation

        Returns:
            TFS DataFrame with twiss results
        """
        logger.debug("Running twiss calculation")
        if "observe" not in twiss_kwargs:
            twiss_kwargs["observe"] = 1  # Default to no observation if not set

        try:
            self.mad["tws", "flw"] = self.mad.twiss(sequence="loaded_sequence", **twiss_kwargs)
        except ValueError as e:
            logger.error(f"Error during twiss calculation: {e}")
            raise RuntimeError("Twiss failed - check MAD output for details") from e

        df = self.mad.tws.to_df()
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

    def get_variables(self, *names: str) -> tuple[float, ...]:
        """
        Get MAD variable values.

        Args:
            names: Variable names

        Returns:
            Variable values
        """
        return self.mad.recv_vars(*names, shallow_copy=True)

    def set_magnet_strengths(self, strengths: dict[str, float]) -> None:
        """
        Set magnet strengths using standardised naming conventions.

        Args:
            strengths: Dictionary of magnet strengths with '.k[0-2]' suffix naming
        """
        suffixes = {".k0", ".k1", ".k2", ".kick"}
        logger.debug(f"Setting {len(strengths)} magnet strengths")

        variables_to_set = {}
        for name, strength in strengths.items():
            if not any(suffix in name for suffix in suffixes):
                raise ValueError(f"Magnet name '{name}' must end with one of {suffixes}")
            magnet_name, var = name.rsplit(".", 1)
            variables_to_set[f"MADX['{magnet_name}'].{var}"] = strength

        self.set_variables(**variables_to_set)

    def apply_corrector_strengths(self, corrector_table: pd.DataFrame) -> None:
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
                        self.mad.send(f"loaded_sequence['{ename}'].{attr} = {self.py_name}:recv()")
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
                {
                    "expr": f"\\t -> math.abs(t.q1)-(62+{target_qx})",
                    "name": "'q1'",
                    "tol": 1e-6,
                },
                {
                    "expr": f"\\t -> math.abs(t.q2)-(60+{target_qy})",
                    "name": "'q2'",
                    "tol": 1e-6,
                },
            ],
            info=2,
        )

        matched_tunes = {
            qx_knob: self.mad[f"MADX['{qx_knob}']"],
            qy_knob: self.mad[f"MADX['{qy_knob}']"],
        }

        logger.info(f"Matched tune values: {matched_tunes}")
        return matched_tunes

    def run_tracking(
        self,
        x: float = 0,
        px: float = 0,
        y: float = 0,
        py: float = 0,
        t: float = 0,
        pt: float = 0,
        nturns: int = 1,
        **kwargs,
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
            X0={"x": x, "px": px, "y": y, "py": py, "t": t, "pt": pt},
            nturn=nturns,
            **kwargs,
        )

    def get_tracking_data(self) -> pd.DataFrame:
        """
        Retrieve tracking data.

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
