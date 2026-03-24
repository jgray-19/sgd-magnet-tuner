"""MAD-NG interface for LHC model creation workflows."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from pymadng_utils.mad import CoreMadInterface

from .config import (
    AC_MARKER_OFFSET,
    AC_MARKER_PATTERN,
    DRV_TUNES,
    MODEL_COLUMNS,
    MODEL_HEADER,
    MODEL_STRENGTHS,
    NAT_TUNES,
    TUNE_MATCH_FMIN,
    TUNE_MATCH_RTOL,
    TUNE_MATCH_TOLERANCE,
)
from .tfs_utils import convert_multiple_tfs_files

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

    from aba_optimiser.accelerators import Accelerator


LOGGER = logging.getLogger(__name__)
_LHC_SEQUENCE_RE = re.compile(r"lhcb([12])$", re.IGNORECASE)


class ModelCreatorMadngInterface(CoreMadInterface):
    """MAD-NG interface specialised for repository model-creation tasks."""

    def __init__(self, accelerator: Accelerator, **mad_kwargs):
        self.accelerator = accelerator
        self.beam = self._resolve_beam()
        super().__init__(**mad_kwargs)

        if not self.accelerator.sequence_file.exists():
            raise FileNotFoundError(
                "Saved sequence file not found in "
                f"{self.accelerator.sequence_file}. Run make_madx_sequence first."
            )

        self.load_sequence(self.accelerator.sequence_file, self.accelerator.seq_name)
        self.setup_beam(beam_energy=self.accelerator.beam_energy)

    def __enter__(self) -> ModelCreatorMadngInterface:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()

    def _resolve_beam(self) -> int:
        beam = getattr(self.accelerator, "beam", None)
        if isinstance(beam, int):
            return beam

        match = _LHC_SEQUENCE_RE.fullmatch(self.accelerator.seq_name)
        if match is not None:
            return int(match.group(1))

        raise ValueError(
            "ModelCreatorMadngInterface requires an accelerator with an LHC beam "
            f"identifier; got sequence {self.accelerator.seq_name!r}."
        )

    def initialise_model(self, tunes: list[float] | None = None) -> None:
        """Initialise the loaded MAD-NG sequence and match it to target tunes."""
        if tunes is None:
            tunes = NAT_TUNES

        LOGGER.info(
            "Initialising MAD-NG model for %s with tunes %s",
            self.accelerator.seq_name,
            tunes,
        )
        self.match_model_tunes(tunes)

    def get_current_tunes(self, label: str = "") -> tuple[float, float]:
        """Return the current fractional tunes from the loaded sequence."""
        self.mad.send(f"""
local tbl = twiss {{sequence=loaded_sequence}};
{self.py_name}:send({{tbl.q1, tbl.q2}}, true)
        """)
        q1, q2 = self.mad.recv()

        if not isinstance(q1, float) or not isinstance(q2, float):
            raise TypeError(f"Expected float tunes, got {type(q1)} and {type(q2)}")

        log_msg = f"{label} tunes" if label else "Tunes"
        print(f"{log_msg}: Q1={q1:.6f}, Q2={q2:.6f}")
        return q1, q2

    def match_model_tunes(self, target_tunes: list[float]) -> None:
        """Match the loaded sequence to the requested fractional tunes."""
        q1, q2 = self.get_current_tunes("Initial")
        q1_frac = q1 % 1
        q2_frac = q2 % 1

        if (
            abs(target_tunes[0] - q1_frac) < TUNE_MATCH_TOLERANCE
            and abs(target_tunes[1] - q2_frac) < TUNE_MATCH_TOLERANCE
        ):
            print("Tunes already matched within tolerance, skipping matching.")
            return

        qx_var, qy_var = self.accelerator.get_tune_variables()
        qx_integer, qy_integer = self.accelerator.get_tune_integers()
        target_q1_abs = qx_integer + target_tunes[0]
        target_q2_abs = qy_integer + target_tunes[1]

        self.mad.send(f"""
match {{
  command := twiss {{sequence=loaded_sequence}},
  variables = {{
    rtol={TUNE_MATCH_RTOL},
    {{ var = 'MADX.{qx_var}', name='{qx_var}' }},
    {{ var = 'MADX.{qy_var}', name='{qy_var}' }},
  }},
  equalities = {{
    {{ expr = \\t -> math.abs(t.q1)-{target_q1_abs}, name='q1' }},
    {{ expr = \\t -> math.abs(t.q2)-{target_q2_abs}, name='q2' }},
  }},
  objective = {{ fmin={TUNE_MATCH_FMIN} }},
}};
        """)

        self.get_current_tunes("Final")

    def add_strength_columns(self, table_name: str) -> None:
        """Add multipole strength columns to a MAD-NG twiss table."""
        self.mad.send(f"""
strength_cols = {self.py_name}:recv()
MAD.gphys.melmcol({table_name}, strength_cols)
        """).send(MODEL_STRENGTHS)

    def configure_bpm_observation(self) -> None:
        """Configure the loaded sequence to observe BPM elements only."""
        self.mad.send("""
local observed in MAD.element.flags
loaded_sequence:deselect(observed)
        """)
        self.mad.send(
            f'loaded_sequence:select(observed, {{pattern="{self.accelerator.bpm_pattern}"}})'
        )

    def compute_and_export_twiss_tables(
        self,
        model_dir: Path,
        *,
        tunes: list[float],
        drv_tunes: list[float],
    ) -> None:
        """Compute natural and AC-dipole twiss tables and export them to disk."""
        self.mad.send(
            """
hnams = PY_NAME:recv()
cols = PY_NAME:recv()
str_cols = PY_NAME:recv()

cols = MAD.utility.tblcat(cols, str_cols)

twiss_elements = twiss { sequence=loaded_sequence, coupling=true }
twiss_elements:select(nil, \ -> true)
twiss_elements:deselect{pattern="drift"}
        """.replace("PY_NAME", self.py_name)
        )
        self.mad.send(MODEL_HEADER).send(MODEL_COLUMNS).send(MODEL_STRENGTHS)

        self.add_strength_columns("twiss_elements")
        self.mad.send(
            f"""twiss_elements:write("{model_dir / "twiss_elements.dat"}", cols, hnams)"""
        )

        self.configure_bpm_observation()
        self.mad.send("""
twiss_data = twiss {sequence=loaded_sequence, coupling=true, observe=1}
        """)

        ac_marker = AC_MARKER_PATTERN.format(beam=self.beam)
        self.mad.send(f"""
local hackicker, vackicker in MAD.element
loaded_sequence:install{{
    hackicker "hackicker" {{
        at = {AC_MARKER_OFFSET},
        from = "{ac_marker}",
        nat_q = {tunes[0]:.5e},
        drv_q = {drv_tunes[0]:.5e},
        ac_bet = twiss_elements['{ac_marker}'].beta11,
    }},
    vackicker "vackicker" {{
        at = {AC_MARKER_OFFSET},
        from = "{ac_marker}",
        nat_q = {tunes[1]:.5e},
        drv_q = {drv_tunes[1]:.5e},
        ac_bet = twiss_elements['{ac_marker}'].beta22,
    }}
}}
twiss_ac = twiss {{sequence=loaded_sequence, coupling=true, observe=1}}
        """)

        self.add_strength_columns("twiss_ac")
        self.add_strength_columns("twiss_data")

        self.mad.send(f"""
twiss_ac:write("{model_dir / "twiss_ac.dat"}", cols, hnams)
twiss_data:write("{model_dir / "twiss.dat"}", cols, hnams)
print("Exported twiss tables")
{self.py_name}:send("export_complete")
        """)
        result = self.mad.recv()
        if result != "export_complete":
            raise RuntimeError(f"Failed to export twiss tables: {result}")

        print(f"Successfully exported twiss tables to {model_dir}")

    def update_model(
        self,
        model_dir: Path,
        *,
        tunes: list[float] | None = None,
        drv_tunes: list[float] | None = None,
    ) -> None:
        """Run the complete MAD-NG model update workflow for one accelerator."""
        if tunes is None:
            tunes = NAT_TUNES
        if drv_tunes is None:
            drv_tunes = DRV_TUNES

        print(f"\n{'=' * 60}")
        print(f"Updating model for {self.accelerator.seq_name} with MAD-NG")
        print(f"Natural tunes: {tunes}, Driven tunes: {drv_tunes}")
        print(f"{'=' * 60}\n")

        self.initialise_model(tunes=tunes)
        self.compute_and_export_twiss_tables(model_dir, tunes=tunes, drv_tunes=drv_tunes)

        tfs_files = [
            model_dir / "twiss_ac.dat",
            model_dir / "twiss_elements.dat",
            model_dir / "twiss.dat",
        ]
        convert_multiple_tfs_files(tfs_files)
        print(f"\nModel update complete for {self.accelerator.seq_name}\n")
