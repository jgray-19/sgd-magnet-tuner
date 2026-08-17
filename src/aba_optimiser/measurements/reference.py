"""Reference closed orbit for measurement reconstruction.

The reconstruction subtracts a reference orbit from the measurement and restores
it again through the transport operator. Whatever is common to the data and the
reference cancels exactly -- BPM reading offsets, the static error orbit, any
momentum-independent contamination -- which is why the reference has to be
*measured* on the same instrument as the data.

Referencing to the model orbit instead is a bias, not a variance: it does not
average down with turns and it is invisible to every harmonic-fit quality
metric. Measured in ``tmom-recon-study`` (``results/10_offmom_offset.csv``,
``10_offmom_noise.csv``): with a 1 mm per-BPM reading offset the relative px
error is 2.5e-1 against a model reference and 4.7e-4 against a measured one, and
the model-reference error stays flat at 4.5e-2 across four decades of BPM noise.

The reference is therefore a required input everywhere in this package. There is
no default: a caller that genuinely has no fitted reference asks for ``"model"``
explicitly and gets a warning, so the compromise is greppable rather than silent.

A reference is also a *momentum origin*, never just an orbit. The momentum handed
to the reconstruction is the offset from the pt the reference was evaluated at,
not the measurement's absolute pt; passing the absolute value costs a factor 66
in reconstructed px (``results/10_offmom_refmom.csv``) and is invisible at first
order because the whole penalty sits in the second-order dispersion term. That is
why this module resolves the orbit and the momentum together, from a
:class:`~tmom_recon.reference.MomentumReference` that carries both, so the pair
cannot be split by a caller. ``tmom_recon.calculate_pz`` takes the absolute
``measurement_pt`` and does the subtraction against ``reference.pt`` itself.
"""

from __future__ import annotations

import logging
from typing import Literal, TypeAlias

import pandas as pd
from tmom_recon.reference import MomentumReference

LOGGER = logging.getLogger(__name__)

#: A reference carrying its own momentum origin, or ``"model"`` to opt explicitly
#: into the biased model closed orbit at ``pt = 0``.
ClosedOrbitReference: TypeAlias = "MomentumReference | Literal['model']"


def model_closed_orbit_reference(twiss: pd.DataFrame) -> MomentumReference:
    """Build a reference from the model twiss, at pt = 0. Biased; see module docstring."""
    orbit = pd.DataFrame(index=twiss.index)
    for column in ("x", "y", "px", "py"):
        orbit[column] = twiss[column] if column in twiss.columns else 0.0
    orbit.index.name = twiss.index.name
    return MomentumReference(closed_orbit=orbit, pt=0.0)


def resolve_closed_orbit_reference(
    reference: ClosedOrbitReference, twiss: pd.DataFrame
) -> MomentumReference:
    """Return the reference to reconstruct against, warning if it is the model one."""
    if isinstance(reference, MomentumReference):
        return reference
    if reference != "model":
        raise ValueError(
            f"reference_closed_orbit must be a MomentumReference or 'model', got {reference!r}"
        )
    LOGGER.warning(
        "Reconstructing against the MODEL closed orbit. This is a bias of one to two "
        "orders of magnitude that no fit-quality metric will show; supply a measured "
        "or fitted MomentumReference instead."
    )
    return model_closed_orbit_reference(twiss)
