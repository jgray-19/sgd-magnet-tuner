"""This contains copied physics from MAD-NG"""

import logging
from math import sqrt

LOGGER = logging.getLogger(__name__)


def get_beam_beta(mass, energy):
    """Calculate the relativistic beta (v/c) for a particle.

    Parameters
    ----------
    mass : float
        Particle rest mass in GeV/c².
    energy : float
        Total particle energy in GeV.

    Returns
    -------
    float
        Relativistic beta (dimensionless, v/c).
    """
    LOGGER.debug("Calculating beam beta for mass=%f, energy=%f", mass, energy)
    beta0_sq = (1 - mass / energy) * (1 + mass / energy)
    return sqrt(beta0_sq)


def dp2pt(dp, mass, energy = None, pc = None):
    """Convert relative momentum deviation dp/p to transverse momentum pt/p.

    Parameters
    ----------
    dp : float
        Relative momentum deviation (dp/p, dimensionless).
    mass : float
        Particle rest mass in GeV/c².
    energy : float
        Total particle energy in GeV.
    pc : float
        Canonical momentum in GeV.

    Returns
    -------
    float
        Transverse momentum relative to total momentum (pt/p, dimensionless).
    """
    if energy is None and pc is not None:
        energy = momentum_to_total_energy(pc, mass)
    elif energy is None:
        raise ValueError("Either energy or pc must be provided")
    if dp == 0:
        LOGGER.debug("dp2pt: dp is zero, returning 0.0")
        return 0.0
    LOGGER.debug("Calculating dp2pt for dp=%f, mass=%f, energy=%f", dp, mass, energy)
    beta0 = get_beam_beta(mass, energy)
    # Cancellation-free identity sqrt(a) - b = (a - b^2) / (sqrt(a) + b): the
    # naive sqrt((1+dp)^2 + (1/beta0^2 - 1)) - 1/beta0 subtracts two ~1
    # quantities to yield a ~dp result, losing several significant digits for
    # small dp/low beta0 (e.g. PSB, beta0~0.52). The rewritten numerator
    # 2*dp + dp^2 has no cancellation and stays accurate to machine precision.
    inv_beta0 = 1 / beta0
    radicand = (1 + dp) ** 2 + (inv_beta0**2 - 1)
    return (2 * dp + dp * dp) / (sqrt(radicand) + inv_beta0)


def momentum_to_total_energy(pc: float, mass: float) -> float:
    """Convert canonical momentum ``pc`` [GeV] to total energy [GeV].

    Uses the relativistic relation ``E = sqrt(pc^2 + m^2)``. This is *not* a
    kinetic-energy converter: the total energy for a kinetic energy ``T`` is
    ``T + m``, which is the convention used everywhere the accelerator's
    ``energy`` attribute is built (``energy = kinetic_energy + mass``). Pass the
    canonical momentum here, never the kinetic energy, or the resulting beta0 —
    and hence every dp/p <-> pt conversion — will silently disagree with the
    rest of the toolchain.
    """
    if pc <= 0:
        raise ValueError(f"Canonical momentum pc must be positive, got {pc}")
    return sqrt(pc**2 + mass**2)


def kinetic_to_total_energy(kinetic_energy: float, mass: float) -> float:
    """Convert kinetic energy ``T`` [GeV] to total energy ``E`` [GeV]."""
    return kinetic_energy + mass


def deltap_wrt_reference_total_energy(
    kinetic_energy: float,
    machine_deltap: float,
    reference_kinetic_energy: float,
    mass: float,
) -> float:
    """Convert machine dp/p to a reference-relative value using total energies."""
    total_energy = kinetic_to_total_energy(kinetic_energy, mass)
    reference_total_energy = kinetic_to_total_energy(reference_kinetic_energy, mass)
    measured_total_energy = total_energy * (1 + machine_deltap)
    return (measured_total_energy - reference_total_energy) / reference_total_energy
