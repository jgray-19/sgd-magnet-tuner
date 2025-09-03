"""This contains copied physics from MAD-NG"""

import logging
from math import sqrt

LOGGER = logging.getLogger(__name__)


def get_beam_beta(mass, energy):
    LOGGER.debug("Calculating beam beta for mass=%f, energy=%f", mass, energy)
    beta0_sq = (1 - mass / energy) * (1 + mass / energy)
    return sqrt(beta0_sq)


def dp2pt(dp, mass, energy):
    if dp == 0:
        LOGGER.debug("dp2pt: dp is zero, returning 0.0")
        return 0.0
    LOGGER.debug("Calculating dp2pt for dp=%f, mass=%f, energy=%f", dp, mass, energy)
    _beta0 = 1 / get_beam_beta(mass, energy)
    return sqrt((1 + dp) ** 2 + (_beta0**2 - 1)) - _beta0
