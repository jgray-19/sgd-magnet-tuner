from aba_optimiser.accelerators import LHC
from aba_optimiser.mad.lhc_optimising_interface import (
    GradientDescentMadInterface,
    LHCOptimisationMadInterface,
)


# A function that dispatches MAD interface class based on the accelerator instance
def get_mad_interface(accelerator) -> type[GradientDescentMadInterface]:
    """Get the MAD interface class for a given accelerator.
    Args:
        accelerator: Accelerator instance
    Returns:
        MAD interface class corresponding to the accelerator type
    """
    try:
        return MAD_INTERFACE_DISPATCH[accelerator.__class__]
    except KeyError:
        raise ValueError(f"No MAD interface found for accelerator type: {accelerator.__class__.__name__}")

# Internal dispatch mapping (can remain as a dict for simplicity)
MAD_INTERFACE_DISPATCH = {
    LHC: LHCOptimisationMadInterface,
}
