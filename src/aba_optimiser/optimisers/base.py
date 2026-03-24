"""Abstract optimiser interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import numpy as np


class BaseOptimiser(ABC):
    """Common interface for all optimisers used by the training loop."""

    OPTIMISER_NAME: ClassVar[str]
    _REGISTRY: ClassVar[dict[str, type[BaseOptimiser]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = getattr(cls, "OPTIMISER_NAME", "")
        if name:
            BaseOptimiser._REGISTRY[name] = cls

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseOptimiser:
        """Create an optimiser from its registry name."""
        try:
            optimiser_cls = cls._REGISTRY[name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown optimiser type: {name}. Available: {sorted(cls._REGISTRY)}"
            ) from exc
        return optimiser_cls(**kwargs)

    @abstractmethod
    def step(self, params: np.ndarray, grads: np.ndarray, lr: float) -> np.ndarray:
        """Apply one optimisation update step."""

    @abstractmethod
    def state_to_dict(self) -> dict[str, Any]:
        """Serialise optimiser state to a JSON-compatible dictionary."""

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore optimiser state from a dictionary."""
