"""
Common baseline interface for synthetic data generators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class BaselineResult:
    """Standard return type for all baseline / method generators."""

    synthetic: Optional[pd.DataFrame]
    labels: Optional[np.ndarray]
    runtime_seconds: float = 0.0
    available: bool = True
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    training_history: Optional[list] = None


class BaselineAdapter:
    """Base adapter. Subclasses implement generate()."""

    name: str = "base"

    def generate(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        target_col: str,
        n_synthetic: int,
        random_state: int = 42,
        task: str = "classification",
        **kwargs,
    ) -> BaselineResult:
        raise NotImplementedError


class UnavailableBaseline(BaselineAdapter):
    """Marks a baseline as not implemented / not installed without fabricating results."""

    def __init__(self, name: str, reason: str):
        self.name = name
        self.reason = reason

    def generate(self, *args, **kwargs) -> BaselineResult:
        return BaselineResult(
            synthetic=None,
            labels=None,
            available=False,
            reason=self.reason,
            metadata={"method": self.name},
        )
