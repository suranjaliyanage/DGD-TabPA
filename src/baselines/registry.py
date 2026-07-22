"""
Baseline method registry.

Implemented:
  - smote (and regression noise fallback)

Registered as unavailable (honest adapters — no fabricated metrics):
  - tvae, ctab_gan_plus, tabddpm, mtabgen
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
import pandas as pd

from .interface import BaselineAdapter, BaselineResult, UnavailableBaseline
from .smote_baseline import generate_smote_synthetic


class SMOTEBaseline(BaselineAdapter):
    name = "smote"

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
        t0 = time.perf_counter()
        syn = generate_smote_synthetic(
            X_train,
            y_train,
            target_col=target_col,
            n_synthetic=n_synthetic,
            random_state=random_state,
            task=task,
        )
        labels = syn[target_col].values
        return BaselineResult(
            synthetic=syn.drop(columns=[target_col]),
            labels=labels,
            runtime_seconds=time.perf_counter() - t0,
            available=True,
            metadata={"method": "smote", "task": task},
        )


UNAVAILABLE_REASONS = {
    "tvae": (
        "TVAE baseline adapter registered but not implemented in this repo. "
        "Install/integrate an SDV or Synthcity TVAE wrapper to enable."
    ),
    "ctab_gan_plus": (
        "CTAB-GAN+ baseline adapter registered but not implemented in this repo. "
        "Integrate the official CTAB-GAN+ package to enable."
    ),
    "tabddpm": (
        "TabDDPM baseline adapter registered but not implemented as an external "
        "baseline. The project's MLP denoiser ablation is TabDDPM-style only, "
        "not a full TabDDPM reimplementation."
    ),
    "mtabgen": (
        "MTabGen baseline adapter registered but not implemented in this repo."
    ),
}


def get_baseline_registry() -> Dict[str, BaselineAdapter]:
    registry: Dict[str, BaselineAdapter] = {
        "smote": SMOTEBaseline(),
    }
    for name, reason in UNAVAILABLE_REASONS.items():
        registry[name] = UnavailableBaseline(name, reason)
    return registry


def list_baseline_status() -> Dict[str, dict]:
    reg = get_baseline_registry()
    out = {}
    for name, adapter in reg.items():
        if isinstance(adapter, UnavailableBaseline):
            out[name] = {"available": False, "reason": adapter.reason}
        else:
            out[name] = {"available": True, "reason": None}
    # dgd_tabpa is the primary pipeline method, not a BaselineAdapter
    out["dgd_tabpa"] = {
        "available": True,
        "reason": None,
        "note": "Primary method via run_experiment.py --method dgd_tabpa",
    }
    return out
