"""
Module 4: Privacy Guardrail & Sanitization (DP-SGD)

Wraps the training optimizer with Opacus PrivacyEngine to enforce
Differential Privacy guarantees during diffusion model training.

Provides:
  - Configurable privacy budget (epsilon, delta)
  - Per-sample gradient clipping
  - Calibrated Gaussian noise injection
  - Cumulative privacy spend tracking
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional


class DPSGDWrapper:
    """
    Wrapper around Opacus PrivacyEngine for DP-SGD training.

    Converts a standard PyTorch training loop into a differentially private
    one by clipping per-sample gradients and adding calibrated noise.
    """

    def __init__(
        self,
        enabled: bool = False,
        target_epsilon: float = 8.0,
        target_delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.1,
    ):
        self.enabled = enabled
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.privacy_engine = None
        self._epsilon_history = []

    def attach(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: DataLoader,
        epochs: int,
    ) -> tuple:
        """
        Attach DP-SGD to the model and optimizer.

        Returns the (possibly wrapped) model, optimizer, and data_loader.
        If DP is disabled, returns them unchanged.
        """
        if not self.enabled:
            return model, optimizer, data_loader

        try:
            from opacus import PrivacyEngine
            from opacus.validators import ModuleValidator

            if not ModuleValidator.is_valid(model):
                model = ModuleValidator.fix(model)
                optimizer = type(optimizer)(model.parameters(), **{
                    k: v for k, v in optimizer.defaults.items()
                })

            self.privacy_engine = PrivacyEngine()

            model, optimizer, data_loader = self.privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=data_loader,
                epochs=epochs,
                target_epsilon=self.target_epsilon,
                target_delta=self.target_delta,
                max_grad_norm=self.max_grad_norm,
            )

            print(f"\n[DP-SGD] Privacy engine attached:")
            print(f"  Target epsilon: {self.target_epsilon}")
            print(f"  Target delta: {self.target_delta}")
            print(f"  Max grad norm: {self.max_grad_norm}")
            print(f"  Noise multiplier: {optimizer.noise_multiplier:.4f}")

            return model, optimizer, data_loader

        except ImportError:
            print("[DP-SGD] WARNING: Opacus not installed. Running without DP.")
            self.enabled = False
            return model, optimizer, data_loader

    def get_epsilon(self) -> Optional[float]:
        """Get current cumulative epsilon spent."""
        if not self.enabled or self.privacy_engine is None:
            return None
        try:
            eps = self.privacy_engine.get_epsilon(self.target_delta)
            self._epsilon_history.append(eps)
            return eps
        except Exception:
            return None

    def get_privacy_report(self) -> dict:
        """Return a summary of privacy spending."""
        if not self.enabled:
            return {"enabled": False, "message": "DP-SGD is disabled"}

        current_eps = self.get_epsilon()
        return {
            "enabled": True,
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
            "current_epsilon": current_eps,
            "max_grad_norm": self.max_grad_norm,
            "epsilon_history": self._epsilon_history,
            "budget_remaining": (
                self.target_epsilon - current_eps if current_eps else None
            ),
        }

    @staticmethod
    def from_config(config: dict) -> "DPSGDWrapper":
        """Create a DPSGDWrapper from a config dictionary."""
        privacy_cfg = config.get("privacy", {})
        return DPSGDWrapper(
            enabled=privacy_cfg.get("enabled", False),
            target_epsilon=privacy_cfg.get("epsilon", 8.0),
            target_delta=privacy_cfg.get("delta", 1e-5),
            max_grad_norm=privacy_cfg.get("max_grad_norm", 1.0),
            noise_multiplier=privacy_cfg.get("noise_multiplier", 1.1),
        )
