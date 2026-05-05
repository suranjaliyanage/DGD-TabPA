"""
Module 3 (Part 2): Diffusion-Guided Dataset Distillation Loop

Bi-level optimization that condenses N real samples into M synthetic latent codes
(M << N) using distribution matching (MMD loss).

Outer loop: optimize learnable distilled data points Z_syn
Inner loop: train a lightweight surrogate classifier on decoded synthetic data
Matching: MMD between features extracted from real vs synthetic batches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import Optional
from pathlib import Path


class SurrogateClassifier(nn.Module):
    """Lightweight MLP surrogate used in the inner loop of distillation."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )
        self.feature_extractor = nn.Sequential(*list(self.net.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Compute Gaussian (RBF) kernel matrix between x and y."""
    x_norm = (x ** 2).sum(1).unsqueeze(1)
    y_norm = (y ** 2).sum(1).unsqueeze(0)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    return torch.exp(-dist / (2.0 * sigma ** 2))


def compute_mmd(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Maximum Mean Discrepancy (MMD) between two sets of samples.
    Measures the distance between distributions in a reproducing kernel Hilbert space.
    """
    xx = gaussian_kernel(x, x, sigma)
    yy = gaussian_kernel(y, y, sigma)
    xy = gaussian_kernel(x, y, sigma)
    return xx.mean() + yy.mean() - 2.0 * xy.mean()


def multi_scale_mmd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """MMD with multiple kernel bandwidths for robustness."""
    sigmas = [0.1, 0.5, 1.0, 2.0, 5.0]
    mmd = torch.tensor(0.0, device=x.device)
    for s in sigmas:
        mmd = mmd + compute_mmd(x, y, sigma=s)
    return mmd / len(sigmas)


class DistillationLoop:
    """
    Diffusion-Guided Dataset Distillation via Distribution Matching.

    Given a trained diffusion model and real data, optimizes a small set of
    synthetic data points that match the distribution of the real data in
    a learned feature space.
    """

    def __init__(
        self,
        diffusion_model: nn.Module,
        total_dim: int,
        num_classes: int,
        num_synthetic: int = 500,
        distill_lr: float = 0.01,
        distill_epochs: int = 200,
        inner_steps: int = 10,
        device: torch.device = None,
    ):
        self.diffusion_model = diffusion_model
        self.total_dim = total_dim
        self.num_classes = num_classes
        self.num_synthetic = num_synthetic
        self.distill_lr = distill_lr
        self.distill_epochs = distill_epochs
        self.inner_steps = inner_steps
        self.device = device or torch.device("cpu")

        # Initialize learnable synthetic data and labels
        self.syn_data = nn.Parameter(
            torch.randn(num_synthetic, total_dim, device=self.device) * 0.1
        )

        # Balanced label assignment for synthetic data
        labels_per_class = num_synthetic // num_classes
        remainder = num_synthetic % num_classes
        label_list = []
        for c in range(num_classes):
            count = labels_per_class + (1 if c < remainder else 0)
            label_list.extend([c] * count)
        self.syn_labels = torch.tensor(
            label_list[:num_synthetic], dtype=torch.long, device=self.device
        )

        self.optimizer = torch.optim.Adam([self.syn_data], lr=distill_lr)
        self.surrogate = None

    def _init_surrogate(self) -> SurrogateClassifier:
        """Create a fresh surrogate classifier for the inner loop."""
        model = SurrogateClassifier(
            input_dim=self.total_dim,
            num_classes=self.num_classes,
        ).to(self.device)
        return model

    def _inner_loop(self, surrogate: SurrogateClassifier, syn_data: torch.Tensor, syn_labels: torch.Tensor):
        """Train surrogate classifier on synthetic data for inner_steps."""
        opt = torch.optim.SGD(surrogate.parameters(), lr=0.01, momentum=0.9)
        surrogate.train()

        for _ in range(self.inner_steps):
            perm = torch.randperm(syn_data.size(0), device=self.device)
            batch_size = min(64, syn_data.size(0))
            idx = perm[:batch_size]

            logits = surrogate(syn_data[idx])
            loss = F.cross_entropy(logits, syn_labels[idx])

            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

    def distill(
        self,
        X_real: torch.Tensor,
        y_real: torch.Tensor,
        batch_size: int = 256,
    ) -> tuple:
        """
        Run the bi-level distillation loop.

        Outer loop: optimize syn_data to minimize MMD between feature
        distributions of real and synthetic data.

        Args:
            X_real: (N, total_dim) real training data
            y_real: (N,) real labels
            batch_size: batch size for sampling real data

        Returns:
            (syn_data, syn_labels) optimized synthetic dataset
        """
        X_real = X_real.to(self.device)
        y_real = y_real.to(self.device)
        n_real = X_real.size(0)

        self.diffusion_model.eval()

        loss_history = []

        print(f"\nDistillation: {n_real} real -> {self.num_synthetic} synthetic")
        print(f"  Epochs: {self.distill_epochs}, Inner steps: {self.inner_steps}")

        for epoch in tqdm(range(1, self.distill_epochs + 1), desc="Distilling"):
            # Initialize fresh surrogate each epoch for stability
            surrogate = self._init_surrogate()

            # Inner loop: train surrogate on current synthetic data
            self._inner_loop(surrogate, self.syn_data, self.syn_labels)

            surrogate.eval()

            # Sample a real batch
            idx = torch.randperm(n_real)[:batch_size]
            X_real_batch = X_real[idx]

            # Extract features from both real and synthetic data
            feat_real = surrogate.extract_features(X_real_batch)
            feat_syn = surrogate.extract_features(self.syn_data)

            # Outer loop: minimize MMD between feature distributions
            mmd_loss = multi_scale_mmd(feat_real.detach(), feat_syn)

            # Additional class-conditional matching
            class_loss = torch.tensor(0.0, device=self.device)
            for c in range(self.num_classes):
                real_c = feat_real[y_real[idx] == c]
                syn_c = feat_syn[self.syn_labels == c]
                if len(real_c) > 1 and len(syn_c) > 1:
                    class_loss = class_loss + multi_scale_mmd(real_c.detach(), syn_c)

            total_loss = mmd_loss + 0.5 * class_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            loss_history.append(total_loss.item())

            if epoch % 50 == 0 or epoch == 1:
                print(f"  Epoch {epoch}/{self.distill_epochs} | "
                      f"MMD: {mmd_loss.item():.6f} | "
                      f"Class: {class_loss.item():.6f} | "
                      f"Total: {total_loss.item():.6f}")

        final_syn_data = self.syn_data.detach().clone()
        final_syn_labels = self.syn_labels.clone()

        return final_syn_data, final_syn_labels, loss_history

    def save(self, path: str):
        """Save distilled dataset."""
        torch.save({
            "syn_data": self.syn_data.detach().cpu(),
            "syn_labels": self.syn_labels.cpu(),
        }, path)
        print(f"Distilled dataset saved to {path}")

    @staticmethod
    def load(path: str, device: torch.device = None) -> tuple:
        """Load a saved distilled dataset."""
        device = device or torch.device("cpu")
        ckpt = torch.load(path, map_location=device)
        return ckpt["syn_data"], ckpt["syn_labels"]
