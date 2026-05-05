"""
Module 3 (Part 1): DDPM Forward and Reverse Diffusion Process

Implements the Denoising Diffusion Probabilistic Model for tabular data:
  - Forward process: q(x_t | x_0) adds Gaussian noise
  - Reverse process: p_theta(x_{t-1} | x_t) denoises via the Transformer
  - Linear beta schedule from beta_start to beta_end
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


def linear_beta_schedule(num_timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02):
    return torch.linspace(beta_start, beta_end, num_timesteps)


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008):
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """Gather values from a at indices t and reshape for broadcasting with x_shape."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion process for tabular data generation.

    Manages the noise schedule, forward noising, loss computation, and
    reverse sampling (both standard DDPM and DDIM-style fast sampling).
    """

    def __init__(
        self,
        denoiser: nn.Module,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        loss_type: str = "mse",
    ):
        super().__init__()
        self.denoiser = denoiser
        self.num_timesteps = num_timesteps
        self.loss_type = loss_type

        if beta_schedule == "linear":
            betas = linear_beta_schedule(num_timesteps, beta_start, beta_end)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        # Posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward process: add noise to x_start at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def compute_loss(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        mask_ratio: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute the denoising loss for training.

        Args:
            x_start: (batch, total_dim) clean data
            t: (batch,) sampled timesteps
            labels: (batch,) class labels
            noise: optional pre-sampled noise
            mask_ratio: fraction of features to mask (1.0 = full generation)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)

        # Create mask for dynamic masking
        batch_size = x_start.size(0)
        device = x_start.device
        if mask_ratio >= 1.0:
            mask = torch.ones(batch_size, x_start.size(1), dtype=torch.bool, device=device)
        else:
            mask = self.denoiser.create_dynamic_mask(batch_size, device, mask_ratio)

        predicted_noise = self.denoiser(
            x_noisy=x_noisy,
            t=t,
            labels=labels,
            x_clean=x_start,
            mask=mask,
            mask_ratio=mask_ratio,
        )

        if self.loss_type == "mse":
            # Only compute loss on masked features
            if mask_ratio >= 1.0:
                loss = nn.functional.mse_loss(predicted_noise, noise)
            else:
                mask_float = mask.float()
                diff = (predicted_noise - noise) ** 2
                loss = (diff * mask_float).sum() / mask_float.sum().clamp(min=1.0)
        elif self.loss_type == "l1":
            if mask_ratio >= 1.0:
                loss = nn.functional.l1_loss(predicted_noise, noise)
            else:
                mask_float = mask.float()
                diff = torch.abs(predicted_noise - noise)
                loss = (diff * mask_float).sum() / mask_float.sum().clamp(min=1.0)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_index: int,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Single reverse step: p_theta(x_{t-1} | x_t)."""
        # Full generation mask
        mask = torch.ones(
            x_t.size(0), x_t.size(1), dtype=torch.bool, device=x_t.device
        )

        predicted_noise = self.denoiser(
            x_noisy=x_t, t=t, labels=labels, mask=mask, mask_ratio=1.0
        )

        beta_t = extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alpha = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alpha = extract(self.sqrt_recip_alphas, t, x_t.shape)

        model_mean = sqrt_recip_alpha * (x_t - beta_t * predicted_noise / sqrt_one_minus_alpha)

        if t_index == 0:
            return model_mean
        else:
            posterior_var = extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_var) * noise

    @torch.no_grad()
    def sample(
        self,
        labels: torch.Tensor,
        shape: tuple,
        device: torch.device,
        sampling_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate synthetic tabular samples via iterative reverse diffusion.

        Args:
            labels: (n_samples,) class labels for conditional generation
            shape: (n_samples, total_dim) shape of output
            device: target device
            sampling_steps: if set, use DDIM-style subsampled steps
        Returns:
            (n_samples, total_dim) generated clean samples
        """
        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        if sampling_steps is not None and sampling_steps < self.num_timesteps:
            timesteps = self._get_ddim_timesteps(sampling_steps)
        else:
            timesteps = list(reversed(range(self.num_timesteps)))

        for i, t_val in enumerate(timesteps):
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            x = self.p_sample(x, t, t_val, labels)

        return x

    def _get_ddim_timesteps(self, num_steps: int) -> list:
        """Uniformly subsample timesteps for fast DDIM-style sampling."""
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        return list(reversed(timesteps))
