"""
Embedding layers for the DGD-TabPA Transformer denoising network.

- ColumnEmbedding: per-column linear projections into a shared d_model space
- TimestepEmbedding: sinusoidal positional encoding for diffusion timesteps
- LabelEmbedding: learnable class/label embeddings for conditional generation
"""

import math
import torch
import torch.nn as nn


class ColumnEmbedding(nn.Module):
    """
    Projects each feature dimension of a tabular row into a shared d_model space.

    For a row with total_dim features, this produces a sequence of total_dim
    token embeddings, each of size d_model. This treats every feature as an
    independent "token" so the Transformer can learn inter-feature attention.
    """

    def __init__(self, total_dim: int, d_model: int):
        super().__init__()
        self.total_dim = total_dim
        self.d_model = d_model
        self.projections = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(total_dim)
        ])
        self.position_emb = nn.Parameter(torch.randn(1, total_dim, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, total_dim) raw feature values
        Returns:
            (batch, total_dim, d_model) sequence of token embeddings
        """
        batch_size = x.size(0)
        tokens = []
        for i in range(self.total_dim):
            col_val = x[:, i : i + 1]  # (batch, 1)
            tokens.append(self.projections[i](col_val))  # (batch, d_model)
        out = torch.stack(tokens, dim=1)  # (batch, total_dim, d_model)
        out = out + self.position_emb[:, : self.total_dim, :]
        return out


class TimestepEmbedding(nn.Module):
    """
    Sinusoidal embedding for diffusion timesteps, following the standard
    approach from DDPM (Ho et al., 2020).

    Maps scalar timestep t to a d_model-dimensional vector.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch,) integer timesteps
        Returns:
            (batch, d_model) timestep embeddings
        """
        half_dim = self.d_model // 2
        emb_scale = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb_scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # (batch, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # (batch, d_model)
        if self.d_model % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return self.mlp(emb)


class LabelEmbedding(nn.Module):
    """
    Learnable embedding for class labels, used for conditional generation.
    Includes a special "unconditional" class index for classifier-free guidance.
    """

    def __init__(self, num_classes: int, d_model: int):
        super().__init__()
        self.num_classes = num_classes
        # +1 for unconditional token
        self.embedding = nn.Embedding(num_classes + 1, d_model)
        self.unconditional_idx = num_classes

    def forward(self, labels: torch.Tensor, unconditional: bool = False) -> torch.Tensor:
        """
        Args:
            labels: (batch,) integer class labels
            unconditional: if True, returns the unconditional embedding
        Returns:
            (batch, d_model) label embeddings
        """
        if unconditional:
            idx = torch.full_like(labels, self.unconditional_idx)
            return self.embedding(idx)
        return self.embedding(labels)
