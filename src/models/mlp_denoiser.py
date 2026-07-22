"""
MLP denoiser baseline (TabDDPM-style) for ablation studies.

Drop-in replacement for TabularTransformerDenoiser with the same forward signature
expected by GaussianDiffusion.
"""

import torch
import torch.nn as nn

from .embeddings import TimestepEmbedding, LabelEmbedding


class MLPDenoiser(nn.Module):
    """
    Simple MLP noise predictor conditioned on timestep and label via concatenation.
    Used as the ablation baseline against the Transformer + Conditioning Attention model.
    """

    def __init__(
        self,
        total_dim: int,
        num_classes: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.total_dim = total_dim
        self.d_model = d_model

        self.timestep_embedding = TimestepEmbedding(d_model)
        self.label_embedding = LabelEmbedding(num_classes, d_model)

        hidden = d_ff
        in_dim = total_dim + 2 * d_model
        layers = []
        dims = [in_dim] + [hidden] * max(n_encoder_layers + n_decoder_layers, 2) + [total_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def create_dynamic_mask(
        self,
        batch_size: int,
        device: torch.device,
        mask_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Match Transformer API; MLP ignores mask structure and predicts all dims."""
        rand = torch.rand(batch_size, self.total_dim, device=device)
        return rand < mask_ratio

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
        x_clean: torch.Tensor = None,
        mask: torch.Tensor = None,
        mask_ratio: float = 0.5,
    ) -> torch.Tensor:
        t_emb = self.timestep_embedding(t)
        l_emb = self.label_embedding(labels)
        h = torch.cat([x_noisy, t_emb, l_emb], dim=-1)
        return self.net(h)
