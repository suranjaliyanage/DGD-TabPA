"""
Module 2: Structure-Aware Latent Manifold Encoding

Transformer-based Encoder-Decoder Denoising Network for the DGD-TabPA framework.

Key components:
  - Encoder: processes condition embeddings (unmasked features)
  - Decoder with Conditioning Attention: Q = masked features, K/V = encoder output
  - Dynamic Masking: randomly partitions features into conditioned/masked sets

This replaces the standard MLP denoiser used in TabDDPM, enabling the model to
capture long-range inter-feature interactions through attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .embeddings import ColumnEmbedding, TimestepEmbedding, LabelEmbedding


class ConditioningAttentionLayer(nn.Module):
    """
    Specialized cross-attention layer where:
      - Q (queries) come from the masked/noisy feature embeddings
      - K, V (keys/values) come from the encoder's condition embeddings

    This is the key architectural contribution described in the report,
    replacing additive conditioning used in TabDDPM to reduce learning bias.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        masked_emb: torch.Tensor,
        condition_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            masked_emb: (batch, n_masked, d_model) noisy feature queries
            condition_emb: (batch, n_cond, d_model) clean condition keys/values
        Returns:
            (batch, n_masked, d_model) refined feature embeddings
        """
        residual = masked_emb
        x = self.norm1(masked_emb)
        x, _ = self.cross_attn(query=x, key=condition_emb, value=condition_emb)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.ff(x) + residual
        return x


class DecoderBlock(nn.Module):
    """
    Single decoder block: self-attention on masked features, then
    conditioning cross-attention from encoder output, then feed-forward.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_sa = nn.LayerNorm(d_model)
        self.cond_attn = ConditioningAttentionLayer(d_model, n_heads, dropout)

    def forward(
        self,
        masked_emb: torch.Tensor,
        condition_emb: torch.Tensor,
    ) -> torch.Tensor:
        # Self-attention among masked features
        residual = masked_emb
        x = self.norm_sa(masked_emb)
        x, _ = self.self_attn(query=x, key=x, value=x)
        x = x + residual

        # Cross-attention with condition
        x = self.cond_attn(x, condition_emb)
        return x


class TabularTransformerDenoiser(nn.Module):
    """
    Full Transformer-based denoising network for tabular diffusion.

    Architecture:
      1. Column embeddings project each feature to d_model
      2. Dynamic masking splits features into conditioned & masked sets
      3. Encoder processes conditioned features + timestep + label
      4. Decoder refines masked (noisy) features via conditioning attention
      5. Output head projects back to per-feature predictions
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
        use_conditioning_attention: bool = True,
    ):
        super().__init__()
        self.total_dim = total_dim
        self.d_model = d_model
        # Ablation: when False, use additive context instead of cross-attention
        self.use_conditioning_attention = use_conditioning_attention

        # Embeddings
        self.col_embedding = ColumnEmbedding(total_dim, d_model)
        self.timestep_embedding = TimestepEmbedding(d_model)
        self.label_embedding = LabelEmbedding(num_classes, d_model)

        # Encoder: standard Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # Decoder: custom blocks with conditioning attention
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, dropout)
            for _ in range(n_decoder_layers)
        ])

        # Output projection: from d_model back to scalar per feature
        self.output_head = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(total_dim)
        ])

        self.norm_out = nn.LayerNorm(d_model)
        # Additive fusion path for attention ablation
        self.additive_proj = nn.Linear(d_model, d_model)

    def create_dynamic_mask(
        self,
        batch_size: int,
        device: torch.device,
        mask_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Generate a random binary mask for dynamic feature masking.
        Each row gets an independent random partition.

        Returns:
            mask: (batch, total_dim) bool tensor. True = masked (to predict),
                  False = conditioned (visible).
        """
        rand = torch.rand(batch_size, self.total_dim, device=device)
        mask = rand < mask_ratio
        # Ensure at least one feature is conditioned and one is masked
        all_masked = mask.all(dim=1)
        if all_masked.any():
            idx = torch.randint(0, self.total_dim, (all_masked.sum(),), device=device)
            mask[all_masked, idx] = False
        none_masked = (~mask).all(dim=1)
        if none_masked.any():
            idx = torch.randint(0, self.total_dim, (none_masked.sum(),), device=device)
            mask[none_masked, idx] = True
        return mask

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
        x_clean: torch.Tensor = None,
        mask: torch.Tensor = None,
        mask_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Denoise x_noisy conditioned on timestep t, labels, and unmasked features.

        During training (generation mode, mask_ratio=1.0 or close to it), all
        features are masked and the model learns to denoise from noise.
        With partial masking, the model also learns imputation.

        Args:
            x_noisy: (batch, total_dim) noisy features at timestep t
            t: (batch,) diffusion timestep
            labels: (batch,) class labels
            x_clean: (batch, total_dim) clean features for conditioning.
                     If None, uses x_noisy for conditioned features.
            mask: (batch, total_dim) pre-computed mask. If None, creates one.
            mask_ratio: fraction of features to mask (default 0.5)
        Returns:
            (batch, total_dim) predicted noise epsilon
        """
        batch_size = x_noisy.size(0)
        device = x_noisy.device

        if mask is None:
            mask = self.create_dynamic_mask(batch_size, device, mask_ratio)

        # For full generation (all features masked), use a full-mask path
        if mask.all():
            return self._forward_full_generation(x_noisy, t, labels)

        # Get clean values for conditioned features
        if x_clean is not None:
            x_condition = x_clean.clone()
        else:
            x_condition = x_noisy.clone()

        # Embed all features
        noisy_emb = self.col_embedding(x_noisy)    # (batch, total_dim, d_model)
        cond_emb = self.col_embedding(x_condition)  # (batch, total_dim, d_model)

        # Timestep and label embeddings
        t_emb = self.timestep_embedding(t)       # (batch, d_model)
        l_emb = self.label_embedding(labels)     # (batch, d_model)
        context = (t_emb + l_emb).unsqueeze(1)   # (batch, 1, d_model)

        # Split by mask: gather conditioned and masked embeddings
        # mask: True = masked (noisy, to predict), False = conditioned (clean)
        masked_indices = []
        cond_indices = []
        for b in range(batch_size):
            masked_indices.append(mask[b].nonzero(as_tuple=True)[0])
            cond_indices.append((~mask[b]).nonzero(as_tuple=True)[0])

        # Pad and gather conditioned embeddings for the encoder
        max_cond = max(len(c) for c in cond_indices)
        max_masked = max(len(m) for m in masked_indices)

        cond_tokens = torch.zeros(batch_size, max_cond + 1, self.d_model, device=device)
        masked_tokens = torch.zeros(batch_size, max_masked, self.d_model, device=device)
        cond_pad_mask = torch.ones(batch_size, max_cond + 1, dtype=torch.bool, device=device)
        masked_pad_mask = torch.ones(batch_size, max_masked, dtype=torch.bool, device=device)

        for b in range(batch_size):
            ci = cond_indices[b]
            mi = masked_indices[b]
            n_c = len(ci)
            n_m = len(mi)
            # Context token at position 0
            cond_tokens[b, 0] = context[b, 0]
            cond_pad_mask[b, 0] = False
            if n_c > 0:
                cond_tokens[b, 1:n_c + 1] = cond_emb[b, ci]
                cond_pad_mask[b, 1:n_c + 1] = False
            if n_m > 0:
                masked_tokens[b, :n_m] = noisy_emb[b, mi]
                masked_pad_mask[b, :n_m] = False

        # Encoder: process conditioned features
        enc_out = self.encoder(
            cond_tokens,
            src_key_padding_mask=cond_pad_mask,
        )  # (batch, max_cond+1, d_model)

        # Decoder: refine masked features via conditioning attention (or additive ablation)
        if self.use_conditioning_attention:
            dec_out = masked_tokens
            for block in self.decoder_blocks:
                dec_out = block(dec_out, enc_out)
        else:
            ctx = self.additive_proj(enc_out[:, 0:1, :])
            dec_out = masked_tokens + ctx

        dec_out = self.norm_out(dec_out)

        # Project back to feature space
        output = torch.zeros(batch_size, self.total_dim, device=device)
        for b in range(batch_size):
            mi = masked_indices[b]
            for j, feat_idx in enumerate(mi):
                output[b, feat_idx] = self.output_head[feat_idx](dec_out[b, j]).squeeze(-1)
            # For conditioned features, predict zero noise
            ci = cond_indices[b]
            for feat_idx in ci:
                output[b, feat_idx] = 0.0

        return output

    def _forward_full_generation(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full generation mode: all features are masked.
        The encoder gets only the timestep+label context token.
        """
        batch_size = x_noisy.size(0)

        noisy_emb = self.col_embedding(x_noisy)  # (batch, total_dim, d_model)

        t_emb = self.timestep_embedding(t)
        l_emb = self.label_embedding(labels)
        context = (t_emb + l_emb).unsqueeze(1)  # (batch, 1, d_model)

        # Encoder with only context token
        enc_out = self.encoder(context)  # (batch, 1, d_model)

        if self.use_conditioning_attention:
            dec_out = noisy_emb
            for block in self.decoder_blocks:
                dec_out = block(dec_out, enc_out)
        else:
            # Ablation: additive timestep/label context (no cross-attention)
            ctx = self.additive_proj(enc_out)
            dec_out = noisy_emb + ctx

        dec_out = self.norm_out(dec_out)

        # Project each token back to scalar
        output = torch.zeros(batch_size, self.total_dim, device=x_noisy.device)
        for i in range(self.total_dim):
            output[:, i] = self.output_head[i](dec_out[:, i]).squeeze(-1)

        return output
