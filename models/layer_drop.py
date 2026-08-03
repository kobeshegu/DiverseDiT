"""
Layer Drop (Stochastic Depth) Module for DiverseDiT.

Implements block-level layer drop to improve representation diversity across transformer blocks.
During training, entire blocks are randomly skipped (their input is passed through unchanged),
forcing each block to learn independently useful features rather than relying on residual chains.

Three drop strategies:
  - uniform:  all layers have the same drop probability
  - linear:   drop probability increases linearly from 0 (first layer) to drop_rate (last layer)
  - cosine:   drop probability follows a cosine schedule (gentle start, steep middle, gentle end)

Two drop types:
  - random:    each sample in the batch independently decides whether to drop each layer
  - batch:     the entire batch either drops or keeps each layer (less variance, faster)

Reference: "Deep Networks with Stochastic Depth" (Huang et al., 2016)
"""

import torch
import torch.nn as nn
import math


class LayerDrop(nn.Module):
    """
    Block-level stochastic depth for transformer blocks.

    During training, generates a boolean mask of shape (batch_size, num_layers) or (num_layers,)
    indicating which layers to DROP (True = skip this layer).

    Args:
        num_layers:     total number of transformer blocks
        drop_type:      'random' (per-sample) or 'batch' (whole-batch)
        drop_rate:      maximum / base drop probability
        drop_strategy:  'uniform', 'linear', or 'cosine'
        min_keep_layers: minimum number of layers that must be kept (safety)
    """

    def __init__(
        self,
        num_layers: int,
        drop_type: str = "random",
        drop_rate: float = 0.1,
        drop_strategy: str = "linear",
        min_keep_layers: int = 1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.drop_type = drop_type
        self.drop_rate = drop_rate
        self.drop_strategy = drop_strategy
        self.min_keep_layers = max(min_keep_layers, 1)

        # Pre-compute per-layer drop probabilities
        drop_probs = self._build_drop_probs()
        self.register_buffer("drop_probs", drop_probs, persistent=False)

    # ------------------------------------------------------------------
    # Strategy builders
    # ------------------------------------------------------------------
    def _build_drop_probs(self) -> torch.Tensor:
        """Return a (num_layers,) tensor of per-layer drop probabilities."""
        n = self.num_layers
        if self.drop_strategy == "uniform":
            return torch.full((n,), self.drop_rate)

        elif self.drop_strategy == "linear":
            # Layer 0 -> 0, Layer n-1 -> drop_rate
            return torch.linspace(0.0, self.drop_rate, n)

        elif self.drop_strategy == "cosine":
            # Cosine schedule: gentle start, steep middle, gentle end
            steps = torch.arange(n, dtype=torch.float32)
            probs = 0.5 * self.drop_rate * (1 - torch.cos(math.pi * steps / max(n - 1, 1)))
            return probs

        else:
            raise ValueError(
                f"Unknown drop_strategy '{self.drop_strategy}'. "
                "Choose from: uniform, linear, cosine"
            )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample_drop_pattern(self, batch_size: int = 1) -> torch.Tensor:
        """
        Sample a layer-drop mask.

        Returns:
            drop_mask: bool tensor
                - shape (batch_size, num_layers) when drop_type='random'
                - shape (num_layers,) when drop_type='batch'
                True means DROP (skip) this layer.
        """
        probs = self.drop_probs  # (num_layers,)

        if self.drop_type == "random":
            # Per-sample independent Bernoulli
            rand = torch.rand(batch_size, self.num_layers, device=probs.device)
            drop_mask = rand < probs.unsqueeze(0)  # (B, L)
            # Ensure min_keep_layers per sample
            drop_mask = self._enforce_min_keep(drop_mask, dim=1)

        elif self.drop_type == "batch":
            # Same decision for whole batch
            rand = torch.rand(self.num_layers, device=probs.device)
            drop_mask = rand < probs  # (L,)
            drop_mask = self._enforce_min_keep(drop_mask.unsqueeze(0), dim=1).squeeze(0)

        else:
            raise ValueError(
                f"Unknown drop_type '{self.drop_type}'. Choose from: random, batch"
            )

        return drop_mask

    def _enforce_min_keep(self, mask: torch.Tensor, dim: int) -> torch.Tensor:
        """If too many layers are dropped, randomly un-drop some."""
        num_kept = (~mask).sum(dim=dim)  # per row
        too_few = num_kept < self.min_keep_layers

        if not too_few.any():
            return mask

        # For rows that drop too many, randomly un-drop until min_keep_layers
        for idx in too_few.nonzero(as_tuple=True)[0]:
            dropped = mask[idx].nonzero(as_tuple=True)[0]
            n_restore = self.min_keep_layers - (~mask[idx]).sum().item()
            perm = torch.randperm(dropped.size(0), device=mask.device)[:n_restore]
            mask[idx, dropped[perm]] = False

        return mask

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_drop_probs(self) -> torch.Tensor:
        """Return the per-layer drop probabilities (for logging / visualization)."""
        return self.drop_probs

    def extra_repr(self) -> str:
        return (
            f"num_layers={self.num_layers}, drop_type={self.drop_type}, "
            f"drop_rate={self.drop_rate}, drop_strategy={self.drop_strategy}, "
            f"min_keep_layers={self.min_keep_layers}"
        )


def create_layer_drop(
    num_layers: int,
    drop_type: str = "random",
    drop_rate: float = 0.1,
    drop_strategy: str = "uniform",
    min_keep_layers: int = 1,
) -> LayerDrop:
    """Factory function used by SiT to create a LayerDrop module."""
    return LayerDrop(
        num_layers=num_layers,
        drop_type=drop_type,
        drop_rate=drop_rate,
        drop_strategy=drop_strategy,
        min_keep_layers=min_keep_layers,
    )
