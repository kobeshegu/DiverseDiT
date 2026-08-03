import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    """Create sinusoidal embeddings for continuous timesteps in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TrajectorySampler:
    """Sample paired high/mid/low timestep schedules for trajectory SSL."""

    def __init__(
        self,
        anchors=(0.85, 0.50, 0.15),
        base_jitter=0.10,
        view_jitter=0.03,
        min_gap=0.12,
    ):
        self.anchors = tuple(float(x) for x in anchors)
        self.base_jitter = float(base_jitter)
        self.view_jitter = float(view_jitter)
        self.min_gap = float(min_gap)

    @property
    def num_steps(self):
        return len(self.anchors)

    def _enforce_order(self, t):
        t = torch.sort(t.clamp(0.0, 1.0), dim=1, descending=True).values
        if t.shape[1] <= 1 or self.min_gap <= 0:
            return t

        ordered = [t[:, :1]]
        for i in range(1, t.shape[1]):
            max_allowed = ordered[-1] - self.min_gap
            ordered.append(torch.minimum(t[:, i : i + 1], max_allowed))
        t = torch.cat(ordered, dim=1).clamp(0.0, 1.0)
        return t

    def sample_pair(self, batch_size, device, dtype=torch.float32):
        anchors = torch.tensor(self.anchors, device=device, dtype=dtype).view(1, -1)
        base = anchors + torch.empty(
            batch_size, self.num_steps, device=device, dtype=dtype
        ).uniform_(-self.base_jitter, self.base_jitter)

        noise_a = torch.randn(batch_size, self.num_steps, device=device, dtype=dtype)
        noise_b = torch.randn(batch_size, self.num_steps, device=device, dtype=dtype)
        t_a = base + noise_a * self.view_jitter
        t_b = base + noise_b * self.view_jitter
        return self._enforce_order(t_a), self._enforce_order(t_b)


def interpolate_trajectory(images, noises, timesteps, loss_fn):
    """Interpolate clean latents and one shared noise tensor along a trajectory."""
    shape = (images.shape[0], timesteps.shape[1]) + (1,) * (images.ndim - 1)
    t = timesteps.view(shape).to(device=images.device, dtype=images.dtype)
    alpha_t, sigma_t, _, _ = loss_fn.interpolant(t)
    if noises.ndim == images.ndim:
        noises = noises.unsqueeze(1)
    return alpha_t * images.unsqueeze(1) + sigma_t * noises


def flatten_trajectory(x):
    """Convert [B, K, C, H, W] to [B*K, C, H, W]."""
    bsz, steps = x.shape[:2]
    return x.reshape(bsz * steps, *x.shape[2:])


def repeat_labels_for_trajectory(labels, num_steps):
    return labels.unsqueeze(1).expand(labels.shape[0], num_steps).reshape(-1)


class TrajectoryEncoder(nn.Module):
    """A small temporal encoder over pooled SiT hidden states."""

    def __init__(
        self,
        in_dim,
        embed_dim=768,
        out_dim=256,
        num_layers=2,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        time_embed_dim=256,
    ):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        self.input_proj = nn.Linear(in_dim, embed_dim)
        self.time_embed_dim = time_embed_dim
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, out_dim),
        )

    def forward(self, features, timesteps, normalize=True):
        """
        Args:
            features: [B, K, T, D] hidden states.
            timesteps: [B, K] timesteps matching the trajectory order.
        Returns:
            [B, out_dim] trajectory embedding/logits.
        """
        if features.ndim != 4:
            raise ValueError(f"Expected features [B, K, T, D], got {tuple(features.shape)}")
        pooled = features.mean(dim=2)
        x = self.input_proj(self.in_norm(pooled))
        t_emb = timestep_embedding(timesteps.reshape(-1), self.time_embed_dim)
        t_emb = self.time_proj(t_emb).reshape(timesteps.shape[0], timesteps.shape[1], -1)
        x = self.temporal_encoder(x + t_emb)
        x = self.out_norm(x.mean(dim=1))
        x = self.head(x)
        if normalize:
            x = F.normalize(x, dim=-1)
        return x


class TrajectoryDINOLoss(nn.Module):
    """DINO-style cross-view self-distillation for trajectory embeddings."""

    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        teacher_temp=0.04,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_a, teacher_b, student_b=None, teacher_a=None, accelerator=None):
        teacher_b = teacher_b.detach()
        loss = self._cross_entropy(student_a, teacher_b)
        teacher_outputs = [teacher_b]
        if student_b is not None and teacher_a is not None:
            teacher_a = teacher_a.detach()
            loss = 0.5 * (loss + self._cross_entropy(student_b, teacher_a))
            teacher_outputs.append(teacher_a)
        self.update_center(torch.cat(teacher_outputs, dim=0), accelerator=accelerator)
        return loss

    def _cross_entropy(self, student, teacher):
        student_logp = F.log_softmax(student / self.student_temp, dim=-1)
        teacher_p = F.softmax((teacher - self.center) / self.teacher_temp, dim=-1)
        return -(teacher_p * student_logp).sum(dim=-1).mean()

    @torch.no_grad()
    def update_center(self, teacher_output, accelerator=None):
        if accelerator is not None:
            teacher_output = accelerator.gather(teacher_output)
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)


def vicreg_loss(x, y, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, eps=1e-4):
    x = x.float()
    y = y.float()
    repr_loss = F.mse_loss(x, y)

    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    std_x = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
    std_y = torch.sqrt(y.var(dim=0, unbiased=False) + eps)
    std_loss = 0.5 * (F.relu(1 - std_x).mean() + F.relu(1 - std_y).mean())

    cov_x = (x.T @ x) / max(x.shape[0] - 1, 1)
    cov_y = (y.T @ y) / max(y.shape[0] - 1, 1)
    cov_loss = _off_diagonal(cov_x).pow_(2).sum().div(x.shape[1])
    cov_loss = cov_loss + _off_diagonal(cov_y).pow_(2).sum().div(y.shape[1])
    return sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss


def symmetric_infonce_loss(x, y, temperature=0.1):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    logits = x @ y.T / temperature
    targets = torch.arange(x.shape[0], device=x.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


def _off_diagonal(x):
    n, m = x.shape
    if n != m:
        raise ValueError("Expected a square covariance matrix.")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
