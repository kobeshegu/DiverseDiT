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
        sampler="jittered",
        semantic_bins=32,
        semantic_mix=0.5,
        semantic_temperature=0.2,
        semantic_momentum=0.95,
        semantic_warmup_steps=10000,
        semantic_min_t=0.05,
        semantic_max_t=0.95,
    ):
        self.anchors = tuple(float(x) for x in anchors)
        self.base_jitter = float(base_jitter)
        self.view_jitter = float(view_jitter)
        self.min_gap = float(min_gap)
        self.sampler = sampler
        self.semantic_bins = int(semantic_bins)
        self.semantic_mix = float(semantic_mix)
        self.semantic_temperature = float(semantic_temperature)
        self.semantic_momentum = float(semantic_momentum)
        self.semantic_warmup_steps = int(semantic_warmup_steps)
        self.semantic_min_t = float(semantic_min_t)
        self.semantic_max_t = float(semantic_max_t)
        self.semantic_scores = None
        self.semantic_updates = 0

        if self.sampler not in {"jittered", "semantic"}:
            raise ValueError(f"Unsupported trajectory sampler: {self.sampler}")
        if self.semantic_bins <= 1:
            raise ValueError("semantic_bins must be greater than 1")
        if not (0 <= self.semantic_mix <= 1):
            raise ValueError("semantic_mix must be in [0, 1]")

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

    def _sample_jittered_base(self, batch_size, device, dtype):
        anchors = torch.tensor(self.anchors, device=device, dtype=dtype).view(1, -1)
        return anchors + torch.empty(
            batch_size, self.num_steps, device=device, dtype=dtype
        ).uniform_(-self.base_jitter, self.base_jitter)

    def _semantic_active(self, step):
        return (
            self.sampler == "semantic"
            and self.semantic_scores is not None
            and self.semantic_updates > 0
            and (step is None or step >= self.semantic_warmup_steps)
        )

    def _ensure_semantic_state(self, device):
        if self.semantic_scores is None:
            self.semantic_scores = torch.zeros(self.semantic_bins, device=device, dtype=torch.float32)
        else:
            self.semantic_scores = self.semantic_scores.to(device=device, dtype=torch.float32)

    def _stage_ranges(self):
        anchors = sorted(self.anchors, reverse=True)
        boundaries = [(anchors[i] + anchors[i + 1]) / 2 for i in range(len(anchors) - 1)]
        ranges = []
        for i in range(len(anchors)):
            if i == 0:
                low, high = boundaries[0], self.semantic_max_t
            elif i == len(anchors) - 1:
                low, high = self.semantic_min_t, boundaries[-1]
            else:
                low, high = boundaries[i], boundaries[i - 1]
            ranges.append((max(self.semantic_min_t, low), min(self.semantic_max_t, high)))
        return ranges

    def _sample_semantic_base(self, batch_size, device, dtype):
        self._ensure_semantic_state(device)
        centers = torch.linspace(
            self.semantic_min_t,
            self.semantic_max_t,
            self.semantic_bins,
            device=device,
            dtype=torch.float32,
        )
        bin_width = (self.semantic_max_t - self.semantic_min_t) / max(self.semantic_bins - 1, 1)
        base_steps = []

        for low, high in self._stage_ranges():
            mask = (centers >= low) & (centers <= high)
            if not mask.any():
                nearest = torch.argmin((centers - (low + high) * 0.5).abs())
                mask[nearest] = True

            scores = self.semantic_scores[mask]
            semantic_probs = F.softmax(scores / max(self.semantic_temperature, 1e-6), dim=0)
            uniform_probs = torch.ones_like(semantic_probs) / semantic_probs.numel()
            probs = (1 - self.semantic_mix) * uniform_probs + self.semantic_mix * semantic_probs
            local_indices = torch.multinomial(probs, batch_size, replacement=True)
            selected_centers = centers[mask][local_indices].to(dtype=dtype)
            jitter = torch.empty(batch_size, device=device, dtype=dtype).uniform_(
                -0.5 * bin_width,
                0.5 * bin_width,
            )
            base_steps.append((selected_centers + jitter).clamp(self.semantic_min_t, self.semantic_max_t))

        return torch.stack(base_steps, dim=1)

    def sample_pair(self, batch_size, device, dtype=torch.float32, step=None):
        if self._semantic_active(step):
            base = self._sample_semantic_base(batch_size, device, dtype)
        else:
            base = self._sample_jittered_base(batch_size, device, dtype)

        noise_a = torch.randn(batch_size, self.num_steps, device=device, dtype=dtype)
        noise_b = torch.randn(batch_size, self.num_steps, device=device, dtype=dtype)
        t_a = base + noise_a * self.view_jitter
        t_b = base + noise_b * self.view_jitter
        return self._enforce_order(t_a), self._enforce_order(t_b)

    @torch.no_grad()
    def update_semantic_scores(self, teacher_features, timesteps):
        """
        Update semantic-emergence bin scores with EMA teacher feature velocity.

        Args:
            teacher_features: [B, K, T, D] hidden states from the EMA teacher.
            timesteps: [B, K] ordered timesteps used for teacher_features.
        """
        if self.sampler != "semantic" or teacher_features.shape[1] < 2:
            return

        self._ensure_semantic_state(teacher_features.device)
        pooled = teacher_features.float().mean(dim=2)
        pooled = F.layer_norm(pooled, (pooled.shape[-1],))
        pooled = F.normalize(pooled, dim=-1)
        velocity = 1 - (pooled[:, :-1] * pooled[:, 1:]).sum(dim=-1)
        midpoint = 0.5 * (timesteps[:, :-1].float() + timesteps[:, 1:].float())

        scaled = (midpoint - self.semantic_min_t) / max(self.semantic_max_t - self.semantic_min_t, 1e-6)
        bin_idx = torch.clamp((scaled * self.semantic_bins).long(), 0, self.semantic_bins - 1)
        flat_idx = bin_idx.reshape(-1)
        flat_velocity = velocity.reshape(-1).clamp_min(0)

        sums = torch.zeros_like(self.semantic_scores)
        counts = torch.zeros_like(self.semantic_scores)
        sums.scatter_add_(0, flat_idx, flat_velocity)
        counts.scatter_add_(0, flat_idx, torch.ones_like(flat_velocity))
        mask = counts > 0
        if mask.any():
            new_scores = sums[mask] / counts[mask].clamp_min(1)
            self.semantic_scores[mask] = (
                self.semantic_momentum * self.semantic_scores[mask]
                + (1 - self.semantic_momentum) * new_scores
            )
            self.semantic_updates += 1

    def state_dict(self):
        return {
            "semantic_scores": None if self.semantic_scores is None else self.semantic_scores.detach().cpu(),
            "semantic_updates": self.semantic_updates,
        }

    def load_state_dict(self, state):
        if not state:
            return
        self.semantic_scores = state.get("semantic_scores", None)
        self.semantic_updates = int(state.get("semantic_updates", 0))


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


class TrajectoryPatchEncoder(nn.Module):
    """Project per-timestep patch tokens and predict an EMA teacher target."""

    def __init__(self, in_dim, embed_dim=768, out_dim=256, time_embed_dim=256):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.projector = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, out_dim),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, out_dim),
        )
        self.predictor = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, features, timesteps, predict=True):
        """
        Args:
            features: [B, K, T, D] hidden states.
            timesteps: [B, K] matching timesteps.
            predict: apply the online predictor after projection.
        Returns:
            Patch embeddings with shape [B, K, T, out_dim].
        """
        if features.ndim != 4:
            raise ValueError(f"Expected features [B, K, T, D], got {tuple(features.shape)}")
        if features.shape[:2] != timesteps.shape:
            raise ValueError(
                f"Feature/timestep shape mismatch: {tuple(features.shape)} vs {tuple(timesteps.shape)}"
            )

        projected = self.projector(features)
        time_features = timestep_embedding(
            timesteps.reshape(-1),
            self.time_embed_dim,
        )
        time_features = self.time_proj(time_features).reshape(
            timesteps.shape[0],
            timesteps.shape[1],
            1,
            -1,
        )
        projected = projected + time_features
        return self.predictor(projected) if predict else projected


def trajectory_patch_loss(
    student,
    teacher,
    sim_coeff=1.0,
    std_coeff=1.0,
    cov_coeff=0.04,
    eps=1e-4,
):
    """Patch alignment plus variance/covariance regularization for the online branch."""
    if student.ndim != 4 or teacher.ndim != 4:
        raise ValueError("Expected student and teacher features shaped [B, K, T, D].")
    if teacher.shape[1] != 1:
        raise ValueError(f"Expected one low-noise teacher timestep, got {teacher.shape[1]}")
    if student.shape[0] != teacher.shape[0] or student.shape[2:] != teacher.shape[2:]:
        raise ValueError(
            f"Student/teacher patch shape mismatch: {tuple(student.shape)} vs {tuple(teacher.shape)}"
        )

    teacher = teacher.detach().expand(-1, student.shape[1], -1, -1)
    student_float = student.float()
    teacher_float = teacher.float()
    student_norm = F.normalize(student_float, dim=-1)
    teacher_norm = F.normalize(teacher_float, dim=-1)
    sim_loss = (2 - 2 * (student_norm * teacher_norm).sum(dim=-1)).mean()

    std_loss, cov_loss = _trajectory_variance_covariance(student_float, eps=eps)

    total = sim_coeff * sim_loss + std_coeff * std_loss + cov_coeff * cov_loss
    return total, {
        "sim": sim_loss,
        "std": std_loss,
        "cov": cov_loss,
    }


def trajectory_patch_infonce_loss(
    student,
    teacher,
    accelerator=None,
    num_patches=16,
    temperature=0.1,
    negative_sim_threshold=0.95,
    min_negatives=32,
    nce_coeff=1.0,
    std_coeff=0.1,
    cov_coeff=0.005,
    eps=1e-4,
):
    """Cross-noise patch InfoNCE with other-image EMA teacher negatives."""
    if student.ndim != 4 or teacher.ndim != 4:
        raise ValueError("Expected student and teacher features shaped [B, K, T, D].")
    if teacher.shape[1] != 1:
        raise ValueError(f"Expected one low-noise teacher timestep, got {teacher.shape[1]}")
    if student.shape[0] != teacher.shape[0] or student.shape[2:] != teacher.shape[2:]:
        raise ValueError(
            f"Student/teacher patch shape mismatch: {tuple(student.shape)} vs {tuple(teacher.shape)}"
        )
    if num_patches <= 0:
        raise ValueError("num_patches must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if min_negatives < 0:
        raise ValueError("min_negatives must be non-negative")

    batch_size, student_steps, total_patches, feature_dim = student.shape
    sampled_count = min(int(num_patches), total_patches)
    patch_indices = torch.randperm(total_patches, device=student.device)[:sampled_count]

    student_float = student.float()
    local_teacher = teacher[:, 0, patch_indices].detach().float()
    queries = student_float[:, :, patch_indices]

    if accelerator is not None:
        global_teacher = accelerator.gather(local_teacher)
        process_index = accelerator.process_index
    else:
        global_teacher = local_teacher
        process_index = 0

    queries = F.normalize(queries, dim=-1)
    local_teacher = F.normalize(local_teacher, dim=-1)
    global_teacher = F.normalize(global_teacher, dim=-1)
    keys = global_teacher.reshape(-1, feature_dim)

    query_image_ids = (
        process_index * batch_size
        + torch.arange(batch_size, device=student.device)
    ).view(batch_size, 1, 1).expand(-1, student_steps, sampled_count).reshape(-1)
    key_image_ids = torch.arange(
        global_teacher.shape[0],
        device=student.device,
    ).repeat_interleave(sampled_count)

    local_targets = (
        process_index * batch_size * sampled_count
        + torch.arange(batch_size, device=student.device).unsqueeze(1) * sampled_count
        + torch.arange(sampled_count, device=student.device).unsqueeze(0)
    )
    targets = local_targets.unsqueeze(1).expand(
        -1,
        student_steps,
        -1,
    ).reshape(-1)

    queries = queries.reshape(-1, feature_dim)
    positive_teacher = local_teacher.unsqueeze(1).expand(
        -1,
        student_steps,
        -1,
        -1,
    ).reshape(-1, feature_dim)
    cosine_logits = queries @ keys.T

    # Other patches from the same image are neither positives nor negatives.
    base_valid = key_image_ids.unsqueeze(0) != query_image_ids.unsqueeze(1)
    valid = base_valid
    if negative_sim_threshold < 1.0:
        teacher_similarity = positive_teacher @ keys.T
        filtered_valid = base_valid & (teacher_similarity <= negative_sim_threshold)
        enough_negatives = filtered_valid.sum(dim=1) >= min_negatives
        valid = torch.where(enough_negatives.unsqueeze(1), filtered_valid, base_valid)

    negative_values = cosine_logits.detach()[valid]
    negative_cosine = (
        negative_values.mean()
        if negative_values.numel() > 0
        else cosine_logits.new_zeros(())
    )
    valid.scatter_(1, targets.unsqueeze(1), True)
    logits = cosine_logits / temperature
    logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    raw_nce = F.cross_entropy(logits, targets)

    negative_counts = valid.sum(dim=1).sub(1)
    normalizer = torch.log(negative_counts.float().mean().clamp_min(1) + 1)
    normalized_nce = raw_nce / normalizer.clamp_min(eps)
    std_loss, cov_loss = _trajectory_variance_covariance(student_float, eps=eps)
    sim_loss = (2 - 2 * (queries * positive_teacher).sum(dim=-1)).mean()

    total = (
        nce_coeff * normalized_nce
        + std_coeff * std_loss
        + cov_coeff * cov_loss
    )
    return total, {
        "nce": normalized_nce,
        "nce_raw": raw_nce,
        "sim": sim_loss,
        "std": std_loss,
        "cov": cov_loss,
        "valid_negatives": negative_counts.float().mean(),
        "positive_cosine": 1 - 0.5 * sim_loss.detach(),
        "negative_cosine": negative_cosine,
    }


def trajectory_global_infonce_loss(
    student,
    teacher,
    accelerator=None,
    temperature=0.2,
    negative_sim_threshold=0.95,
    min_negatives=32,
    eps=1e-4,
):
    """Image-level cross-noise InfoNCE over spatially pooled patch features."""
    if student.ndim != 4 or teacher.ndim != 4:
        raise ValueError("Expected student and teacher features shaped [B, K, T, D].")
    if student.shape[1] != 1 or teacher.shape[1] != 1:
        raise ValueError("Global InfoNCE expects one student and one teacher timestep.")
    if student.shape[0] != teacher.shape[0] or student.shape[2:] != teacher.shape[2:]:
        raise ValueError(
            f"Student/teacher patch shape mismatch: {tuple(student.shape)} vs {tuple(teacher.shape)}"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if min_negatives < 0:
        raise ValueError("min_negatives must be non-negative")

    batch_size = student.shape[0]
    queries = F.normalize(student[:, 0].float().mean(dim=1), dim=-1)
    local_teacher = F.normalize(
        teacher[:, 0].detach().float().mean(dim=1),
        dim=-1,
    )

    if accelerator is not None:
        global_teacher = accelerator.gather(local_teacher)
        process_index = accelerator.process_index
    else:
        global_teacher = local_teacher
        process_index = 0
    global_teacher = F.normalize(global_teacher, dim=-1)

    query_image_ids = (
        process_index * batch_size
        + torch.arange(batch_size, device=student.device)
    )
    key_image_ids = torch.arange(global_teacher.shape[0], device=student.device)
    targets = query_image_ids
    cosine_logits = queries @ global_teacher.T

    base_valid = key_image_ids.unsqueeze(0) != query_image_ids.unsqueeze(1)
    valid = base_valid
    if negative_sim_threshold < 1.0:
        teacher_similarity = local_teacher @ global_teacher.T
        filtered_valid = base_valid & (teacher_similarity <= negative_sim_threshold)
        enough_negatives = filtered_valid.sum(dim=1) >= min_negatives
        valid = torch.where(enough_negatives.unsqueeze(1), filtered_valid, base_valid)

    negative_values = cosine_logits.detach()[valid]
    negative_cosine = (
        negative_values.mean()
        if negative_values.numel() > 0
        else cosine_logits.new_zeros(())
    )
    valid.scatter_(1, targets.unsqueeze(1), True)
    logits = (cosine_logits / temperature).masked_fill(
        ~valid,
        torch.finfo(cosine_logits.dtype).min,
    )
    raw_nce = F.cross_entropy(logits, targets)
    negative_counts = valid.sum(dim=1).sub(1)
    normalizer = torch.log(negative_counts.float().mean().clamp_min(1) + 1)
    normalized_nce = raw_nce / normalizer.clamp_min(eps)
    positive_cosine = cosine_logits.gather(1, targets.unsqueeze(1)).mean()

    return normalized_nce, {
        "nce": normalized_nce,
        "nce_raw": raw_nce,
        "positive_cosine": positive_cosine.detach(),
        "negative_cosine": negative_cosine,
        "valid_negatives": negative_counts.float().mean(),
    }


def trajectory_hierarchical_contrastive_loss(
    student,
    teacher,
    accelerator=None,
    num_patches=16,
    patch_temperature=0.2,
    global_temperature=0.2,
    negative_sim_threshold=0.95,
    min_negatives=32,
    patch_nce_coeff=1.0,
    global_nce_coeff=0.5,
    positive_coeff=0.25,
    std_coeff=0.1,
    cov_coeff=0.0,
    eps=1e-4,
):
    """Noise-aware contrast: mid-level patches and high-level global semantics."""
    if student.ndim != 4 or teacher.ndim != 4:
        raise ValueError("Expected student and teacher features shaped [B, K, T, D].")
    if student.shape[1] != 2:
        raise ValueError(
            f"Hierarchical contrast expects high/mid student timesteps, got {student.shape[1]}"
        )

    _, patch_components = trajectory_patch_infonce_loss(
        student[:, 1:2],
        teacher,
        accelerator=accelerator,
        num_patches=num_patches,
        temperature=patch_temperature,
        negative_sim_threshold=negative_sim_threshold,
        min_negatives=min_negatives,
        nce_coeff=1.0,
        std_coeff=0.0,
        cov_coeff=0.0,
        eps=eps,
    )
    _, global_components = trajectory_global_infonce_loss(
        student[:, 0:1],
        teacher,
        accelerator=accelerator,
        temperature=global_temperature,
        negative_sim_threshold=negative_sim_threshold,
        min_negatives=min_negatives,
        eps=eps,
    )
    std_loss, cov_loss = _trajectory_variance_covariance(student.float(), eps=eps)

    total = (
        patch_nce_coeff * patch_components["nce"]
        + global_nce_coeff * global_components["nce"]
        + positive_coeff * patch_components["sim"]
        + std_coeff * std_loss
        + cov_coeff * cov_loss
    )
    return total, {
        "patch_nce": patch_components["nce"],
        "patch_nce_raw": patch_components["nce_raw"],
        "patch_sim": patch_components["sim"],
        "patch_positive_cosine": patch_components["positive_cosine"],
        "patch_negative_cosine": patch_components["negative_cosine"],
        "patch_valid_negatives": patch_components["valid_negatives"],
        "global_nce": global_components["nce"],
        "global_nce_raw": global_components["nce_raw"],
        "global_positive_cosine": global_components["positive_cosine"],
        "global_negative_cosine": global_components["negative_cosine"],
        "global_valid_negatives": global_components["valid_negatives"],
        "std": std_loss,
        "cov": cov_loss,
    }


def _trajectory_variance_covariance(student, eps=1e-4):
    # Pool only for anti-collapse regularization; alignment itself remains patch-wise.
    pooled = student.mean(dim=2).reshape(-1, student.shape[-1])
    pooled = pooled - pooled.mean(dim=0)
    std = torch.sqrt(pooled.var(dim=0, unbiased=False) + eps)
    std_loss = F.relu(1 - std).mean()

    if pooled.shape[0] > 1:
        cov = (pooled.T @ pooled) / (pooled.shape[0] - 1)
        cov_loss = _off_diagonal(cov).pow(2).sum().div(pooled.shape[1])
    else:
        cov_loss = pooled.new_zeros(())

    return std_loss, cov_loss


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
