"""Export controlled SiT trajectory features for invariant/variant analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import CustomDataset
from models.sit import SiT_models


def saved_value(saved_args, name, default=None):
    if saved_args is None:
        return default
    if isinstance(saved_args, dict):
        return saved_args.get(name, default)
    return getattr(saved_args, name, default)


def infer_projector_dims(state_dict: dict[str, torch.Tensor]) -> list[int]:
    dims = []
    index = 0
    while f"projectors.{index}.4.weight" in state_dict:
        dims.append(state_dict[f"projectors.{index}.4.weight"].shape[0])
        index += 1
    return dims


def build_model(checkpoint: dict, args: argparse.Namespace, device: torch.device):
    state_dict = checkpoint[args.state_key]
    saved_args = checkpoint.get("args")
    model_name = args.model or saved_value(saved_args, "model")
    resolution = args.resolution or saved_value(saved_args, "resolution", 256)
    if model_name is None:
        raise ValueError("--model is required when the checkpoint has no saved args")

    has_factorization = any(
        key.startswith("factorization_head.") for key in state_dict
    )
    has_invariance = any(
        key.startswith("invariance_head.") for key in state_dict
    )
    if (
        has_factorization
        and "factorization_head.persistent_decoder.3.weight" not in state_dict
    ):
        raise ValueError(
            "checkpoint uses the pre-balanced TFCR head; export it with the "
            "matching historical commit or retrain balanced_additive_v1"
        )
    factor_dim = (
        state_dict["factorization_head.persistent_projector.3.weight"].shape[0]
        if has_factorization else 256
    )
    factor_projector_dim = (
        state_dict["factorization_head.persistent_projector.1.weight"].shape[0]
        if has_factorization else 1024
    )
    has_transition = any(
        key.startswith("factorization_head.transition_predictor.")
        for key in state_dict
    )
    detected_invariant_projector_type = (
        "linear"
        if "invariance_head.projector.weight" in state_dict
        else "mlp"
    )
    saved_invariant_projector_type = saved_value(
        saved_args, "invariant_projector_type", detected_invariant_projector_type
    )
    if (
        has_invariance
        and saved_invariant_projector_type != detected_invariant_projector_type
    ):
        raise ValueError(
            "saved invariant projector type disagrees with checkpoint keys: "
            f"{saved_invariant_projector_type!r} versus "
            f"{detected_invariant_projector_type!r}"
        )
    if has_invariance and detected_invariant_projector_type == "linear":
        invariant_dim = state_dict["invariance_head.projector.weight"].shape[0]
        invariant_projector_dim = 1024
    elif has_invariance:
        invariant_dim = state_dict[
            "invariance_head.projector.3.weight"
        ].shape[0]
        invariant_projector_dim = state_dict[
            "invariance_head.projector.1.weight"
        ].shape[0]
    else:
        invariant_dim, invariant_projector_dim = 256, 1024
    num_classes = saved_value(saved_args, "num_classes", 1000)
    label_rows = state_dict["y_embedder.embedding_table.weight"].shape[0]

    model = SiT_models[model_name](
        input_size=resolution // 8,
        num_classes=num_classes,
        use_cfg=label_rows > num_classes,
        class_dropout_prob=(
            saved_value(saved_args, "cfg_prob", 0.1)
            if label_rows > num_classes else 0.0
        ),
        z_dims=infer_projector_dims(state_dict),
        encoder_depth=saved_value(saved_args, "encoder_depth", 8),
        skip_layer_connection=saved_value(
            saved_args, "skip_layer_connection", False
        ),
        block_diversity_loss=False,
        trajectory_factorization=has_factorization,
        factor_dim=factor_dim,
        factor_projector_dim=factor_projector_dim,
        factor_source_depth=saved_value(saved_args, "factor_source_depth", None),
        factor_target_depth=saved_value(saved_args, "factor_target_depth", None),
        factor_transition=has_transition,
        trajectory_invariance=has_invariance,
        invariant_dim=invariant_dim,
        invariant_projector_dim=invariant_projector_dim,
        invariant_source_depth=saved_value(
            saved_args, "invariant_source_depth", None
        ),
        invariant_projector_type=detected_invariant_projector_type,
        fused_attn=False,
        qk_norm=saved_value(saved_args, "qk_norm", False),
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, saved_args, resolution, has_factorization, has_invariance


def sample_latent(moments: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    moments = moments.squeeze(1)
    mean, std = torch.chunk(moments, 2, dim=1)
    posterior_noise = torch.randn(
        mean.shape, generator=generator, device=mean.device, dtype=mean.dtype
    )
    return (mean + std * posterior_noise) * 0.18215


def interpolate(x0: torch.Tensor, noise: torch.Tensor, timestep: float, path: str):
    t = x0.new_tensor(timestep)
    if path == "linear":
        return (1 - t) * x0 + t * noise
    if path == "cosine":
        signal = torch.cos(t * torch.pi / 2)
        noise_scale = torch.sin(t * torch.pi / 2)
        return signal * x0 + noise_scale * noise
    raise ValueError(f"unsupported path type: {path}")


def capture_depths(model, depths: list[int]):
    captured = {}
    handles = []
    for depth in depths:
        if not 1 <= depth <= len(model.blocks):
            raise ValueError(f"depth {depth} is outside [1, {len(model.blocks)}]")

        def hook(_module, _inputs, output, current_depth=depth):
            captured[current_depth] = output

        handles.append(model.blocks[depth - 1].register_forward_hook(hook))
    return captured, handles


def stratified_indices(
    labels: np.ndarray, count: int, class_count: int, seed: int
) -> list[int]:
    """Choose repeated examples per class so class probes are meaningful."""
    rng = np.random.default_rng(seed)
    available_classes = np.unique(labels)
    selected_classes = rng.choice(
        available_classes,
        size=min(class_count, len(available_classes)),
        replace=False,
    )
    pools = []
    for label in selected_classes:
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        pools.append(indices.tolist())
    selected = []
    while len(selected) < count and any(pools):
        for pool in pools:
            if pool and len(selected) < count:
                selected.append(pool.pop())
    return selected


@torch.no_grad()
def extract(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.num_samples <= 0 or args.num_noises <= 0:
        raise ValueError("--num-samples and --num-noises must be positive")
    if args.analysis_class_count <= 0:
        raise ValueError("--analysis-class-count must be positive")
    if len(args.timesteps) < 2 or any(
        current >= following
        for current, following in zip(args.timesteps, args.timesteps[1:])
    ):
        raise ValueError("--timesteps must contain two or more increasing values")
    if any(not 0.0 <= timestep <= 1.0 for timestep in args.timesteps):
        raise ValueError("all --timesteps must be in [0, 1]")
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if args.state_key not in checkpoint:
        raise KeyError(f"checkpoint has no '{args.state_key}' state dictionary")
    (
        model,
        saved_args,
        resolution,
        has_factorization,
        has_invariance,
    ) = build_model(checkpoint, args, device)
    path_type = saved_value(saved_args, "path_type", "linear")
    depths = list(args.depths) if args.depths else [
        1,
        len(model.blocks) // 4,
        len(model.blocks) // 2,
        3 * len(model.blocks) // 4,
        len(model.blocks),
    ]
    invariant_source_depth = None
    if has_invariance:
        invariant_source_depth = model.invariant_source_depth
        depths.append(invariant_source_depth)
    depths = sorted(set(max(1, depth) for depth in depths))
    captured, handles = capture_depths(model, depths)
    variant_basis = None
    if (
        has_invariance
        and model.invariance_head.projector_type == "linear"
    ):
        variant_basis = model.invariance_head.orthonormal_basis()

    dataset = CustomDataset(args.data_dir)
    if len(dataset) == 0:
        raise ValueError("--data-dir contains no examples")
    if dataset.labels.ndim != 1:
        raise ValueError("trajectory analysis requires scalar class labels")
    sample_count = min(args.num_samples, len(dataset))
    sample_indices = stratified_indices(
        dataset.labels, sample_count, args.analysis_class_count, args.seed
    )
    loader = DataLoader(
        Subset(dataset, sample_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    feature_batches = []
    persistent_batches = []
    evolving_batches = []
    invariant_batches = []
    variant_batches = []
    label_batches = []

    try:
        for _raw_images, moments, labels in loader:
            moments = moments.to(device)
            labels = labels.to(device)
            x0 = sample_latent(moments, generator)
            noise_bank = torch.randn(
                (x0.shape[0], args.num_noises, *x0.shape[1:]),
                generator=generator,
                device=device,
                dtype=x0.dtype,
            )
            batch_features, batch_persistent, batch_evolving = [], [], []
            batch_invariant, batch_variant = [], []
            for timestep in args.timesteps:
                time_features, time_persistent, time_evolving = [], [], []
                time_invariant, time_variant = [], []
                for noise_index in range(args.num_noises):
                    model_input = interpolate(
                        x0, noise_bank[:, noise_index], timestep, path_type
                    )
                    captured.clear()
                    output = model(
                        model_input,
                        torch.full((x0.shape[0],), timestep, device=device),
                        labels,
                        return_factorization=has_factorization,
                        return_invariance=has_invariance,
                    )
                    pooled = torch.stack(
                        [
                            captured[depth].float().mean(dim=1)
                            for depth in depths
                        ],
                        dim=1,
                    )
                    time_features.append(pooled.cpu())
                    if has_factorization:
                        factors = output["factorization"]
                        time_persistent.append(
                            factors["persistent"].float().mean(dim=1).cpu()
                        )
                        time_evolving.append(
                            factors["evolving"].float().mean(dim=1).cpu()
                        )
                    if has_invariance:
                        invariance = output["invariance"]
                        if variant_basis is not None:
                            # Use orthonormal row-space coordinates so the
                            # reported invariant and complementary energies are
                            # geometrically meaningful even before W is exactly
                            # orthogonal.  source_features is the post-skip
                            # tensor actually consumed by the projector.
                            source_feature = invariance["source_features"].float()
                            coordinates = source_feature @ variant_basis
                            projection = coordinates @ variant_basis.T
                            time_invariant.append(
                                coordinates.mean(dim=1).cpu()
                            )
                            time_variant.append(
                                (source_feature - projection).mean(dim=1).cpu()
                            )
                        else:
                            time_invariant.append(
                                invariance["features"].float().mean(dim=1).cpu()
                            )
                batch_features.append(torch.stack(time_features, dim=1))
                if has_factorization:
                    batch_persistent.append(torch.stack(time_persistent, dim=1))
                    batch_evolving.append(torch.stack(time_evolving, dim=1))
                if has_invariance:
                    batch_invariant.append(torch.stack(time_invariant, dim=1))
                    if time_variant:
                        batch_variant.append(torch.stack(time_variant, dim=1))
            # [B, time, noise, layer, channel]
            feature_batches.append(torch.stack(batch_features, dim=1).numpy())
            if has_factorization:
                persistent_batches.append(
                    torch.stack(batch_persistent, dim=1).numpy()
                )
                evolving_batches.append(torch.stack(batch_evolving, dim=1).numpy())
            if has_invariance:
                invariant_batches.append(
                    torch.stack(batch_invariant, dim=1).numpy()
                )
                if batch_variant:
                    variant_batches.append(
                        torch.stack(batch_variant, dim=1).numpy()
                    )
            label_batches.append(labels.cpu().numpy())
    finally:
        for handle in handles:
            handle.remove()

    payload = {
        "features": np.concatenate(feature_batches, axis=0),
        "labels": np.concatenate(label_batches, axis=0),
        "timesteps": np.asarray(args.timesteps, dtype=np.float32),
        "depths": np.asarray(depths, dtype=np.int64),
        "method": np.asarray(args.method),
        "resolution": np.asarray(resolution),
        "sample_indices": np.asarray(sample_indices, dtype=np.int64),
    }
    if has_factorization:
        payload["persistent"] = np.concatenate(persistent_batches, axis=0)
        payload["evolving"] = np.concatenate(evolving_batches, axis=0)
    if has_invariance:
        payload["invariant"] = np.concatenate(invariant_batches, axis=0)
        payload["invariant_source_depth"] = np.asarray(invariant_source_depth)
        if variant_batches:
            payload["variant"] = np.concatenate(variant_batches, axis=0)
            payload["invariant_basis_rank"] = np.asarray(
                variant_basis.shape[1]
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"Saved controlled features {payload['features'].shape} to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--state-key", choices=["ema", "model"], default="ema")
    parser.add_argument("--depths", type=int, nargs="+", default=None)
    parser.add_argument(
        "--timesteps", type=float, nargs="+", default=[0.05, 0.25, 0.5, 0.75, 0.95]
    )
    parser.add_argument("--num-noises", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument(
        "--analysis-class-count",
        type=int,
        default=64,
        help="number of classes sampled with repeated examples",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    extract(parse_args())
