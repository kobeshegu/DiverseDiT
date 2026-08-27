"""Quantify what stays and what changes in diffusion representations.

The input is a balanced tensor with axes [source, timestep, noise, layer,
channel].  A hierarchical variance decomposition then assigns feature energy
to source-stable, timestep-varying, and noise-varying components without
assuming that any component is semantic, structural, or textural.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EPS = 1e-12


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA after centering examples."""
    x = x.reshape(x.shape[0], -1).astype(np.float64)
    y = y.reshape(y.shape[0], -1).astype(np.float64)
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    norm_x = np.linalg.norm(x.T @ x, ord="fro")
    norm_y = np.linalg.norm(y.T @ y, ord="fro")
    return float(cross / max(norm_x * norm_y, EPS))


def variance_partition(features: np.ndarray) -> dict[str, float]:
    """Apply the law of total variance to [source, time, noise, channel]."""
    if features.ndim != 4:
        raise ValueError("expected [source, timestep, noise, channel]")
    x = features.astype(np.float64)
    source_mean = x.mean(axis=(1, 2), keepdims=True)
    source_time_mean = x.mean(axis=2, keepdims=True)
    grand_mean = x.mean(axis=(0, 1, 2), keepdims=True)

    source_energy = np.mean((source_mean - grand_mean) ** 2)
    timestep_energy = np.mean((source_time_mean - source_mean) ** 2)
    noise_energy = np.mean((x - source_time_mean) ** 2)
    total = source_energy + timestep_energy + noise_energy
    return {
        "source_energy": float(source_energy),
        "timestep_energy": float(timestep_energy),
        "noise_energy": float(noise_energy),
        "source_fraction": float(source_energy / max(total, EPS)),
        "timestep_fraction": float(timestep_energy / max(total, EPS)),
        "noise_fraction": float(noise_energy / max(total, EPS)),
        "total_energy": float(total),
    }


def cross_timestep_retrieval(features: np.ndarray) -> float:
    """Retrieve source identity between the cleanest and noisiest views."""
    source_time = features.mean(axis=2)
    query = source_time[:, 0]
    gallery = source_time[:, -1]
    query = query / np.maximum(np.linalg.norm(query, axis=-1, keepdims=True), EPS)
    gallery = gallery / np.maximum(np.linalg.norm(gallery, axis=-1, keepdims=True), EPS)
    prediction = (query @ gallery.T).argmax(axis=1)
    return float(np.mean(prediction == np.arange(features.shape[0])))


def timestep_cka(features: np.ndarray) -> np.ndarray:
    """CKA matrix across timesteps after averaging noise replicates."""
    source_time = features.mean(axis=2)
    count = source_time.shape[1]
    result = np.empty((count, count), dtype=np.float64)
    for i in range(count):
        for j in range(i, count):
            value = linear_cka(source_time[:, i], source_time[:, j])
            result[i, j] = result[j, i] = value
    return result


def layer_cka(features: np.ndarray) -> np.ndarray:
    """CKA across layers after removing controlled-view replication."""
    count = features.shape[3]
    source_features = features.mean(axis=(1, 2))
    result = np.empty((count, count), dtype=np.float64)
    for i in range(count):
        for j in range(i, count):
            value = linear_cka(source_features[:, i], source_features[:, j])
            result[i, j] = result[j, i] = value
    return result


def ridge_classification_accuracy(
    features: np.ndarray, labels: np.ndarray, seed: int = 0, ridge: float = 1e-2
) -> float:
    """Small closed-form linear probe on source-averaged representations."""
    if len(np.unique(labels)) < 2 or len(labels) < 8:
        return float("nan")
    x = features.mean(axis=(1, 2)).astype(np.float64)
    classes, encoded = np.unique(labels, return_inverse=True)
    targets = np.eye(len(classes), dtype=np.float64)[encoded]
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for class_index in range(len(classes)):
        indices = np.flatnonzero(encoded == class_index)
        rng.shuffle(indices)
        split = min(len(indices) - 1, max(1, int(0.8 * len(indices))))
        if split <= 0:
            continue
        train_parts.append(indices[:split])
        test_parts.append(indices[split:])
    if not train_parts or not test_parts:
        return float("nan")
    train = np.concatenate(train_parts)
    test = np.concatenate(test_parts)
    if len(test) == 0:
        return float("nan")

    mean = x[train].mean(axis=0, keepdims=True)
    scale = x[train].std(axis=0, keepdims=True) + 1e-6
    train_x = (x[train] - mean) / scale
    test_x = (x[test] - mean) / scale
    # The dual form avoids a potentially very large channel-by-channel solve.
    kernel = train_x @ train_x.T
    dual = np.linalg.solve(
        kernel + ridge * np.eye(len(train)), targets[train]
    )
    prediction = (test_x @ train_x.T @ dual).argmax(axis=1)
    return float(np.mean(prediction == encoded[test]))


def summarize_layers(
    features: np.ndarray, depths: np.ndarray, labels: np.ndarray | None
) -> list[dict[str, float]]:
    if features.ndim != 5:
        raise ValueError("features must have shape [source, timestep, noise, layer, channel]")
    rows = []
    for layer_index, depth in enumerate(depths.tolist()):
        layer_features = features[:, :, :, layer_index]
        row = {"depth": int(depth), **variance_partition(layer_features)}
        row["cross_timestep_retrieval"] = cross_timestep_retrieval(layer_features)
        if labels is not None:
            row["class_probe_accuracy"] = ridge_classification_accuracy(
                layer_features, labels
            )
        rows.append(row)
    return rows


def validate_archive(
    features: np.ndarray,
    depths: np.ndarray,
    timesteps: np.ndarray,
    labels: np.ndarray | None,
    branches: dict[str, np.ndarray],
) -> None:
    """Reject incomplete archives before computing potentially misleading metrics."""
    if features.ndim != 5:
        raise ValueError(
            "features must have shape [source, timestep, noise, layer, channel]"
        )
    if any(size == 0 for size in features.shape):
        raise ValueError("all feature axes must be non-empty")
    if depths.ndim != 1 or len(depths) != features.shape[3]:
        raise ValueError("depths must contain one entry per feature layer")
    if timesteps.ndim != 1 or len(timesteps) != features.shape[1]:
        raise ValueError("timesteps must contain one entry per trajectory view")
    if len(timesteps) < 2 or np.any(np.diff(timesteps) <= 0):
        raise ValueError(
            "timesteps must be strictly increasing and contain two or more views"
        )
    if labels is not None and (
        labels.ndim != 1 or len(labels) != features.shape[0]
    ):
        raise ValueError("labels must contain one scalar label per source example")
    if not np.isfinite(features).all():
        raise ValueError("features contain NaN or infinite values")
    expected_prefix = features.shape[:3]
    for name, branch in branches.items():
        if branch.ndim != 4 or branch.shape[:3] != expected_prefix:
            raise ValueError(
                f"{name} must have shape [source, timestep, noise, channel]"
            )
        if not np.isfinite(branch).all():
            raise ValueError(f"{name} contains NaN or infinite values")


def analyze_archive(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        required = {"features", "depths", "timesteps"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"archive is missing required arrays: {sorted(missing)}")
        features = archive["features"]
        depths = archive["depths"]
        timesteps = archive["timesteps"]
        labels = archive["labels"] if "labels" in archive else None
        branches = {
            name: archive[name]
            for name in ("persistent", "evolving")
            if name in archive
        }
        method = (
            str(archive["method"].item()) if "method" in archive else path.stem
        )

    validate_archive(features, depths, timesteps, labels, branches)
    report = {
        "schema": "trajectory-representation-v1",
        "method": method,
        "source": str(path),
        "shape": list(features.shape),
        "timesteps": timesteps.tolist(),
        "layers": summarize_layers(features, depths, labels),
        "layer_cka": layer_cka(features).tolist(),
        "timestep_cka_by_layer": [
            timestep_cka(features[:, :, :, index]).tolist()
            for index in range(features.shape[3])
        ],
    }
    for name, branch in branches.items():
        report[name] = variance_partition(branch)
        report[name]["cross_timestep_retrieval"] = cross_timestep_retrieval(
            branch
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("analysis/results.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [analyze_archive(path) for path in args.archives]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {len(reports)} method report(s) to {args.output}")


if __name__ == "__main__":
    main()
