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


def ridge_predict(
    train_x: np.ndarray,
    train_targets: np.ndarray,
    test_x: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """Return ridge scores using the smaller of the primal and dual systems."""
    sample_count, feature_count = train_x.shape
    if sample_count <= feature_count:
        kernel = train_x @ train_x.T
        dual = np.linalg.solve(
            kernel + ridge * np.eye(sample_count), train_targets
        )
        return test_x @ train_x.T @ dual
    covariance = train_x.T @ train_x
    weights = np.linalg.solve(
        covariance + ridge * np.eye(feature_count), train_x.T @ train_targets
    )
    return test_x @ weights


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


def cross_noise_retrieval(features: np.ndarray) -> float:
    """Retrieve source identity across independent noise realizations."""
    if features.shape[2] < 2:
        return float("nan")
    source_noise = features.mean(axis=1)
    query = source_noise[:, 0]
    gallery = source_noise[:, -1]
    query = query / np.maximum(np.linalg.norm(query, axis=-1, keepdims=True), EPS)
    gallery = gallery / np.maximum(np.linalg.norm(gallery, axis=-1, keepdims=True), EPS)
    prediction = (query @ gallery.T).argmax(axis=1)
    return float(np.mean(prediction == np.arange(features.shape[0])))


def within_class_retrieval(
    query: np.ndarray, gallery: np.ndarray, labels: np.ndarray
) -> float:
    """Retrieve instances only among same-class candidates."""
    query = query / np.maximum(np.linalg.norm(query, axis=-1, keepdims=True), EPS)
    gallery = gallery / np.maximum(
        np.linalg.norm(gallery, axis=-1, keepdims=True), EPS
    )
    similarities = query @ gallery.T
    predictions = []
    targets = []
    for source_index, label in enumerate(labels):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < 2:
            continue
        predictions.append(candidates[similarities[source_index, candidates].argmax()])
        targets.append(source_index)
    if not targets:
        return float("nan")
    return float(np.mean(np.asarray(predictions) == np.asarray(targets)))


def within_class_cross_timestep_retrieval(
    features: np.ndarray, labels: np.ndarray
) -> float:
    """Retrieve the source across endpoint timesteps within its class."""
    source_time = features.mean(axis=2)
    return within_class_retrieval(source_time[:, 0], source_time[:, -1], labels)


def within_class_cross_noise_retrieval(
    features: np.ndarray, labels: np.ndarray
) -> float:
    """Retrieve the source across noise views within its class."""
    if features.shape[2] < 2:
        return float("nan")
    source_noise = features.mean(axis=1)
    return within_class_retrieval(source_noise[:, 0], source_noise[:, -1], labels)


def within_source_noise_retrieval(features: np.ndarray) -> float:
    """Match the same noise realization across the first and last timestep.

    Noise-bank indices are meaningful only within a source image, so retrieval
    is performed separately for each source rather than treating an arbitrary
    noise index as a dataset-wide class.
    """
    if features.shape[1] < 2 or features.shape[2] < 2:
        return float("nan")
    query = features[:, 0]
    gallery = features[:, -1]
    query = query - query.mean(axis=1, keepdims=True)
    gallery = gallery - gallery.mean(axis=1, keepdims=True)
    query = query / np.maximum(np.linalg.norm(query, axis=-1, keepdims=True), EPS)
    gallery = gallery / np.maximum(
        np.linalg.norm(gallery, axis=-1, keepdims=True), EPS
    )
    similarities = np.einsum("snc,smc->snm", query, gallery)
    prediction = similarities.argmax(axis=-1)
    targets = np.broadcast_to(np.arange(features.shape[2]), prediction.shape)
    return float(np.mean(prediction == targets))


def timestep_probe_accuracy(
    features: np.ndarray, seed: int = 0, ridge: float = 1e-2
) -> float:
    """Predict timestep on held-out sources after averaging noise replicates."""
    source_count, timestep_count = features.shape[:2]
    if source_count < 2 or timestep_count < 2:
        return float("nan")
    source_time = features.mean(axis=2).astype(np.float64)
    # Remove each source's stable offset so a large invariant component cannot
    # swamp a smaller but linearly decodable timestep direction.
    source_time = source_time - source_time.mean(axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    source_indices = rng.permutation(source_count)
    split = min(source_count - 1, max(1, int(0.8 * source_count)))
    train_sources = source_indices[:split]
    test_sources = source_indices[split:]
    train_x = source_time[train_sources].reshape(-1, source_time.shape[-1])
    test_x = source_time[test_sources].reshape(-1, source_time.shape[-1])
    train_y = np.tile(np.arange(timestep_count), len(train_sources))
    test_y = np.tile(np.arange(timestep_count), len(test_sources))

    mean = train_x.mean(axis=0, keepdims=True)
    scale = train_x.std(axis=0, keepdims=True) + 1e-6
    train_x = (train_x - mean) / scale
    test_x = (test_x - mean) / scale
    targets = np.eye(timestep_count, dtype=np.float64)[train_y]
    prediction = ridge_predict(train_x, targets, test_x, ridge).argmax(axis=1)
    return float(np.mean(prediction == test_y))


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
    scores = ridge_predict(train_x, targets[train], test_x, ridge)
    prediction = scores.argmax(axis=1)
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
        row["cross_noise_retrieval"] = cross_noise_retrieval(layer_features)
        row["timestep_probe_accuracy"] = timestep_probe_accuracy(layer_features)
        row["within_source_noise_retrieval"] = (
            within_source_noise_retrieval(layer_features)
        )
        if labels is not None:
            row["class_probe_accuracy"] = ridge_classification_accuracy(
                layer_features, labels
            )
            row["within_class_cross_timestep_retrieval"] = (
                within_class_cross_timestep_retrieval(layer_features, labels)
            )
            row["within_class_cross_noise_retrieval"] = (
                within_class_cross_noise_retrieval(layer_features, labels)
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
            for name in ("persistent", "evolving", "invariant", "variant")
            if name in archive
        }
        method = (
            str(archive["method"].item()) if "method" in archive else path.stem
        )
        invariant_kind = (
            str(archive["invariant_kind"].item())
            if "invariant_kind" in archive else None
        )

    validate_archive(features, depths, timesteps, labels, branches)
    report = {
        "schema": "trajectory-representation-v1",
        "method": method,
        "invariant_kind": invariant_kind,
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
        report[name]["cross_noise_retrieval"] = cross_noise_retrieval(branch)
        report[name]["timestep_probe_accuracy"] = timestep_probe_accuracy(branch)
        report[name]["within_source_noise_retrieval"] = (
            within_source_noise_retrieval(branch)
        )
        if labels is not None:
            report[name]["class_probe_accuracy"] = (
                ridge_classification_accuracy(branch, labels)
            )
            report[name]["within_class_cross_timestep_retrieval"] = (
                within_class_cross_timestep_retrieval(branch, labels)
            )
            report[name]["within_class_cross_noise_retrieval"] = (
                within_class_cross_noise_retrieval(branch, labels)
            )
    if "invariant" in report and "variant" in report:
        report["subspace_contrast"] = {
            key: report["invariant"][key] - report["variant"][key]
            for key in (
                "source_fraction",
                "timestep_fraction",
                "noise_fraction",
                "cross_timestep_retrieval",
                "cross_noise_retrieval",
                "timestep_probe_accuracy",
                "within_source_noise_retrieval",
            )
        }
        invariant_width = branches["invariant"].shape[-1]
        variant_width = branches["variant"].shape[-1]
        captured_energy = {}
        for component in ("source", "timestep", "noise", "total"):
            invariant_energy = (
                report["invariant"][f"{component}_energy"] * invariant_width
            )
            variant_energy = (
                report["variant"][f"{component}_energy"] * variant_width
            )
            captured_energy[f"{component}_fraction"] = float(
                invariant_energy
                / max(invariant_energy + variant_energy, EPS)
            )
        report["subspace_energy_capture"] = captured_energy
        if labels is not None:
            report["subspace_contrast"].update({
                key: report["invariant"][key] - report["variant"][key]
                for key in (
                    "class_probe_accuracy",
                    "within_class_cross_timestep_retrieval",
                    "within_class_cross_noise_retrieval",
                )
            })
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
