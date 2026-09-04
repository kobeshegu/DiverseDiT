import numpy as np

from analysis.trajectory_metrics import (
    analyze_archive,
    cross_noise_retrieval,
    cross_timestep_retrieval,
    layer_cka,
    timestep_probe_accuracy,
    variance_partition,
    within_class_cross_noise_retrieval,
    within_class_cross_timestep_retrieval,
    within_source_noise_retrieval,
)


def synthetic_features():
    rng = np.random.default_rng(5)
    sources, timesteps, noises, layers, channels = 16, 3, 4, 2, 24
    source_signal = 3.0 * rng.standard_normal((sources, 1, 1, 1, channels))
    time_signal = rng.standard_normal((1, timesteps, 1, 1, channels))
    noise_signal = 0.25 * rng.standard_normal(
        (sources, timesteps, noises, layers, channels)
    )
    return source_signal + time_signal + noise_signal


def test_variance_partition_is_complete_and_identifies_dominant_source_signal():
    features = synthetic_features()[:, :, :, 0]
    result = variance_partition(features)
    fractions = (
        result["source_fraction"]
        + result["timestep_fraction"]
        + result["noise_fraction"]
    )
    assert np.isclose(fractions, 1.0)
    assert result["source_fraction"] > result["timestep_fraction"]
    assert result["timestep_fraction"] > result["noise_fraction"]


def test_controlled_views_retrieve_source_and_produce_valid_layer_cka():
    features = synthetic_features()
    assert cross_timestep_retrieval(features[:, :, :, 0]) > 0.9
    assert cross_noise_retrieval(features[:, :, :, 0]) > 0.9
    assert timestep_probe_accuracy(features[:, :, :, 0]) > 0.9
    labels = np.repeat(np.arange(4), 4)
    assert within_class_cross_timestep_retrieval(
        features[:, :, :, 0], labels
    ) > 0.9
    assert within_class_cross_noise_retrieval(
        features[:, :, :, 0], labels
    ) > 0.9
    cka = layer_cka(features)
    assert cka.shape == (2, 2)
    assert np.allclose(np.diag(cka), 1.0)
    assert np.all((cka >= -1e-6) & (cka <= 1.0 + 1e-6))


def test_noise_retrieval_tracks_a_noise_component_across_timesteps():
    rng = np.random.default_rng(17)
    sources, timesteps, noises, channels = 8, 3, 4, 32
    source_signal = rng.standard_normal((sources, 1, 1, channels))
    noise_signal = 2.0 * rng.standard_normal((sources, 1, noises, channels))
    features = source_signal + noise_signal + 0.01 * rng.standard_normal(
        (sources, timesteps, noises, channels)
    )
    assert within_source_noise_retrieval(features) > 0.9


def test_invariant_and_variant_archives_report_subspace_contrast(tmp_path):
    features = synthetic_features()
    invariant = features[:, :, :, 0]
    rng = np.random.default_rng(13)
    variant = rng.standard_normal(invariant.shape)
    archive_path = tmp_path / "subspace.npz"
    np.savez_compressed(
        archive_path,
        features=features,
        invariant=invariant,
        variant=variant,
        depths=np.asarray([1, 2]),
        timesteps=np.asarray([0.1, 0.5, 0.9]),
        labels=np.repeat(np.arange(4), 4),
        method=np.asarray("orbit-subspace"),
    )

    report = analyze_archive(archive_path)
    assert "invariant" in report
    assert "variant" in report
    assert "subspace_contrast" in report
    assert "subspace_energy_capture" in report
    assert report["invariant"]["cross_noise_retrieval"] > 0.9
    assert "timestep_probe_accuracy" in report["subspace_contrast"]
    assert "within_source_noise_retrieval" in report["subspace_contrast"]
    assert "within_class_cross_timestep_retrieval" in report["subspace_contrast"]
    assert "within_class_cross_noise_retrieval" in report["subspace_contrast"]
    assert 0.0 <= report["subspace_energy_capture"]["total_fraction"] <= 1.0
