import numpy as np

from analysis.trajectory_metrics import (
    cross_timestep_retrieval,
    layer_cka,
    variance_partition,
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
    cka = layer_cka(features)
    assert cka.shape == (2, 2)
    assert np.allclose(np.diag(cka), 1.0)
    assert np.all((cka >= -1e-6) & (cka <= 1.0 + 1e-6))
