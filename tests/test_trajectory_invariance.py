import pytest
import torch

from loss import SILoss
from models.sit import SiT


def build_tiny_invariant_model(source_depth=2, invariant_dim=16):
    return SiT(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=64,
        decoder_hidden_size=64,
        encoder_depth=2,
        depth=4,
        num_heads=4,
        num_classes=10,
        class_dropout_prob=0.1,
        z_dims=[],
        trajectory_invariance=True,
        invariant_dim=invariant_dim,
        invariant_projector_dim=32,
        invariant_source_depth=source_depth,
        fused_attn=False,
        qk_norm=False,
    )


class RecordingModel:
    def __call__(
        self, x, t, invariant_group_count, invariant_view_count, **kwargs
    ):
        self.x = x.detach()
        self.t = t.detach()
        self.kwargs = kwargs
        group_batch = invariant_view_count * invariant_group_count
        features = x[:group_batch].reshape(group_batch, 16, 16)
        return {"x": torch.zeros_like(x), "zs": [], "invariance": {
            "features": features,
            "basis_orthogonality_loss": features.new_zeros(()),
        }}


def test_invariant_forward_is_training_only_and_has_expected_shape():
    model = build_tiny_invariant_model()
    paired_batch = 6
    output = model(
        torch.randn(paired_batch, 4, 8, 8),
        torch.rand(paired_batch),
        torch.randint(0, 10, (paired_batch,)),
        trajectory_pair=True,
        return_invariance=True,
        invariant_group_count=2,
        invariant_view_count=3,
    )

    assert output["x"].shape == (paired_batch, 4, 8, 8)
    assert output["invariance"]["features"].shape == (paired_batch, 16, 16)
    assert output["invariance"]["source_features"].shape == (
        paired_batch, 16, 64
    )
    assert torch.isfinite(output["invariance"]["basis_orthogonality_loss"])
    basis = model.invariance_head.normalized_basis()
    assert torch.allclose(
        basis.norm(dim=-1), torch.ones(16), atol=1e-6, rtol=1e-6
    )

    inference_output = model(
        torch.randn(2, 4, 8, 8),
        torch.rand(2),
        torch.randint(0, 10, (2,)),
    )
    assert "invariance" not in inference_output


def test_auxiliary_head_does_not_shift_seeded_backbone_initialization():
    common = dict(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=64,
        decoder_hidden_size=64,
        encoder_depth=2,
        depth=4,
        num_heads=4,
        num_classes=10,
        z_dims=[],
        fused_attn=False,
        qk_norm=False,
    )
    torch.manual_seed(19)
    baseline = SiT(**common)
    baseline_next_random = torch.rand(8)
    torch.manual_seed(19)
    invariant = SiT(
        **common,
        trajectory_invariance=True,
        invariant_dim=16,
        invariant_projector_dim=32,
        invariant_source_depth=2,
    )
    invariant_next_random = torch.rand(8)

    invariant_state = invariant.state_dict()
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, invariant_state[name]), name
    assert torch.equal(baseline_next_random, invariant_next_random)


def test_invariant_objective_backpropagates_to_projector_and_backbone():
    torch.manual_seed(11)
    model = build_tiny_invariant_model()
    loss_fn = SILoss(
        trajectory_invariance=True,
        projection=False,
        invariant_min_delta_t=0.05,
        invariant_max_delta_t=0.2,
    )
    losses = loss_fn(
        model,
        torch.randn(4, 4, 8, 8),
        model_kwargs={
            "y": torch.randint(0, 10, (4,)),
            "force_drop_ids": torch.tensor([False, True, False, True]),
        },
        invariant_batch_ratio=0.75,
    )
    objective_names = (
        "invariant_time_loss",
        "invariant_noise_loss",
        "invariant_image_variance_loss",
        "invariant_spatial_variance_loss",
        "invariant_covariance_loss",
        "invariant_relation_loss",
        "invariant_basis_loss",
    )
    total = losses["denoising_loss"].mean() + sum(
        losses[name] for name in objective_names
    )
    total.backward()

    assert model.invariance_head.projector.weight.grad is not None
    assert model.blocks[0].attn.qkv.weight.grad is not None
    assert losses["denoising_loss"].shape == (4,)
    assert losses["invariant_batch_fraction"].item() == 0.75
    assert 0.05 <= losses["invariant_mean_time_delta"].item() <= 0.2
    for name in objective_names:
        assert torch.isfinite(losses[name]).all()


def test_every_group_contains_time_and_noise_interventions():
    model = build_tiny_invariant_model()
    loss_fn = SILoss(
        trajectory_invariance=True,
        projection=False,
        invariant_min_delta_t=0.05,
        invariant_max_delta_t=0.2,
    )
    losses = loss_fn(
        model,
        torch.randn(4, 4, 8, 8),
        model_kwargs={"y": torch.randint(0, 10, (4,))},
    )

    assert losses["invariant_views_per_group"].item() == 3.0
    assert losses["invariant_time_reliability"].item() > 0.0
    assert losses["invariant_noise_reliability"].item() > 0.0


def test_three_view_group_isolates_time_noise_and_condition(monkeypatch):
    monkeypatch.setattr(
        torch, "randperm", lambda size, device=None: torch.arange(size, device=device)
    )
    images = torch.randn(2, 4, 8, 8)
    force_drop_ids = torch.tensor([False, True])
    model = RecordingModel()
    SILoss(
        trajectory_invariance=True,
        projection=False,
        invariant_min_delta_t=0.05,
        invariant_max_delta_t=0.2,
    )(
        model,
        images,
        model_kwargs={
            "y": torch.tensor([1, 2]),
            "force_drop_ids": force_drop_ids,
        },
    )

    time_a = model.t[:2].reshape(-1, 1, 1, 1)
    time_b = model.t[2:4].reshape(-1, 1, 1, 1)
    scaled_noise_a = (model.x[:2] - (1.0 - time_a) * images) * time_b
    scaled_noise_b = (model.x[2:4] - (1.0 - time_b) * images) * time_a
    assert torch.allclose(scaled_noise_a, scaled_noise_b, atol=2e-5, rtol=2e-5)
    assert torch.equal(
        model.kwargs["force_drop_ids"],
        torch.cat([force_drop_ids, force_drop_ids, force_drop_ids]),
    )
    assert torch.equal(model.t[:2], model.t[4:6])
    assert not torch.allclose(model.x[:2], model.x[4:6])
    assert model.t[:6].max().item() <= 0.8


def test_factorization_and_invariance_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        SiT(
            input_size=8,
            patch_size=2,
            in_channels=4,
            hidden_size=64,
            decoder_hidden_size=64,
            encoder_depth=2,
            depth=4,
            num_heads=4,
            num_classes=10,
            z_dims=[],
            trajectory_factorization=True,
            trajectory_invariance=True,
            fused_attn=False,
            qk_norm=False,
        )
