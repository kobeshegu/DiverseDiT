import torch

from loss import SILoss
from models.sit import SiT


def build_tiny_model(transition=False):
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
        z_dims=[],
        trajectory_factorization=True,
        factor_dim=16,
        factor_projector_dim=32,
        factor_source_depth=2,
        factor_target_depth=4,
        factor_transition=transition,
        fused_attn=False,
        qk_norm=False,
    )


def test_factorized_forward_shapes_and_swap_outputs():
    model = build_tiny_model(transition=True)
    batch_size = 2
    paired_batch = 2 * batch_size
    output = model(
        torch.randn(paired_batch, 4, 8, 8),
        torch.rand(paired_batch),
        torch.randint(0, 10, (paired_batch,)),
        trajectory_pair=True,
        factor_delta_t=torch.tensor([0.2, -0.4]),
    )

    factors = output["factorization"]
    assert output["x"].shape == (paired_batch, 4, 8, 8)
    assert factors["persistent"].shape == (paired_batch, 16, 16)
    assert factors["evolving"].shape == (paired_batch, 16, 16)
    assert factors["target"].shape == (paired_batch, 16, 64)
    assert factors["recomposed"].shape == factors["target"].shape
    assert factors["transitioned"].shape == factors["evolving"].shape


def test_factorization_loss_backpropagates_to_both_branches():
    torch.manual_seed(7)
    model = build_tiny_model(transition=True)
    loss_fn = SILoss(
        trajectory_factorization=True,
        projection=False,
        factor_pair_cross_noise_prob=0.5,
        factor_min_delta_t=0.2,
        factor_max_delta_t=0.4,
        factor_transition=True,
    )
    losses = loss_fn(
        model,
        torch.randn(2, 4, 8, 8),
        model_kwargs={"y": torch.randint(0, 10, (2,))},
    )
    total = (
        losses["denoising_loss"].mean()
        + losses["factor_inv_loss"].mean()
        + losses["factor_recom_loss"].mean()
        + losses["factor_transition_loss"].mean()
    )
    total.backward()

    head = model.factorization_head
    assert head.persistent_projector[-1].weight.grad is not None
    assert head.evolving_projector[-1].weight.grad is not None
    assert head.recomposer[-1].weight.grad is not None
    assert head.transition_predictor[-1].weight.grad is not None
    assert losses["factor_inv_loss"].shape == (2,)
    assert losses["factor_recom_loss"].shape == (2,)
    assert 0.2 <= losses["mean_delta_t"].item() <= 0.4


def test_non_factorized_path_remains_available_without_repa():
    model = SiT(
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
    losses = SILoss(projection=False)(
        model,
        torch.randn(2, 4, 8, 8),
        model_kwargs={"y": torch.randint(0, 10, (2,))},
    )
    assert losses["denoising_loss"].shape == (2,)
    assert losses["proj_loss"].item() == 0.0
