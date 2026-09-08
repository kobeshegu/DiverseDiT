import torch
import torch.nn.functional as F

from loss import SILoss
from models.sit import SiT, gradient_reverse


def build_tiny_model(
    transition=False, velocity_recomposition=False, adversarial=False
):
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
        factor_velocity_recomposition=velocity_recomposition,
        factor_adversarial=adversarial,
        factor_adversarial_timestep_bins=4,
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
        return_factorization=True,
    )

    factors = output["factorization"]
    assert output["x"].shape == (paired_batch, 4, 8, 8)
    assert factors["persistent"].shape == (paired_batch, 16, 16)
    assert factors["evolving"].shape == (paired_batch, 16, 16)
    assert factors["target"].shape == (paired_batch, 16, 64)
    assert factors["persistent_component"].shape == factors["target"].shape
    assert factors["evolving_component"].shape == factors["target"].shape
    assert factors["recomposed"].shape == factors["target"].shape
    assert factors["transitioned"].shape == factors["evolving"].shape

    inference_output = model(
        torch.randn(batch_size, 4, 8, 8),
        torch.rand(batch_size),
        torch.randint(0, 10, (batch_size,)),
    )
    assert "factorization" not in inference_output


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
        + losses["factor_persistent_loss"].mean()
        + losses["factor_evolving_loss"].mean()
        + losses["factor_recom_loss"].mean()
        + losses["factor_transition_loss"].mean()
    )
    total.backward()

    head = model.factorization_head
    assert head.persistent_projector[-1].weight.grad is not None
    assert head.evolving_projector[-1].weight.grad is not None
    assert head.persistent_decoder[-1].weight.grad is not None
    assert head.evolving_decoder[-1].weight.grad is not None
    assert head.transition_predictor[-1].weight.grad is not None
    assert losses["factor_inv_loss"].shape == (2,)
    assert losses["factor_persistent_loss"].shape == (2,)
    assert losses["factor_evolving_loss"].shape == (2,)
    assert losses["factor_recom_loss"].shape == (2,)
    assert torch.isfinite(losses["persistent_usage_gap"])
    assert torch.isfinite(losses["evolving_usage_gap"])
    assert torch.allclose(
        losses["common_energy_fraction"] + losses["residual_energy_fraction"],
        torch.tensor(1.0),
        atol=1e-5,
    )
    assert 0.2 <= losses["mean_delta_t"].item() <= 0.4


def test_factor_batch_ratio_adds_views_only_for_selected_sources():
    model = build_tiny_model()
    loss_fn = SILoss(
        trajectory_factorization=True,
        projection=False,
        factor_min_delta_t=0.2,
        factor_max_delta_t=0.4,
    )
    losses = loss_fn(
        model,
        torch.randn(4, 4, 8, 8),
        model_kwargs={"y": torch.randint(0, 10, (4,))},
        factor_batch_ratio=0.5,
    )
    assert losses["denoising_loss"].shape == (4,)
    assert losses["factor_inv_loss"].shape == (2,)
    assert losses["factor_batch_fraction"].item() == 0.5


def test_transition_loss_ignores_independent_noise_pairs():
    model = build_tiny_model(transition=True)
    loss_fn = SILoss(
        trajectory_factorization=True,
        projection=False,
        factor_pair_cross_noise_prob=1.0,
        factor_min_delta_t=0.2,
        factor_max_delta_t=0.4,
        factor_transition=True,
    )
    losses = loss_fn(
        model,
        torch.randn(2, 4, 8, 8),
        model_kwargs={"y": torch.randint(0, 10, (2,))},
    )
    assert losses["factor_transition_loss"].item() == 0.0


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


class RecordingFactorModel:
    def __init__(self, model):
        self.model = model

    def __call__(self, x, t, **kwargs):
        self.inputs = x.detach()
        self.timesteps = t.detach()
        self.kwargs = kwargs
        return self.model(x, t, **kwargs)


def test_orthogonal_orbit_changes_exactly_one_nuisance_per_pair(monkeypatch):
    monkeypatch.setattr(
        torch, "randperm", lambda size, device=None: torch.arange(size, device=device)
    )
    torch.manual_seed(23)
    pair_count = 8
    images = torch.randn(pair_count, 4, 8, 8)
    model = RecordingFactorModel(build_tiny_model())
    losses = SILoss(
        trajectory_factorization=True,
        projection=False,
        factor_orbit_mode="orthogonal",
        factor_orbit_noise_only_prob=0.5,
        factor_min_delta_t=0.2,
        factor_max_delta_t=0.4,
    )(
        model,
        images,
        model_kwargs={
            "y": torch.randint(0, 10, (pair_count,)),
            "force_drop_ids": torch.arange(pair_count) % 2 == 0,
        },
    )

    time_a = model.timesteps[:pair_count].reshape(-1, 1, 1, 1)
    time_b = model.timesteps[pair_count:].reshape(-1, 1, 1, 1)
    scaled_noise_a = (
        model.inputs[:pair_count] - (1.0 - time_a) * images
    ) * time_b
    scaled_noise_b = (
        model.inputs[pair_count:] - (1.0 - time_b) * images
    ) * time_a
    time_changed = (time_a - time_b).abs().flatten() > 1e-6
    noise_changed = torch.tensor([
        not torch.allclose(
            scaled_noise_a[i], scaled_noise_b[i], atol=2e-5, rtol=2e-5
        )
        for i in range(pair_count)
    ], device=time_changed.device)

    assert torch.all(torch.logical_xor(time_changed, noise_changed))
    assert losses["factor_joint_intervention_fraction"].item() == 0.0
    if losses["factor_time_only_fraction"].item() > 0:
        assert 0.2 <= losses["factor_time_intervention_delta"].item() <= 0.4
    assert torch.allclose(
        losses["factor_time_only_fraction"]
        + losses["factor_noise_only_fraction"],
        torch.tensor(1.0),
    )
    assert torch.equal(
        model.kwargs["force_drop_ids"][:pair_count],
        model.kwargs["force_drop_ids"][pair_count:],
    )


def test_reliable_target_selects_stable_discriminative_channels():
    loss_fn = SILoss(
        trajectory_factorization=True,
        projection=False,
        factor_reliable_target=True,
        factor_reliability_keep_ratio=0.5,
    )
    # Channel 0 varies by source and is identical across views. Channel 1 has
    # no between-source signal and flips across the orbit.
    stable = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]]).unsqueeze(1)
    nuisance_a = torch.ones_like(stable)
    nuisance_b = -nuisance_a
    target_a = torch.cat([stable, nuisance_a], dim=-1)
    target_b = torch.cat([stable, nuisance_b], dim=-1)
    target = torch.cat([target_a, target_b], dim=0)
    zeros = torch.zeros(8, 1, 2)
    factorization = {
        "persistent": zeros.clone(),
        "evolving": zeros.clone(),
        "persistent_component": zeros.clone(),
        "evolving_component": zeros.clone(),
        "target": target,
        "recomposed": zeros.clone(),
        "wrong_evolving_recomposed": zeros.clone(),
    }

    losses = loss_fn._factorization_losses(factorization)
    assert torch.allclose(
        losses["factor_target_reliability"], torch.tensor(0.5), atol=1e-5
    )
    assert losses["factor_target_selected_fraction"].item() == 0.5
    assert torch.allclose(
        losses["factor_target_gate_mean"], torch.tensor(0.5), atol=1e-5
    )
    assert torch.allclose(
        losses["common_energy_fraction"]
        + losses["residual_energy_fraction"],
        torch.tensor(1.0),
        atol=1e-5,
    )


def test_velocity_recomposition_is_training_only_and_backpropagates():
    torch.manual_seed(29)
    model = build_tiny_model(velocity_recomposition=True)
    loss_fn = SILoss(
        trajectory_factorization=True,
        projection=False,
        factor_orbit_mode="orthogonal",
        factor_min_delta_t=0.2,
        factor_max_delta_t=0.4,
    )
    losses = loss_fn(
        model,
        torch.randn(4, 4, 8, 8),
        model_kwargs={"y": torch.randint(0, 10, (4,))},
    )
    losses["factor_velocity_recom_loss"].mean().backward()

    head = model.factorization_head
    assert losses["factor_velocity_recom_loss"].shape == (4,)
    assert head.persistent_velocity_decoder[-1].weight.grad is not None
    assert head.evolving_velocity_decoder[-1].weight.grad is not None
    assert head.persistent_projector[-1].weight.grad is not None
    assert head.evolving_projector[-1].weight.grad is not None

    inference = model(
        torch.randn(2, 4, 8, 8),
        torch.rand(2),
        torch.randint(0, 10, (2,)),
    )
    assert "factorization" not in inference


def test_velocity_decoder_does_not_shift_shared_initialization():
    torch.manual_seed(31)
    control = build_tiny_model(velocity_recomposition=False)
    control_next_random = torch.rand(8)
    torch.manual_seed(31)
    treatment = build_tiny_model(velocity_recomposition=True)
    treatment_next_random = torch.rand(8)

    treatment_state = treatment.state_dict()
    for name, value in control.state_dict().items():
        assert torch.equal(value, treatment_state[name]), name
    assert torch.equal(control_next_random, treatment_next_random)


def test_gradient_reversal_preserves_forward_and_scales_backward():
    inputs = torch.tensor([1.0, -2.0], requires_grad=True)
    reversed_inputs = gradient_reverse(inputs, scale=0.25)
    assert torch.equal(reversed_inputs, inputs)
    reversed_inputs.sum().backward()
    assert torch.allclose(inputs.grad, torch.full_like(inputs, -0.25))


def test_adversarial_nuisance_losses_train_all_heads_and_factors():
    torch.manual_seed(37)
    model = build_tiny_model(
        velocity_recomposition=True, adversarial=True
    )
    loss_fn = SILoss(
        trajectory_factorization=True,
        projection=False,
        factor_orbit_mode="orthogonal",
        factor_orbit_noise_only_prob=0.5,
        factor_min_delta_t=0.2,
        factor_max_delta_t=0.4,
        factor_adversarial=True,
        factor_adversarial_timestep_bins=4,
    )
    losses = loss_fn(
        model,
        torch.randn(4, 4, 8, 8),
        model_kwargs={"y": torch.randint(0, 10, (4,))},
        factor_adversarial_grl_scale=0.2,
    )
    objective_names = (
        "factor_adv_persistent_time_loss",
        "factor_adv_persistent_orbit_loss",
        "factor_probe_evolving_time_loss",
        "factor_probe_evolving_orbit_loss",
    )
    sum(losses[name].mean() for name in objective_names).backward()

    head = model.factorization_head
    assert losses["factor_adv_persistent_time_loss"].shape == (8,)
    assert losses["factor_adv_persistent_orbit_loss"].shape == (4,)
    assert head.persistent_time_discriminator[-1].weight.grad is not None
    assert head.persistent_orbit_discriminator[-1].weight.grad is not None
    assert head.evolving_time_probe[-1].weight.grad is not None
    assert head.evolving_orbit_probe[-1].weight.grad is not None
    assert head.persistent_projector[-1].weight.grad is not None
    assert head.evolving_projector[-1].weight.grad is not None
    assert torch.isfinite(losses["factor_time_separation_gap"])
    assert torch.isfinite(losses["factor_orbit_separation_gap"])


def test_zero_grl_is_a_true_critic_only_control():
    torch.manual_seed(41)
    head = build_tiny_model(adversarial=True).factorization_head
    persistent = torch.randn(4, 16, 16, requires_grad=True)
    evolving = torch.randn(4, 16, 16, requires_grad=True)
    predictions = head.predict_nuisance(
        persistent, evolving, grl_scale=0.0
    )
    loss = F.cross_entropy(
        predictions["persistent_time_logits"], torch.tensor([0, 1, 2, 3])
    ) + F.cross_entropy(
        predictions["persistent_orbit_logits"], torch.tensor([0, 1])
    )
    loss.backward()

    assert torch.count_nonzero(persistent.grad).item() == 0
    assert evolving.grad is None
    assert head.persistent_time_discriminator[-1].weight.grad is not None
    assert head.persistent_orbit_discriminator[-1].weight.grad is not None


def test_adversarial_heads_do_not_shift_shared_initialization():
    torch.manual_seed(43)
    control = build_tiny_model(adversarial=False)
    control_next_random = torch.rand(8)
    torch.manual_seed(43)
    treatment = build_tiny_model(adversarial=True)
    treatment_next_random = torch.rand(8)

    treatment_state = treatment.state_dict()
    for name, value in control.state_dict().items():
        assert torch.equal(value, treatment_state[name]), name
    assert torch.equal(control_next_random, treatment_next_random)


def test_adversarial_loss_rejects_non_orthogonal_orbits():
    try:
        SILoss(
            trajectory_factorization=True,
            projection=False,
            factor_adversarial=True,
            factor_orbit_mode="legacy",
        )
    except ValueError as error:
        assert "orthogonal orbit" in str(error)
    else:
        raise AssertionError("adversarial loss accepted an ambiguous orbit")


def test_block_diversity_features_are_not_retained_during_inference():
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
        block_diversity_loss=True,
        fused_attn=False,
        qk_norm=False,
    )
    inputs = torch.randn(2, 4, 8, 8)
    timesteps = torch.rand(2)
    labels = torch.randint(0, 10, (2,))

    model.train()
    assert "block_feas" in model(inputs, timesteps, labels)
    model.eval()
    assert "block_feas" not in model(inputs, timesteps, labels)
