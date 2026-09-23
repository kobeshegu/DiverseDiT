import torch
import torch.nn.functional as F

from loss import SILoss
from models.sit import SiT, gradient_reverse


def build_tiny_model(
    transition=False,
    velocity_recomposition=False,
    native_parameterization=False,
    semantic_conditioning=False,
    semantic_injection_scale=1.0,
    semantic_targets=False,
    adversarial=False,
    selective_invariance=False,
    pair_interaction=False,
    pair_byol_alignment=False,
    path_type="linear",
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
        z_dims=[24] if semantic_conditioning or semantic_targets else [],
        path_type=path_type,
        trajectory_factorization=True,
        factor_dim=16,
        factor_projector_dim=32,
        factor_source_depth=2,
        factor_target_depth=4,
        factor_transition=transition,
        factor_velocity_recomposition=velocity_recomposition,
        factor_native_parameterization=native_parameterization,
        factor_semantic_conditioning=semantic_conditioning,
        factor_semantic_injection_scale=semantic_injection_scale,
        factor_adversarial=adversarial,
        factor_adversarial_timestep_bins=4,
        factor_selective_invariance=selective_invariance,
        factor_selective_dim=8,
        factor_selective_source_depth=2,
        factor_pair_interaction=pair_interaction,
        factor_pair_interaction_depth=2,
        factor_pair_interaction_hidden_ratio=0.25,
        factor_pair_interaction_self_prob=0.25,
        factor_pair_byol_alignment=pair_byol_alignment,
        factor_pair_alignment_dim=8,
        factor_pair_alignment_predictor_dim=16,
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


def test_pair_interaction_supports_paired_and_single_forward():
    model = build_tiny_model(pair_interaction=True)
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
    assert "factor_pair_interaction_rms" in losses

    inference_output = model(
        torch.randn(2, 4, 8, 8),
        torch.rand(2),
        torch.randint(0, 10, (2,)),
    )
    assert inference_output["x"].shape == (2, 4, 8, 8)


def test_random_pair_alignment_loss_is_finite():
    model = build_tiny_model()
    loss_fn = SILoss(
        trajectory_factorization=True,
        projection=False,
        factor_min_delta_t=0.2,
        factor_max_delta_t=0.4,
        factor_pair_random_align=True,
        factor_pair_random_align_dim=8,
    )
    losses = loss_fn(
        model,
        torch.randn(4, 4, 8, 8),
        model_kwargs={"y": torch.randint(0, 10, (4,))},
        factor_batch_ratio=0.5,
    )
    assert torch.isfinite(losses["factor_pair_random_align_loss"]).all()
    assert torch.isfinite(losses["factor_pair_random_variance_loss"]).all()


def test_byol_pair_alignment_loss_backpropagates_to_head():
    model = build_tiny_model(pair_byol_alignment=True)
    loss_fn = SILoss(
        trajectory_factorization=True,
        projection=False,
        factor_min_delta_t=0.2,
        factor_max_delta_t=0.4,
        factor_pair_byol_align=True,
    )
    losses = loss_fn(
        model,
        torch.randn(4, 4, 8, 8),
        model_kwargs={"y": torch.randint(0, 10, (4,))},
        factor_batch_ratio=0.5,
    )
    total = (
        losses["denoising_loss"].mean()
        + losses["factor_pair_byol_align_loss"].mean()
        + losses["factor_pair_byol_variance_loss"].mean()
    )
    total.backward()
    assert torch.isfinite(losses["factor_pair_byol_align_loss"]).all()
    assert model.pair_alignment_head.predictor[-1].weight.grad is not None


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


def test_antithetic_orbit_has_shared_time_and_opposite_noise(monkeypatch):
    torch.manual_seed(33)
    pair_count = 4
    images = torch.randn(pair_count, 4, 8, 8)
    model = RecordingFactorModel(build_tiny_model())
    loss_fn = SILoss(
        trajectory_factorization=True,
        projection=False,
        factor_orbit_mode="antithetic",
    )
    monkeypatch.setattr(
        loss_fn,
        "_sample_times",
        lambda batch: torch.full(
            (batch.shape[0], 1, 1, 1),
            0.4,
            device=batch.device,
            dtype=batch.dtype,
        ),
    )
    losses = loss_fn(
        model,
        images,
        model_kwargs={"y": torch.randint(0, 10, (pair_count,))},
    )

    time_a = model.timesteps[:pair_count]
    time_b = model.timesteps[pair_count:]
    alpha = (1.0 - time_a).reshape(-1, 1, 1, 1)
    sigma = time_a.reshape(-1, 1, 1, 1)
    noise_a = (model.inputs[:pair_count] - alpha * images) / sigma
    noise_b = (model.inputs[pair_count:] - alpha * images) / sigma
    assert torch.equal(time_a, time_b)
    assert torch.allclose(noise_a, -noise_b, atol=2e-5, rtol=2e-5)
    assert losses["mean_delta_t"].item() == 0.0
    assert losses["factor_noise_only_fraction"].item() == 1.0
    assert losses["factor_joint_intervention_fraction"].item() == 0.0


def test_native_parameterization_recomposes_main_velocity():
    torch.manual_seed(35)
    inputs = torch.randn(4, 4, 8, 8)
    times = torch.rand(4)
    labels = torch.randint(0, 10, (4,))
    for path_type in ("linear", "cosine"):
        model = build_tiny_model(
            native_parameterization=True, path_type=path_type
        )
        output = model(inputs, times, labels)
        native = output["native_parameterization"]
        if path_type == "linear":
            d_alpha = -torch.ones_like(times)
            d_sigma = torch.ones_like(times)
        else:
            angle = times * (torch.pi / 2)
            d_alpha = -(torch.pi / 2) * torch.sin(angle)
            d_sigma = (torch.pi / 2) * torch.cos(angle)
        expected = (
            d_alpha.reshape(-1, 1, 1, 1) * native["source"]
            + d_sigma.reshape(-1, 1, 1, 1) * native["noise"]
        )
        assert output["x"].shape == inputs.shape
        assert native["source"].shape == inputs.shape
        assert native["noise"].shape == inputs.shape
        assert native["base_velocity"].shape == inputs.shape
        assert torch.allclose(output["x"], expected)
        assert "factorization" not in output


def test_native_losses_train_source_noise_and_base_heads():
    torch.manual_seed(39)
    model = build_tiny_model(native_parameterization=True)
    losses = SILoss(
        trajectory_factorization=True,
        projection=False,
        factor_orbit_mode="antithetic",
        factor_native_parameterization=True,
    )(
        model,
        torch.randn(4, 4, 8, 8),
        model_kwargs={"y": torch.randint(0, 10, (4,))},
    )
    objective = (
        losses["denoising_loss"].mean()
        + losses["factor_native_source_loss"].mean()
        + losses["factor_native_noise_loss"].mean()
        + losses["factor_native_antithetic_loss"].mean()
        + 0.1 * losses["factor_native_base_loss"].mean()
    )
    objective.backward()

    head = model.factorization_head
    assert losses["factor_native_source_loss"].shape == (4,)
    assert losses["factor_native_noise_loss"].shape == (4,)
    assert losses["factor_native_base_loss"].shape == (4,)
    assert losses["factor_native_antithetic_loss"].shape == (4,)
    assert head.native_source_decoder[-1].weight.grad is not None
    assert head.native_noise_decoder[-1].weight.grad is not None
    assert model.final_layer.linear.weight.grad is not None
    assert next(model.blocks[-1].parameters()).grad is not None
    assert torch.isfinite(losses["factor_native_source_pair_gap"])
    assert torch.isfinite(losses["factor_native_noise_antisymmetry_error"])


def test_native_heads_do_not_shift_historical_initialization():
    torch.manual_seed(40)
    control = build_tiny_model(native_parameterization=False)
    control_next_random = torch.rand(8)
    torch.manual_seed(40)
    treatment = build_tiny_model(native_parameterization=True)
    treatment_next_random = torch.rand(8)

    treatment_state = treatment.state_dict()
    for name, value in control.state_dict().items():
        assert torch.equal(value, treatment_state[name]), name
    assert torch.equal(control_next_random, treatment_next_random)


def test_semantic_source_alignment_and_film_train_end_to_end():
    torch.manual_seed(41)
    model = build_tiny_model(semantic_conditioning=True)
    # The stock zero output head intentionally blocks backbone gradients on the
    # very first SiT step.  Give it a trained-like readout to test the complete
    # source-FiLM-to-velocity gradient path.
    torch.nn.init.normal_(model.final_layer.linear.weight, std=0.02)
    images = torch.randn(4, 4, 8, 8)
    clean_targets = [torch.randn(4, 16, 24)]
    losses = SILoss(
        trajectory_factorization=True,
        projection=False,
        factor_semantic_conditioning=True,
    )(
        model,
        images,
        model_kwargs={"y": torch.randint(0, 10, (4,))},
        zs=clean_targets,
    )
    objective = (
        losses["denoising_loss"].mean()
        + 0.5 * losses["factor_semantic_repa_loss"]
        + 0.005 * losses["factor_semantic_decorrelation_loss"]
    )
    objective.backward()

    head = model.factorization_head
    assert losses["factor_semantic_source_consistency_loss"].shape == (4,)
    assert torch.isfinite(losses["factor_semantic_repa_loss"])
    assert torch.isfinite(losses["factor_semantic_modulation_rms"])
    assert model.projectors[0][-1].weight.grad is not None
    assert head.persistent_projector[-1].weight.grad is not None
    assert head.semantic_source_gate.grad is not None


def test_semantic_injection_can_be_disabled_without_changing_checkpoint_shape():
    torch.manual_seed(42)
    enabled = build_tiny_model(
        semantic_conditioning=True, semantic_injection_scale=1.0
    )
    torch.manual_seed(42)
    disabled = build_tiny_model(
        semantic_conditioning=True, semantic_injection_scale=0.0
    )
    disabled.load_state_dict(enabled.state_dict())
    with torch.no_grad():
        enabled.factorization_head.semantic_source_gate.fill_(0.1)
        disabled.load_state_dict(enabled.state_dict())
        torch.nn.init.normal_(enabled.final_layer.linear.weight, std=0.02)
        disabled.final_layer.linear.weight.copy_(enabled.final_layer.linear.weight)

    inputs = torch.randn(2, 4, 8, 8)
    times = torch.rand(2)
    labels = torch.randint(0, 10, (2,))
    enabled_output = enabled(inputs, times, labels)["x"]
    disabled_output = disabled(inputs, times, labels)["x"]
    assert enabled.state_dict().keys() == disabled.state_dict().keys()
    assert not torch.allclose(enabled_output, disabled_output)


def test_semantic_modules_do_not_shift_shared_initialization_or_rng():
    torch.manual_seed(43)
    control = build_tiny_model(semantic_targets=True)
    control_next_random = torch.rand(8)
    torch.manual_seed(43)
    treatment = build_tiny_model(semantic_conditioning=True)
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


def test_clean_recovery_is_exact_for_linear_and_cosine_paths():
    torch.manual_seed(47)
    clean = torch.randn(3, 4, 8, 8)
    noise = torch.randn_like(clean)
    time = torch.rand(3, 1, 1, 1) * 0.9 + 0.05
    for path_type in ("linear", "cosine"):
        loss_fn = SILoss(path_type=path_type, projection=False)
        alpha, sigma, d_alpha, d_sigma = loss_fn.interpolant(time)
        noisy = alpha * clean + sigma * noise
        velocity = d_alpha * clean + d_sigma * noise
        recovered = loss_fn._recover_clean_from_velocity(
            noisy, time, velocity
        )
        assert torch.allclose(recovered, clean, atol=2e-5, rtol=2e-5)


def test_clean_consensus_uses_main_velocity_and_backpropagates():
    torch.manual_seed(53)
    model = build_tiny_model()
    losses = SILoss(
        trajectory_factorization=True,
        projection=False,
        factor_clean_consensus=True,
        factor_min_delta_t=0.2,
        factor_max_delta_t=0.4,
    )(
        model,
        torch.randn(4, 4, 8, 8),
        model_kwargs={"y": torch.randint(0, 10, (4,))},
    )
    losses["factor_clean_consensus_loss"].mean().backward()

    assert losses["factor_clean_consensus_loss"].shape == (4,)
    assert torch.isfinite(losses["factor_clean_pair_gap"])
    assert torch.isfinite(losses["factor_clean_source_error"])
    assert model.final_layer.linear.weight.grad is not None
    assert torch.count_nonzero(model.final_layer.linear.weight.grad).item() > 0


def test_selective_forward_exposes_low_rank_features_and_full_source():
    model = build_tiny_model(selective_invariance=True)
    pair_count = 2
    total_count = 5
    output = model(
        torch.randn(total_count, 4, 8, 8),
        torch.rand(total_count),
        torch.randint(0, 10, (total_count,)),
        trajectory_pair=True,
        factor_pair_count=pair_count,
        return_factorization=True,
        return_selective_invariance=True,
    )
    selective = output["selective_invariance"]
    assert selective["features"].shape == (2 * pair_count, 16, 8)
    assert selective["source_features"].shape == (total_count, 16, 64)
    assert selective["basis"].shape == (8, 64)
    assert selective["pair_count"] == pair_count


def test_task_selective_loss_backpropagates_without_second_order_graph():
    torch.manual_seed(59)
    model = build_tiny_model(selective_invariance=True)
    losses = SILoss(
        trajectory_factorization=True,
        projection=False,
        factor_clean_consensus=True,
        factor_selective_invariance=True,
        factor_selective_weighting="task",
        factor_min_delta_t=0.2,
        factor_max_delta_t=0.4,
    )(
        model,
        torch.randn(4, 4, 8, 8),
        model_kwargs={"y": torch.randint(0, 10, (4,))},
    )
    objective = (
        losses["denoising_loss"].mean()
        + 0.05 * losses["factor_clean_consensus_loss"].mean()
        + 0.1 * losses["factor_selective_loss"]
        + 0.01 * losses["factor_selective_orth_loss"]
        + 0.02 * losses["factor_selective_variance_loss"]
    )
    objective.backward()

    projector = model.factor_selective_invariance_head.projector
    assert projector.weight.grad is not None
    assert model.x_embedder.proj.weight.grad is not None
    assert torch.isfinite(losses["factor_selective_loss"])
    assert torch.isfinite(losses["factor_selective_source_ratio"])
    assert 1 <= losses["factor_selective_effective_dims"].item() <= 8.01


def test_selective_head_does_not_shift_historical_initialization():
    torch.manual_seed(61)
    control = build_tiny_model(selective_invariance=False)
    control_next_random = torch.rand(8)
    torch.manual_seed(61)
    treatment = build_tiny_model(selective_invariance=True)
    treatment_next_random = torch.rand(8)

    treatment_state = treatment.state_dict()
    for name, value in control.state_dict().items():
        assert torch.equal(value, treatment_state[name]), name
    assert torch.equal(control_next_random, treatment_next_random)
