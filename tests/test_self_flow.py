import torch

from loss import SILoss, noise_gap_weighted_cosine_loss
from models.sit import SiT
from models.trajectory import ContextualDepthJEPAPredictor


def tiny_sit(z_dims=None, encoder_depth=1):
    return SiT(
        input_size=4,
        patch_size=2,
        in_channels=4,
        hidden_size=32,
        decoder_hidden_size=32,
        depth=2,
        encoder_depth=encoder_depth,
        num_heads=4,
        num_classes=10,
        z_dims=[] if z_dims is None else z_dims,
        projector_dim=64,
        fused_attn=False,
        qk_norm=False,
    )


def test_sit_accepts_scalar_and_token_timesteps():
    model = tiny_sit()
    images = torch.randn(2, 4, 4, 4)
    labels = torch.tensor([1, 2])

    scalar = model(images, torch.rand(2), labels)["x"]
    token = model(images, torch.rand(2, 4), labels)["x"]

    assert scalar.shape == images.shape
    assert token.shape == images.shape


def test_self_flow_loss_is_finite_and_backpropagates():
    model = tiny_sit(z_dims=[32])
    teacher = tiny_sit(z_dims=[32])
    teacher.load_state_dict(model.state_dict())
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    loss_fn = SILoss(
        self_flow=True,
        self_flow_mask_ratio=0.25,
        self_flow_teacher_depth=2,
    )
    images = torch.randn(2, 4, 4, 4)
    losses = loss_fn(
        model,
        images,
        {"y": torch.tensor([1, 2])},
        teacher_model=teacher,
    )
    total = losses["denoising_loss"].mean() + 0.8 * losses[
        "self_flow_rep_loss"
    ].mean()
    total.backward()

    assert torch.isfinite(total)
    assert model.projectors[0][0].weight.grad is not None
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert 0 <= losses["self_flow_mask_fraction"] <= 1


def test_noise_gap_loss_ignores_clean_tokens_and_weights_hard_tokens():
    student = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]])
    teacher = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]])
    noise_gap = torch.tensor([[0.0, 0.25, 0.75]])

    loss, metrics = noise_gap_weighted_cosine_loss(
        student, teacher, noise_gap
    )

    assert torch.allclose(loss, torch.tensor(3.5))
    assert torch.allclose(metrics["active_fraction"], torch.tensor(2 / 3))
    assert torch.allclose(metrics["mean_gap"], torch.tensor(0.5))


def test_contextual_self_flow_reuses_main_forward_for_multi_depth_loss():
    torch.manual_seed(0)
    model = tiny_sit()
    teacher = tiny_sit()
    teacher.load_state_dict(model.state_dict())
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    predictor = ContextualDepthJEPAPredictor(
        in_dim=32,
        depth_pairs=[(1, 2)],
        hidden_dim=32,
    )
    loss_fn = SILoss(
        self_flow=True,
        self_flow_mask_ratio=0.25,
        self_flow_contextual=True,
        self_flow_depth_pairs=[(1, 2)],
        self_flow_pair_weights=[1.0],
    )

    losses = loss_fn(
        model,
        torch.randn(4, 4, 4, 4),
        {"y": torch.tensor([1, 2, 3, 4])},
        teacher_model=teacher,
        contextual_predictor=predictor,
    )
    total = losses["denoising_loss"].mean() + 0.8 * losses[
        "self_flow_rep_loss"
    ]
    total.backward()

    assert torch.isfinite(total)
    assert losses["self_flow_hard_fraction"] > 0
    assert predictor.predictors["1_to_2"].predictor[1].weight.grad is not None
    assert all(parameter.grad is None for parameter in teacher.parameters())
