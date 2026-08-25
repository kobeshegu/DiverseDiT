import torch

from loss import SILoss
from models.sit import SiT


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
