# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import torch.nn.functional as F


class _GradientReverse(torch.autograd.Function):
    """Identity in the forward pass and sign-reversed in the backward pass."""

    @staticmethod
    def forward(ctx, inputs, scale):
        ctx.scale = float(scale)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.scale * grad_output, None


def gradient_reverse(inputs, scale=1.0):
    """Reverse only the representation gradient, not discriminator updates."""
    return _GradientReverse.apply(inputs, scale)


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )


class TrajectoryFactorizationHead(nn.Module):
    """Factorize a trajectory feature through balanced additive recomposition.

    The two latents are not assigned hand-designed semantic meanings.  Separate
    decoders make their usage observable: the persistent component predicts the
    pair-symmetric target, while the evolving component predicts the view
    residual.  Their sum reconstructs a deeper feature after swapping the
    persistent code across trajectory views.
    """

    def __init__(self, hidden_size, factor_dim=256, projector_dim=1024,
                 predict_transition=False):
        super().__init__()
        self.factor_dim = factor_dim
        self.projector_dim = projector_dim
        self.persistent_projector = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, factor_dim),
        )
        self.evolving_projector = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, factor_dim),
        )
        self.persistent_decoder = nn.Sequential(
            nn.LayerNorm(factor_dim),
            nn.Linear(factor_dim, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, hidden_size),
        )
        self.evolving_decoder = nn.Sequential(
            nn.LayerNorm(factor_dim),
            nn.Linear(factor_dim, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, hidden_size),
        )
        self.predict_transition = predict_transition
        if predict_transition:
            self.delta_embedder = TimestepEmbedder(factor_dim)
            self.transition_predictor = nn.Sequential(
                nn.LayerNorm(2 * factor_dim),
                nn.Linear(2 * factor_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, factor_dim),
            )
        self.predict_velocity_recomposition = False
        self.predict_native_parameterization = False
        self.predict_semantic_conditioning = False
        self.predict_adversarial_nuisance = False

    def factorize(self, features):
        return (
            self.persistent_projector(features),
            self.evolving_projector(features),
        )

    def decode(self, persistent, evolving):
        """Return observable branch contributions in the target feature space."""
        return (
            self.persistent_decoder(persistent),
            self.evolving_decoder(evolving),
        )

    def recompose(self, persistent, evolving):
        persistent_component, evolving_component = self.decode(
            persistent, evolving
        )
        return persistent_component + evolving_component

    def transition(self, evolving, delta_t):
        if not self.predict_transition:
            raise RuntimeError("Trajectory transition predictor is disabled")
        delta_embedding = self.delta_embedder(delta_t).unsqueeze(1)
        delta_embedding = delta_embedding.expand(-1, evolving.shape[1], -1)
        return self.transition_predictor(
            torch.cat([evolving, delta_embedding], dim=-1)
        )

    def enable_velocity_recomposition(self, output_dim):
        """Add a training-only task decoder without changing legacy heads."""
        if self.predict_velocity_recomposition:
            return
        self.velocity_time_embedder = TimestepEmbedder(self.factor_dim)
        self.persistent_velocity_decoder = nn.Sequential(
            nn.LayerNorm(self.factor_dim),
            nn.Linear(self.factor_dim, self.projector_dim),
            nn.SiLU(),
            nn.Linear(self.projector_dim, output_dim),
        )
        self.evolving_velocity_decoder = nn.Sequential(
            nn.LayerNorm(2 * self.factor_dim),
            nn.Linear(2 * self.factor_dim, self.projector_dim),
            nn.SiLU(),
            nn.Linear(self.projector_dim, output_dim),
        )
        self.predict_velocity_recomposition = True

    def predict_velocity(self, persistent, evolving, timestep):
        """Predict a view target from cross-view persistent/current evolving."""
        if not self.predict_velocity_recomposition:
            raise RuntimeError("Velocity recomposition decoder is disabled")
        time_embedding = self.velocity_time_embedder(timestep).unsqueeze(1)
        time_embedding = time_embedding.expand(-1, evolving.shape[1], -1)
        return (
            self.persistent_velocity_decoder(persistent),
            self.evolving_velocity_decoder(torch.cat([
                evolving, time_embedding
            ], dim=-1)),
        )

    def enable_native_parameterization(self, output_dim):
        """Decode clean-source and noise fields for path-aware velocity."""
        if self.predict_native_parameterization:
            return
        self.native_source_decoder = nn.Sequential(
            nn.LayerNorm(self.factor_dim),
            nn.Linear(self.factor_dim, self.projector_dim),
            nn.SiLU(),
            nn.Linear(self.projector_dim, output_dim),
        )
        self.native_noise_decoder = nn.Sequential(
            nn.LayerNorm(self.factor_dim),
            nn.Linear(self.factor_dim, self.projector_dim),
            nn.SiLU(),
            nn.Linear(self.projector_dim, output_dim),
        )
        self.predict_native_parameterization = True

    def predict_native_components(self, persistent, evolving):
        """Predict x0 and epsilon from the source/evolving factors."""
        if not self.predict_native_parameterization:
            raise RuntimeError("Native parameterization decoder is disabled")
        return (
            self.native_source_decoder(persistent),
            self.native_noise_decoder(evolving),
        )

    def enable_semantic_conditioning(self, hidden_size):
        """Add a source-only semantic readout and late-backbone FiLM path."""
        if self.predict_semantic_conditioning:
            return
        self.semantic_source_gate = nn.Parameter(
            torch.zeros(hidden_size)
        )
        self.predict_semantic_conditioning = True

    def predict_semantic_source(self, source_features, projectors):
        """Project spatial source tokens to clean semantic target spaces."""
        if not self.predict_semantic_conditioning:
            raise RuntimeError("Semantic source conditioning is disabled")
        return [projector(source_features) for projector in projectors]

    def semantic_modulation(self, persistent):
        """Return a channel-gated spatial source shift for the state stream."""
        if not self.predict_semantic_conditioning:
            raise RuntimeError("Semantic source conditioning is disabled")
        source_features = self.persistent_decoder(persistent)
        gate = torch.tanh(self.semantic_source_gate).view(1, 1, -1)
        return source_features, gate * source_features

    def enable_adversarial_nuisance(self, timestep_bins=8):
        """Add persistent adversaries and matched evolving nuisance probes."""
        if self.predict_adversarial_nuisance:
            return
        if timestep_bins < 2:
            raise ValueError("adversarial timestep bins must be at least 2")
        self.adversarial_timestep_bins = timestep_bins

        def classifier(input_dim, output_dim):
            return nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, self.projector_dim),
                nn.SiLU(),
                nn.Linear(self.projector_dim, output_dim),
            )

        self.persistent_time_discriminator = classifier(
            self.factor_dim, timestep_bins
        )
        self.persistent_orbit_discriminator = classifier(
            2 * self.factor_dim, 2
        )
        self.evolving_time_probe = classifier(self.factor_dim, timestep_bins)
        self.evolving_orbit_probe = classifier(2 * self.factor_dim, 2)
        self.predict_adversarial_nuisance = True

    @staticmethod
    def _pair_signature(pooled_features):
        if pooled_features.shape[0] % 2 != 0:
            raise ValueError("paired nuisance prediction needs an even batch")
        view_a, view_b = pooled_features.chunk(2, dim=0)
        return torch.cat([
            (view_a - view_b).abs(), view_a * view_b
        ], dim=-1)

    def predict_nuisance(self, persistent, evolving, grl_scale=1.0):
        """Predict nuisances while purifying only the persistent code.

        The evolving probes receive ordinary gradients so nuisance information
        is relocated instead of merely removed from the shared representation.
        """
        if not self.predict_adversarial_nuisance:
            raise RuntimeError("Adversarial nuisance heads are disabled")
        persistent_pooled = persistent.mean(dim=1)
        evolving_pooled = evolving.mean(dim=1)
        persistent_pair = self._pair_signature(persistent_pooled)
        evolving_pair = self._pair_signature(evolving_pooled)
        return {
            'persistent_time_logits': self.persistent_time_discriminator(
                gradient_reverse(persistent_pooled, grl_scale)
            ),
            'persistent_orbit_logits': self.persistent_orbit_discriminator(
                gradient_reverse(persistent_pair, grl_scale)
            ),
            'evolving_time_logits': self.evolving_time_probe(evolving_pooled),
            'evolving_orbit_logits': self.evolving_orbit_probe(evolving_pair),
        }


class TrajectoryInvariantProjector(nn.Module):
    """Read out a low-dimensional trajectory-invariant representation.

    ``linear`` is the paper-facing default: its normalized rows span an actual
    subspace of the backbone feature space.  ``mlp`` is retained only as an
    ablation that tests whether an expressive readout can absorb the objective.
    Neither readout feeds the denoising head.
    """

    def __init__(self, hidden_size, invariant_dim=256, projector_dim=1024,
                 projector_type="linear"):
        super().__init__()
        if projector_type not in {"linear", "mlp"}:
            raise ValueError("invariant projector type must be 'linear' or 'mlp'")
        if invariant_dim <= 0 or projector_dim <= 0:
            raise ValueError("invariant projector dimensions must be positive")
        if projector_type == "linear" and invariant_dim > hidden_size:
            raise ValueError(
                "linear invariant_dim cannot exceed the backbone hidden size"
            )
        self.projector_type = projector_type
        if projector_type == "linear":
            self.projector = nn.Linear(hidden_size, invariant_dim, bias=False)
        else:
            self.projector = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, invariant_dim),
            )

    def normalized_basis(self):
        if self.projector_type != "linear":
            raise RuntimeError("only the linear projector has a subspace basis")
        return F.normalize(self.projector.weight.float(), dim=-1)

    def orthogonality_loss(self):
        if self.projector_type != "linear":
            reference = next(self.parameters())
            return reference.new_zeros(())
        basis = self.normalized_basis()
        gram = basis @ basis.T
        identity = torch.eye(
            gram.shape[0], device=gram.device, dtype=gram.dtype
        )
        return (gram - identity).square().mean()

    @torch.no_grad()
    def orthonormal_basis(self):
        """Return a rank-aware orthonormal basis for analysis projections."""
        if self.projector_type != "linear":
            raise RuntimeError("only the linear projector has a subspace basis")
        weight = self.projector.weight.float()
        _, singular_values, right_vectors = torch.linalg.svd(
            weight, full_matrices=False
        )
        tolerance = (
            max(weight.shape)
            * torch.finfo(weight.dtype).eps
            * singular_values.max()
        )
        rank = int((singular_values > tolerance).sum().item())
        return right_vectors[:rank].T

    def forward(self, features):
        if self.projector_type == "linear":
            basis = self.normalized_basis().to(dtype=features.dtype)
            return F.linear(features, basis)
        return self.projector(features)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
# Added for cross skip connections, motivated by UViT, modified from SkipDiT    #
#################################################################################  
class ResidualLinear(nn.Module):
    def __init__(self, hidden_states_size):
        super(ResidualLinear, self).__init__()
        self.linear = nn.Linear(2*hidden_states_size, hidden_states_size)
        # self.initialize_weights()

    def forward(self, cat, hidden_states):
        x = self.linear(cat)
        output = x
        return output

class FP32_Layernorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(inputs.float(), self.normalized_shape, self.weight.float(), self.bias.float(),self.eps).to(origin_dtype)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################            
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        self.timestep_embedding = self.positional_embedding
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"]
            )
        if "fused_attn" in block_kwargs.keys():
            self.attn.fused_attn = block_kwargs["fused_attn"]
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
            )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        return x


class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        path_type='edm',
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        decoder_hidden_size=768,
        encoder_depth=8,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_cfg=False,
        z_dims=[768],
        projector_dim=2048,
        ##### added 
        skip_layer_connection=False,
        block_diversity_loss=False,
        trajectory_factorization=False,
        factor_dim=256,
        factor_projector_dim=1024,
        factor_source_depth=None,
        factor_target_depth=None,
        factor_transition=False,
        factor_velocity_recomposition=False,
        factor_native_parameterization=False,
        factor_semantic_conditioning=False,
        factor_self_flow_conditioning=False,
        factor_semantic_injection_scale=1.0,
        factor_adversarial=False,
        factor_adversarial_timestep_bins=8,
        trajectory_invariance=False,
        invariant_dim=256,
        invariant_projector_dim=1024,
        invariant_source_depth=None,
        invariant_projector_type="linear",
        factor_selective_invariance=False,
        factor_selective_dim=128,
        factor_selective_source_depth=None,
        factor_shared_source_depth=None,
        **block_kwargs # fused_attn
    ):
        super().__init__()
        self.path_type = path_type
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_cfg = use_cfg
        self.num_classes = num_classes
        self.z_dims = z_dims
        self.encoder_depth = encoder_depth
        ##### added by authors
        self.depth = depth
        self.skip_layer_connection = skip_layer_connection
        if self.skip_layer_connection:
            # Use standard skip connections, motivated by UViT
            self.skip_linears = nn.ModuleList(
            [
                ResidualLinear(hidden_size) for d in range(depth//2)
            ]
            )
            self.skip_norms = nn.ModuleList(
                [
                    FP32_Layernorm(2 * hidden_size, elementwise_affine=True, eps=1e-6) for d in range(depth//2)
                ]
            )
        ##### added block diversity loss
        self.block_diversity_loss = block_diversity_loss
        self.trajectory_factorization = trajectory_factorization
        self.factor_velocity_recomposition = factor_velocity_recomposition
        self.factor_native_parameterization = factor_native_parameterization
        self.factor_semantic_conditioning = factor_semantic_conditioning
        self.factor_self_flow_conditioning = factor_self_flow_conditioning
        self.factor_source_conditioning = (
            factor_semantic_conditioning or factor_self_flow_conditioning
        )
        self.factor_semantic_injection_scale = factor_semantic_injection_scale
        self.factor_adversarial = factor_adversarial
        self.factor_selective_invariance = factor_selective_invariance
        self.trajectory_invariance = trajectory_invariance
        if self.factor_velocity_recomposition and not self.trajectory_factorization:
            raise ValueError(
                "factor_velocity_recomposition requires trajectory_factorization"
            )
        if self.factor_native_parameterization and not self.trajectory_factorization:
            raise ValueError(
                "factor_native_parameterization requires trajectory_factorization"
            )
        if self.factor_semantic_conditioning and not self.trajectory_factorization:
            raise ValueError(
                "factor_semantic_conditioning requires trajectory_factorization"
            )
        if self.factor_self_flow_conditioning and not self.trajectory_factorization:
            raise ValueError(
                "factor_self_flow_conditioning requires trajectory_factorization"
            )
        if self.factor_semantic_conditioning and len(z_dims) == 0:
            raise ValueError(
                "factor_semantic_conditioning requires semantic target dimensions"
            )
        if factor_semantic_injection_scale < 0:
            raise ValueError("factor_semantic_injection_scale must be non-negative")
        if self.factor_native_parameterization and path_type not in {
            "linear", "cosine"
        }:
            raise ValueError(
                "factor_native_parameterization supports linear/cosine paths"
            )
        if (
            self.factor_native_parameterization
            and self.factor_velocity_recomposition
        ):
            raise ValueError(
                "native parameterization and legacy velocity recomposition "
                "are mutually exclusive"
            )
        if self.factor_semantic_conditioning and self.factor_native_parameterization:
            raise ValueError(
                "semantic conditioning and native parameterization are "
                "mutually exclusive"
            )
        if self.factor_self_flow_conditioning and self.factor_native_parameterization:
            raise ValueError(
                "self-flow source conditioning and native parameterization are "
                "mutually exclusive"
            )
        if self.factor_adversarial and not self.trajectory_factorization:
            raise ValueError("factor_adversarial requires trajectory_factorization")
        if self.factor_selective_invariance and not self.trajectory_factorization:
            raise ValueError(
                "factor_selective_invariance requires trajectory_factorization"
            )
        if factor_adversarial_timestep_bins < 2:
            raise ValueError("factor_adversarial_timestep_bins must be at least 2")
        if self.trajectory_factorization and self.trajectory_invariance:
            raise ValueError(
                "trajectory factorization and trajectory invariance are "
                "mutually exclusive training objectives"
            )
        self.factor_source_depth = (
            encoder_depth if factor_source_depth is None else factor_source_depth
        )
        self.factor_target_depth = (
            depth if factor_target_depth is None else factor_target_depth
        )
        self.factor_selective_source_depth = (
            self.factor_source_depth
            if factor_selective_source_depth is None
            else factor_selective_source_depth
        )
        self.factor_shared_source_depth = (
            self.factor_source_depth
            if factor_shared_source_depth is None
            else factor_shared_source_depth
        )
        if self.trajectory_factorization:
            if not 1 <= self.factor_source_depth <= depth:
                raise ValueError("factor_source_depth must be in [1, depth]")
            if not self.factor_source_depth <= self.factor_target_depth <= depth:
                raise ValueError(
                    "factor_target_depth must be between factor_source_depth and depth"
                )
            if (
                self.factor_selective_invariance
                and not 1 <= self.factor_selective_source_depth <= depth
            ):
                raise ValueError(
                    "factor_selective_source_depth must be in [1, depth]"
                )
            if not 1 <= self.factor_shared_source_depth <= depth:
                raise ValueError(
                    "factor_shared_source_depth must be in [1, depth]"
                )
            self.factorization_head = TrajectoryFactorizationHead(
                hidden_size=hidden_size,
                factor_dim=factor_dim,
                projector_dim=factor_projector_dim,
                predict_transition=factor_transition,
            )
        self.invariant_source_depth = (
            encoder_depth if invariant_source_depth is None
            else invariant_source_depth
        )
        if self.trajectory_invariance:
            if not 1 <= self.invariant_source_depth <= depth:
                raise ValueError("invariant_source_depth must be in [1, depth]")
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
            )
        self.t_embedder = TimestepEmbedder(hidden_size) # timestep embedding type
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)
        ])
        self.projectors = nn.ModuleList([
            build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims
        ])
        self.final_layer = FinalLayer(decoder_hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
        # The optional task decoder is registered after backbone initialization
        # inside a forked RNG context. Enabling it therefore cannot perturb the
        # seeded initialization of the backbone or the historical TFCR heads.
        if self.trajectory_factorization and self.factor_velocity_recomposition:
            with torch.random.fork_rng(devices=[]):
                self.factorization_head.enable_velocity_recomposition(
                    patch_size * patch_size * self.out_channels
                )
                for module in (
                    self.factorization_head.velocity_time_embedder.modules()
                ):
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
                for decoder in (
                    self.factorization_head.persistent_velocity_decoder,
                    self.factorization_head.evolving_velocity_decoder,
                ):
                    for module in decoder.modules():
                        if isinstance(module, nn.Linear):
                            nn.init.xavier_uniform_(module.weight)
                            if module.bias is not None:
                                nn.init.constant_(module.bias, 0)
        # The native path is opt-in and registered after historical modules so
        # enabling it cannot perturb old initializations.  Zero-initialized
        # output layers match the standard SiT head at the first optimization
        # step while the exact x0/epsilon losses immediately train the decoders.
        if self.trajectory_factorization and self.factor_native_parameterization:
            with torch.random.fork_rng(devices=[]):
                self.factorization_head.enable_native_parameterization(
                    patch_size * patch_size * self.out_channels
                )
                for decoder in (
                    self.factorization_head.native_source_decoder,
                    self.factorization_head.native_noise_decoder,
                ):
                    for module in decoder.modules():
                        if isinstance(module, nn.Linear):
                            nn.init.xavier_uniform_(module.weight)
                            if module.bias is not None:
                                nn.init.constant_(module.bias, 0)
                    nn.init.constant_(decoder[-1].weight, 0)
                    nn.init.constant_(decoder[-1].bias, 0)
        # The semantic source path is also opt-in and zero-initialized at its
        # channel gate.  It therefore starts as an exact SiT/A5 forward path,
        # while its source projector immediately receives the clean DINO loss.
        if self.trajectory_factorization and self.factor_source_conditioning:
            with torch.random.fork_rng(devices=[]):
                self.factorization_head.enable_semantic_conditioning(
                    hidden_size
                )
        # Nuisance heads are opt-in and initialized in a forked RNG context so
        # old configurations and shared model parameters stay bitwise aligned.
        if self.trajectory_factorization and self.factor_adversarial:
            with torch.random.fork_rng(devices=[]):
                self.factorization_head.enable_adversarial_nuisance(
                    factor_adversarial_timestep_bins
                )
                for head in (
                    self.factorization_head.persistent_time_discriminator,
                    self.factorization_head.persistent_orbit_discriminator,
                    self.factorization_head.evolving_time_probe,
                    self.factorization_head.evolving_orbit_probe,
                ):
                    for module in head.modules():
                        if isinstance(module, nn.Linear):
                            nn.init.xavier_uniform_(module.weight)
                            if module.bias is not None:
                                nn.init.constant_(module.bias, 0)
        # Register the new auxiliary head after all backbone modules so enabling
        # it does not shift the seeded backbone initialization.  This makes Q0
        # a true initialization- and data-order-matched control for SiT.  The
        # fork also restores the CPU RNG used later by the data sampler.
        if self.trajectory_invariance:
            with torch.random.fork_rng(devices=[]):
                self.invariance_head = TrajectoryInvariantProjector(
                    hidden_size=hidden_size,
                    invariant_dim=invariant_dim,
                    projector_dim=invariant_projector_dim,
                    projector_type=invariant_projector_type,
                )
                for module in self.invariance_head.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
        # VGSC is an A3-compatible selective readout.  Register it last and in
        # a forked RNG context so every historical model and optional head keeps
        # exactly the same seeded initialization when this feature is disabled.
        if self.factor_selective_invariance:
            with torch.random.fork_rng(devices=[]):
                self.factor_selective_invariance_head = (
                    TrajectoryInvariantProjector(
                        hidden_size=hidden_size,
                        invariant_dim=factor_selective_dim,
                        projector_dim=factor_selective_dim,
                        projector_type="linear",
                    )
                )
                nn.init.xavier_uniform_(
                    self.factor_selective_invariance_head.projector.weight
                )

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, patch_size=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs
    
    def forward(
        self,
        x,
        t,
        y,
        return_logvar=False,
        trajectory_pair=False,
        factor_delta_t=None,
        return_factorization=False,
        factor_pair_count=None,
        factor_adversarial_grl_scale=1.0,
        return_invariance=False,
        invariant_group_count=None,
        invariant_view_count=3,
        force_drop_ids=None,
        return_selective_invariance=False,
        return_shared_target=False,
        return_semantic_factorization=False,
    ):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        N, T, D = x.shape

        # timestep and class embedding
        t_embed = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(
            y, self.training, force_drop_ids=force_drop_ids
        )                                               # (N, D)
        c = t_embed + y                                # (N, D)

        skips = []
        collect_block_features = self.block_diversity_loss and self.training
        block_feas = {}
        zs = None
        factor_source = None
        factor_target = None
        factor_selective_source = None
        factor_shared_source = None
        invariant_source = None
        semantic_persistent = None
        semantic_evolving = None
        semantic_persistent_component = None
        semantic_evolving_component = None
        semantic_source_predictions = None
        semantic_modulation_rms = None
        for i, block in enumerate(self.blocks): 
            x = block(x, c) 
            ##### added skip-layer connection
            if i >= self.depth//2:
                if self.skip_layer_connection:
                    skip_linear = self.skip_linears[i-self.depth//2]
                    skip_norm = self.skip_norms[i-self.depth//2]
                    skip = skips.pop()
                    cat = torch.cat([x, skip], dim=-1)
                    cat = skip_norm(cat)
                    x = skip_linear(cat, x)
            if i < self.depth //2:
                skips.append(x)
            ##### added projection loss
            if ((i + 1) == self.encoder_depth
                    and not self.factor_source_conditioning):
                zs = [projector(x.reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors]
            if (self.trajectory_factorization and return_factorization
                    and (i + 1) == self.factor_source_depth):
                factor_source = x
            if (self.factor_source_conditioning
                    and (i + 1) == self.factor_source_depth):
                # The unmodified hidden stream remains the view/state carrier.
                # Only a low-dimensional source code controls the late blocks.
                semantic_persistent, semantic_evolving = (
                    self.factorization_head.factorize(x)
                )
                (
                    semantic_persistent_component,
                    semantic_source_shift,
                ) = self.factorization_head.semantic_modulation(
                    semantic_persistent
                )
                injection_scale = self.factor_semantic_injection_scale
                conditioned_x = x + injection_scale * semantic_source_shift
                semantic_modulation_rms = (
                    conditioned_x.float() - x.float()
                ).square().mean().sqrt().detach()
                x = conditioned_x
                if return_semantic_factorization:
                    if self.factor_semantic_conditioning:
                        semantic_source_predictions = (
                            self.factorization_head.predict_semantic_source(
                                semantic_persistent_component, self.projectors
                            )
                        )
                    else:
                        semantic_source_predictions = []
            if (self.trajectory_factorization and return_factorization
                    and (i + 1) == self.factor_target_depth):
                factor_target = x
            if (self.factor_selective_invariance
                    and return_selective_invariance
                    and (i + 1) == self.factor_selective_source_depth):
                # Keep the full activation object: the main denoising output
                # depends on it, so the loss can query dL_FM / dh without a
                # second-order graph.  Pair slicing happens only for the
                # projector readout below.
                factor_selective_source = x
            if (self.trajectory_factorization
                    and return_shared_target
                    and (i + 1) == self.factor_shared_source_depth):
                factor_shared_source = x
            if (self.trajectory_invariance and return_invariance
                    and (i + 1) == self.invariant_source_depth):
                invariant_source = x
            if collect_block_features:
                ##### get features of all blocks for computing block diversity loss
                block_feas[i] = x 
        final_features = x
        x = self.final_layer(final_features, c)   # (N, T, patch_size ** 2 * out_channels)
        base_velocity = self.unpatchify(x)        # (N, out_channels, H, W)
        # denoising loss
        # a dict to store the results
        result = {'x': base_velocity}
        native_persistent = None
        native_evolving = None
        if self.factor_native_parameterization:
            native_persistent, native_evolving = (
                self.factorization_head.factorize(final_features)
            )
            source_tokens, noise_tokens = (
                self.factorization_head.predict_native_components(
                    native_persistent, native_evolving
                )
            )
            if self.path_type == "linear":
                d_alpha = -torch.ones_like(t)
                d_sigma = torch.ones_like(t)
            elif self.path_type == "cosine":
                angle = t * (math.pi / 2)
                d_alpha = -(math.pi / 2) * torch.sin(angle)
                d_sigma = (math.pi / 2) * torch.cos(angle)
            else:
                raise RuntimeError(
                    "native parameterization supports linear/cosine paths"
                )
            d_alpha = d_alpha.to(source_tokens.dtype).reshape(-1, 1, 1)
            d_sigma = d_sigma.to(noise_tokens.dtype).reshape(-1, 1, 1)
            native_velocity_tokens = (
                d_alpha * source_tokens + d_sigma * noise_tokens
            )
            source_prediction = self.unpatchify(source_tokens)
            noise_prediction = self.unpatchify(noise_tokens)
            result['x'] = self.unpatchify(native_velocity_tokens)
            result['native_parameterization'] = {
                'source': source_prediction,
                'noise': noise_prediction,
                'base_velocity': base_velocity,
            }
        # return all activations for computing block diversity loss
        if collect_block_features:
            result['block_feas'] = block_feas
        result['zs'] = zs
        if self.factor_source_conditioning and return_semantic_factorization:
            if semantic_persistent is None or semantic_evolving is None:
                raise RuntimeError("source-conditioned factorization was not collected")
            semantic_evolving_component = (
                self.factorization_head.evolving_decoder(semantic_evolving)
            )
            semantic_pair_count = None
            if trajectory_pair:
                semantic_pair_count = (
                    N // 2 if factor_pair_count is None else factor_pair_count
                )
                if not 0 < semantic_pair_count <= N // 2:
                    raise ValueError(
                        "factor_pair_count is incompatible with semantic "
                        "factorization"
                    )
            result['semantic_factorization'] = {
                'source': semantic_persistent,
                'evolving': semantic_evolving,
                'source_component': semantic_persistent_component,
                'evolving_component': semantic_evolving_component,
                'source_predictions': semantic_source_predictions,
                'pair_count': semantic_pair_count,
                'modulation_rms': semantic_modulation_rms,
            }
        if self.trajectory_invariance and return_invariance:
            if invariant_source is None:
                raise RuntimeError("invariant source feature was not collected")
            if trajectory_pair:
                group_count = (
                    N // invariant_view_count
                    if invariant_group_count is None
                    else invariant_group_count
                )
                if invariant_view_count < 2:
                    raise ValueError("invariant_view_count must be at least 2")
                if not 0 < group_count <= N // invariant_view_count:
                    raise ValueError(
                        "invariant_group_count is incompatible with batch/view count"
                    )
                invariant_source = invariant_source[
                    :invariant_view_count * group_count
                ]
            result['invariance'] = {
                'features': self.invariance_head(invariant_source),
                # Expose the exact post-skip feature consumed by the readout.
                # Forward hooks on a transformer block observe its output
                # before SiT's external skip fusion, so they are not equivalent
                # for source depths in the decoder half of the network.
                'source_features': invariant_source,
                'basis_orthogonality_loss': (
                    self.invariance_head.orthogonality_loss()
                ),
            }
        if self.factor_selective_invariance and return_selective_invariance:
            if factor_selective_source is None:
                raise RuntimeError(
                    "factor selective-invariance source was not collected"
                )
            selective_readout_source = factor_selective_source
            selective_pair_count = None
            if trajectory_pair:
                selective_pair_count = (
                    N // 2 if factor_pair_count is None else factor_pair_count
                )
                if not 0 < selective_pair_count <= N // 2:
                    raise ValueError(
                        "factor_pair_count is incompatible with selective "
                        "invariance"
                    )
                selective_readout_source = selective_readout_source[
                    :2 * selective_pair_count
                ]
            result['selective_invariance'] = {
                'features': self.factor_selective_invariance_head(
                    selective_readout_source
                ),
                # This must remain the unsliced activation produced inside the
                # backbone.  autograd.grad(loss, a post-hoc slice) is not the
                # same graph query and may be unused.
                'source_features': factor_selective_source,
                'basis': (
                    self.factor_selective_invariance_head.normalized_basis()
                ),
                'basis_orthogonality_loss': (
                    self.factor_selective_invariance_head.orthogonality_loss()
                ),
                'pair_count': selective_pair_count,
            }
        if return_shared_target:
            if not self.trajectory_factorization:
                raise RuntimeError(
                    "shared target readout requires trajectory_factorization"
                )
            if factor_shared_source is None:
                raise RuntimeError("shared target source feature was not collected")
            shared_source = factor_shared_source
            shared_pair_count = None
            if trajectory_pair:
                shared_pair_count = (
                    N // 2 if factor_pair_count is None else factor_pair_count
                )
                if not 0 < shared_pair_count <= N // 2:
                    raise ValueError(
                        "factor_pair_count is incompatible with shared target"
                    )
                shared_source = shared_source[:2 * shared_pair_count]
            result['shared_target'] = {
                'features': shared_source,
                'source_features': factor_shared_source,
                'pair_count': shared_pair_count,
            }
        if self.trajectory_factorization and return_factorization:
            if trajectory_pair:
                pair_count = N // 2 if factor_pair_count is None else factor_pair_count
                if not 0 < pair_count <= N // 2:
                    raise ValueError("factor_pair_count must be in (0, batch_size // 2]")
                factor_batch_size = 2 * pair_count
                factor_target = factor_target[:factor_batch_size]
            if (
                self.factor_source_conditioning
                and semantic_persistent is not None
                and semantic_persistent_component is not None
            ):
                if semantic_evolving_component is None:
                    semantic_evolving_component = (
                        self.factorization_head.evolving_decoder(
                            semantic_evolving
                        )
                    )
                persistent = semantic_persistent
                evolving = semantic_evolving
                persistent_component = semantic_persistent_component
                evolving_component = semantic_evolving_component
                if trajectory_pair:
                    persistent = persistent[:factor_batch_size]
                    evolving = evolving[:factor_batch_size]
                    persistent_component = persistent_component[
                        :factor_batch_size
                    ]
                    evolving_component = evolving_component[:factor_batch_size]
            else:
                if trajectory_pair:
                    factor_source = factor_source[:factor_batch_size]
                persistent, evolving = self.factorization_head.factorize(
                    factor_source
                )
                persistent_component, evolving_component = (
                    self.factorization_head.decode(persistent, evolving)
                )
            factorization = {
                'persistent': persistent,
                'evolving': evolving,
                'persistent_component': persistent_component,
                'evolving_component': evolving_component,
                'target': factor_target,
            }
            if trajectory_pair:
                evolving_a, evolving_b = evolving.chunk(2, dim=0)
                persistent_component_a, persistent_component_b = (
                    persistent_component.chunk(2, dim=0)
                )
                evolving_component_a, evolving_component_b = (
                    evolving_component.chunk(2, dim=0)
                )
                factorization['recomposed'] = torch.cat([
                    persistent_component_b + evolving_component_a,
                    persistent_component_a + evolving_component_b,
                ], dim=0)
                # These intentionally mismatched reconstructions are diagnostics:
                # a useful evolving code should make them worse than the correct swap.
                with torch.no_grad():
                    factorization['wrong_evolving_recomposed'] = torch.cat([
                        persistent_component_b + evolving_component_b,
                        persistent_component_a + evolving_component_a,
                    ], dim=0)
                if self.factorization_head.predict_velocity_recomposition:
                    persistent_velocity, evolving_velocity = (
                        self.factorization_head.predict_velocity(
                            persistent, evolving, t[:factor_batch_size]
                        )
                    )
                    persistent_velocity_a, persistent_velocity_b = (
                        persistent_velocity.chunk(2, dim=0)
                    )
                    swapped_persistent_velocity = torch.cat([
                        persistent_velocity_b, persistent_velocity_a
                    ], dim=0)
                    factorization['persistent_velocity_component'] = (
                        self.unpatchify(persistent_velocity)
                    )
                    factorization['evolving_velocity_component'] = (
                        self.unpatchify(evolving_velocity)
                    )
                    factorization['velocity_recomposed'] = self.unpatchify(
                        swapped_persistent_velocity + evolving_velocity
                    )
                if self.factorization_head.predict_adversarial_nuisance:
                    factorization['nuisance_predictions'] = (
                        self.factorization_head.predict_nuisance(
                            persistent,
                            evolving,
                            grl_scale=factor_adversarial_grl_scale,
                        )
                    )
                if self.factorization_head.predict_transition:
                    if factor_delta_t is None:
                        raise ValueError(
                            "factor_delta_t is required when factor_transition is enabled"
                        )
                    factorization['transitioned'] = torch.cat([
                        self.factorization_head.transition(evolving_a, factor_delta_t),
                        self.factorization_head.transition(evolving_b, -factor_delta_t),
                    ], dim=0)
            result['factorization'] = factorization
        return result


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def SiT_XL_4(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def SiT_XL_8(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def SiT_L_4(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def SiT_L_8(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def SiT_B_4(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def SiT_B_8(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, decoder_hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, decoder_hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def SiT_S_8(**kwargs):
    return SiT(depth=12, hidden_size=384, decoder_hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SiT_models = {
    'SiT-XL/2': SiT_XL_2,  'SiT-XL/4': SiT_XL_4,  'SiT-XL/8': SiT_XL_8,
    'SiT-L/2':  SiT_L_2,   'SiT-L/4':  SiT_L_4,   'SiT-L/8':  SiT_L_8,
    'SiT-B/2':  SiT_B_2,   'SiT-B/4':  SiT_B_4,   'SiT-B/8':  SiT_B_8,
    'SiT-S/2':  SiT_S_2,   'SiT-S/4':  SiT_S_4,   'SiT-S/8':  SiT_S_8,
}
