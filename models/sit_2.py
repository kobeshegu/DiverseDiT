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
from models.wavelet_utils import WaveletSkipNTD, WaveletSkipNTDEnhanced, WaveletSkipSimple, WaveletSkipResidual


class SimpleHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SimpleHead, self).__init__()
        self.linear1 = nn.Linear(in_dim, in_dim+out_dim)
        self.linear2 = nn.Linear(in_dim+out_dim, out_dim)
        self.act = nn.SiLU()
    def forward(self, x):
        x=self.linear1(x)
        x=self.linear2(self.act(x))
        return x

def build_mlp(hidden_size, projector_dim, z_dim, **kwargs):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )

def build_mlp_multiple_projection(hidden_size, projector_dim, z_dim=768):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )
    
def build_conv_projection(hidden_size, z_dim, patch_size=2):
    """
    Build lightweight convolutional projection layer for iREPA.
    Replaces MLP with conv layer to preserve spatial relationships.
    """
    return nn.Sequential(
        nn.Conv2d(hidden_size, z_dim, kernel_size=3, padding=1, bias=True),
        nn.SiLU(),
    )




#################################################################################
#               Build Projection Layers From Improved REPA                      #
#################################################################################
ALL_PROJECTION_LAYER_TYPES = ["mlp", "linear", "conv"]
class ProjectionLayer(nn.Module):
    def __init__(self, projection_layer_type="mlp", **kwargs):
        super().__init__()
        assert projection_layer_type in ALL_PROJECTION_LAYER_TYPES, f"Unsupported projection layer type: {projection_layer_type}. Must be one of {ALL_PROJECTION_LAYER_TYPES}"
        # self.kwargs = kwargs
        self.projection_layer_type = projection_layer_type 
        self.build_projection_layer(projection_layer_type, **kwargs)

    def build_projection_layer(self, projection_layer_type, **kwargs):
        if projection_layer_type == "mlp":
            self.projection_layer = build_mlp(**kwargs)
        elif projection_layer_type == "linear":
            in_dim  = kwargs.pop("hidden_size")
            out_dim = kwargs.pop("z_dim")
            self.projection_layer = nn.Linear(in_dim, out_dim)
        elif projection_layer_type == "conv":
            in_ch  = kwargs.pop("hidden_size")
            out_ch = kwargs.pop("z_dim")
            kernel_size = kwargs.pop("proj_kwargs_kernel_size")
            padding = kernel_size // 2 # to keep spatial dimension
            self.projection_layer = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding)
        else:
            raise ValueError(f"Unsupported projection layer type: {projection_layer_type}")

    def forward(self, x, hw: tuple[int, int] | None = None):
        """
        x: [B, T, D]
        hw: optional (H, W) for non-square token grids (mostly not used).
        """
        B, T, D = x.shape
        if self.projection_layer_type in ("mlp", "linear"):
            x_ = self.projection_layer(x.reshape(B * T, D))
            return x_.reshape(B, T, -1)

        elif self.projection_layer_type == "conv":
            if hw is None:
                H = W = int(math.isqrt(T))
                assert H * W == T, f"conv projector needs square grid or pass hw; got T={T}"
            else:
                H, W = hw
                assert H * W == T, f"Provided hw={hw} but T={T}"

            # [B, T, D] -> [B, D, H, W]
            x_ = x.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()
            y  = self.projection_layer(x_)                  # [B, z_dim, H, W]
            y  = y.permute(0, 2, 3, 1).contiguous()         # [B, H, W, z_dim]
            return y.reshape(B, T, -1)

class SpatialNormalization(nn.Module):
    """
    Spatial normalization layer for iREPA.
    Normalizes patch tokens across spatial dimension to improve spatial contrast.
    
    Formula: y = (x - E[x]) / sqrt(Var[x] + epsilon)
    Where E[x] and Var[x] are computed across spatial dimension.
    """
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, x):
        """
        Args:
            x: (B, T, D) tensor of patch tokens
        Returns:
            y: (B, T, D) normalized patch tokens
        """
        # Compute mean and variance across spatial dimension (dim=1)
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, D)
        var = x.var(dim=1, keepdim=True)    # (B, 1, D)
        
        # Apply normalization
        y = (x - mean) / torch.sqrt(var + self.epsilon)
        return y

class GradientIsolation(torch.autograd.Function):
    """
    Gradient Isolation (Stop-Gradient Barrier) between block groups.

    Forward: identity (x passes through unchanged).
    Backward: gradient is scaled by `alpha`.
      - alpha=0 : full stop-gradient (no gradient flows back)
      - alpha=1 : no effect (full gradient)
      - 0<alpha<1: soft barrier (partial gradient)

    Usage: insert between block groups to force each group to learn
    independently useful features instead of relying on gradient
    signals from later layers.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.alpha, None


def gradient_isolate(x, alpha=0.0):
    """Apply gradient isolation barrier. alpha=0 means full stop-gradient."""
    return GradientIsolation.apply(x, alpha)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# sara similarity
def similarity(x):
    """
    x: (N, T, D)
    """
    x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
    x_sim = torch.matmul(x_norm, x_norm.transpose(1, 2))    # (N, T, T)
    return x_sim

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
        return embeddings, labels


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Supports per-block residual scaling and heterogeneous head/MLP configs.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,
                 block_index=0, total_blocks=28,
                 residual_scaling=False,
                 **block_kwargs):
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

        # --- Per-Block Residual Scaling ---
        # Learnable scalar controlling each block's contribution strength.
        # Initialised with depth-dependent values to break symmetry:
        #   shallow layers → larger scale (learn basic features faster)
        #   deep layers    → smaller scale (refine slowly)
        self.residual_scaling = residual_scaling
        if residual_scaling:
            progress = block_index / max(total_blocks - 1, 1)  # 0 → 1
            init_scale = 1.0 - 0.5 * progress  # 1.0 → 0.5
            self.residual_scale = nn.Parameter(torch.tensor(init_scale))
        else:
            self.residual_scale = None

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        attn_out = gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        mlp_out = gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        if self.residual_scale is not None:
            # Clamp to [0.01, 2.0] for stability
            scale = self.residual_scale.clamp(0.01, 2.0)
            x = x + scale * attn_out
            x = x + scale * mlp_out
        else:
            x = x + attn_out
            x = x + mlp_out

        return x


# class FinalLayer(nn.Module):
#     """
#     The final layer of SiT.
#     """
#     def __init__(self, hidden_size, patch_size, out_channels):
#         super().__init__()
#         self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(hidden_size, 2 * hidden_size, bias=True)
#         )

#     def forward(self, x, c):
#         shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
#         x = modulate(self.norm_final(x), shift, scale)
#         x = self.linear(x)

#         return x


class MidFinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.silu(self.linear(self.norm_final(x)))
        return x
    
class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, cls_token_dim, cls_token=False):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.cls_token = cls_token
        if self.cls_token:
            self.linear_cls = nn.Linear(hidden_size, cls_token_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c, cls=None):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)

        if cls is None:
            x = self.linear(x)
            return x
        else:
            cls_token = self.linear_cls(x[:, 0]).unsqueeze(1)
            x = self.linear(x[:, 1:])
            return x, cls_token.squeeze(1)


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
        eval_mode=False,
        projector_dim=2048,
        projection_layer_type="mlp",
        proj_kwargs_kernel_size=3,
        sra=False,
        projection=False,
        sara=False,
        cka=False,
        feature_dissimilarity=False,
        skipped_layers_eval=False,
        cls_token_loss=False,
        cls_token_dim=768,
        teacher_representation_loss=False,
        teacher_representation_block=6,
        multiple_projection=False,
        multiple_projection_blocks=None,
        auc_mse_prediction=False,
        diff_projection=False,
        diff_projection_blocks=None,
        cross_layer_connection=False,
        wavelet_skip_connection=False,
        wavelet_type='simple',  # 'simple', 'enhanced', 'residual', or 'ntd',
        block_difference_loss=False,  # Enable block difference loss
        block_diversity_loss=False,
        irepa_conv_projection=False,  # Use conv projection instead of MLP
        irepa_spatial_norm=False,  # Use spatial normalization
        layer_drop=False,  # Enable random layer drop
        layer_drop_rate=0.1,  # Layer drop probability
        layer_drop_strategy='uniform',  # Layer drop strategy
        layer_drop_type='random',  # Layer drop type ('random' or 'adaptive')
        # --- New diversity methods ---
        gradient_isolation=False,  # Enable gradient isolation between block groups
        gradient_isolation_alpha=0.0,  # Gradient scale at barriers (0=full stop, 1=none)
        gradient_isolation_layers=None,  # List of layer indices to place barriers AFTER
        block_shuffling=False,  # Enable random block order shuffling during training
        block_shuffling_prob=0.1,  # Probability of shuffling in each forward pass
        block_shuffling_group_size=4,  # Shuffle within groups of this size
        per_block_conditioning=False,  # Enable learnable per-block conditioning offsets
        block_aux_heads=False,  # Enable block-wise auxiliary prediction heads
        block_aux_head_layers=None,  # List of layer indices to attach aux heads
        block_aux_head_target='denoise',  # 'denoise' or 'teacher'
        # --- Structured Heterogeneity (Tier 1 & 2) ---
        residual_scaling=False,  # 1A: per-block learnable residual scale
        block_group_conditioning=False,  # 1B: per-group conditioning transform
        num_cond_groups=4,  # 1B: number of conditioning groups
        heterogeneous_mlp=False,  # 2A: varying MLP ratio across blocks
        alternating_heads=False,  # 2B: alternate attention head counts
        depth_aware_init=False,  # 2C: depth-scaled weight initialization
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
        self.eval_mode = eval_mode
        # sra
        self.sra = sra
        # projection
        self.projection = projection
        # multiple projection
        self.multiple_projection = multiple_projection
        # self.multiple_projection_blocks = multiple_projection_blocks
        if multiple_projection_blocks is not None and self.multiple_projection:
            self.multiple_projection_blocks =  [int(x) for x in self.multiple_projection_blocks] 
        # diff projection
        self.diff_projection = diff_projection
        self.diff_projection_blocks = diff_projection_blocks
        if diff_projection_blocks is not None and self.diff_projection:
                self.diff_projection_blocks =  [int(x) for x in self.diff_projection_blocks] 
        # sara
        self.sara = sara
        # cka similarity
        self.cka = cka
        # feature dissimilarity
        self.feature_dissimilarity = feature_dissimilarity
        # block difference loss
        self.block_difference_loss = block_difference_loss
        # block diversity loss
        self.block_diversity_loss = block_diversity_loss
        # Storage for important blocks (e.g., projection blocks)
        self._important_blocks = None
        # skipped_layers_eval
        self.skipped_layers_eval = skipped_layers_eval
        # cls token 
        self.cls_token_loss = cls_token_loss
        # irepa
        self.projection_layer_type = projection_layer_type
        self.irepa_conv_projection = irepa_conv_projection
        self.irepa_spatial_norm = irepa_spatial_norm
        # aus mse loss
        self.auc_mse_prediction = auc_mse_prediction
        if self.auc_mse_prediction:
            self.mid_final_layer = FinalLayer(decoder_hidden_size, patch_size, self.out_channels, cls_token_dim, self.cls_token_loss)
        if self.cls_token_loss:
            z_dim = self.z_dims[0]
            cls_token_dim = z_dim
            self.cls_projectors2 = nn.Linear(in_features=cls_token_dim, out_features=hidden_size, bias=True)
            self.wg_norm = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
            )
        self.t_embedder = TimestepEmbedder(hidden_size) # timestep embedding type
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        if self.cls_token_loss:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, hidden_size), requires_grad=False)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.depth = depth
        self.residual_scaling = residual_scaling
        self.heterogeneous_mlp = heterogeneous_mlp
        self.alternating_heads = alternating_heads
        self.depth_aware_init = depth_aware_init

        # --- 2A: Compute per-block MLP ratios ---
        if heterogeneous_mlp:
            mlp_ratios = self._build_heterogeneous_mlp_ratios(depth, mlp_ratio)
        else:
            mlp_ratios = [mlp_ratio] * depth

        # --- 2B: Compute per-block head counts ---
        if alternating_heads:
            head_counts = self._build_alternating_heads(depth, num_heads)
        else:
            head_counts = [num_heads] * depth

        self.blocks = nn.ModuleList([
            SiTBlock(
                hidden_size,
                head_counts[i],
                mlp_ratio=mlp_ratios[i],
                block_index=i,
                total_blocks=depth,
                residual_scaling=residual_scaling,
                **block_kwargs
            ) for i in range(depth)
        ])
        # projection
        # if not eval_mode:
        if self.projection:
            if self.irepa_conv_projection:
                # Use iREPA config
                self.projectors = nn.ModuleList([
                    ProjectionLayer(projection_layer_type, hidden_size=hidden_size, z_dim=z_dim, projector_dim=projector_dim, proj_kwargs_kernel_size=proj_kwargs_kernel_size) for z_dim in z_dims
                ])
            else:
                # Use standard MLP projection
                self.projectors = nn.ModuleList([
                    build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims
                ])
        # iREPA spatial normalization
        if self.irepa_spatial_norm:
            self.spatial_norm = SpatialNormalization()
        # Layer drop settings
        self.layer_drop = layer_drop
        self.layer_drop_rate = layer_drop_rate
        self.layer_drop_strategy = layer_drop_strategy
        self.layer_drop_type = layer_drop_type
        # multiple_projection
        if self.multiple_projection and self.multiple_projection_blocks is not None:
            assert len(z_dims) == len(self.multiple_projection_blocks)
            self.multiple_projectors = nn.ModuleList([
                build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims
                ])
        # diff projection
        if self.diff_projection and self.diff_projection_blocks is not None:
            assert len(z_dims) == len(self.diff_projection_blocks)
            self.multiple_diff_projectors = nn.ModuleList([
                build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims
                ])
        # sra loss
        if self.sra:
            self.ap_head = SimpleHead(hidden_size, hidden_size)
        # teacher_representation_loss
        self.teacher_representation_loss = teacher_representation_loss
        self.teacher_representation_block = teacher_representation_block
        if self.teacher_representation_loss:
            self.teacher_head = SimpleHead(hidden_size, hidden_size)
        # Initialize layer drop module
        if self.layer_drop:
            from .layer_drop import create_layer_drop
            self.layer_drop_module = create_layer_drop(
                num_layers=depth,
                drop_type=layer_drop_type,
                drop_rate=layer_drop_rate,
                drop_strategy=layer_drop_strategy,
                min_keep_layers=max(depth // 2, 1)
            )
        else:
            self.layer_drop_module = None

        # --- Gradient Isolation ---
        self.gradient_isolation = gradient_isolation
        self.gradient_isolation_alpha = gradient_isolation_alpha
        if gradient_isolation_layers is not None:
            self.gradient_isolation_layers = [int(x) for x in gradient_isolation_layers]
        else:
            # Default: place barriers at 1/3 and 2/3 depth
            self.gradient_isolation_layers = [depth // 3, 2 * depth // 3] if gradient_isolation else []

        # --- Block Shuffling ---
        self.block_shuffling = block_shuffling
        self.block_shuffling_prob = block_shuffling_prob
        self.block_shuffling_group_size = block_shuffling_group_size

        # --- Per-Block Conditioning Perturbation (simple offset, kept for backward compat) ---
        self.per_block_conditioning = per_block_conditioning
        if per_block_conditioning:
            self.block_cond_offsets = nn.ParameterList([
                nn.Parameter(torch.zeros(hidden_size) * 0.01) for _ in range(depth)
            ])

        # --- 1B: Block-Group Conditioning Transform ---
        self.block_group_conditioning = block_group_conditioning
        self.num_cond_groups = num_cond_groups
        if block_group_conditioning:
            self.cond_transforms = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size, bias=True),
                    nn.SiLU(),
                ) for _ in range(num_cond_groups)
            ])

        # --- Block-wise Auxiliary Prediction Heads ---
        self.block_aux_heads_enabled = block_aux_heads
        self.block_aux_head_target = block_aux_head_target
        if block_aux_head_layers is not None:
            self.block_aux_head_layers = [int(x) for x in block_aux_head_layers]
        else:
            # Default: evenly spaced at 1/4, 1/2, 3/4 depth
            self.block_aux_head_layers = [depth // 4, depth // 2, 3 * depth // 4] if block_aux_heads else []
        if block_aux_heads and len(self.block_aux_head_layers) > 0:
            out_dim = patch_size * patch_size * in_channels  # same as final layer output per token
            self.block_aux_head_projs = nn.ModuleDict({
                str(layer_idx): nn.Sequential(
                    nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
                    nn.Linear(hidden_size, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, out_dim),
                ) for layer_idx in self.block_aux_head_layers
            })
        else:
            self.block_aux_head_projs = None

        # cross_layer_connection
        self.cross_layer_connection = cross_layer_connection
        # wavelet_skip_connection:
        self.wavelet_skip_connection = wavelet_skip_connection
        self.wavelet_type = wavelet_type
        if self.cross_layer_connection:
            if self.wavelet_skip_connection:
                # Use wavelet-based skip connections for (N, T, D) format
                if wavelet_type == 'simple':
                    # Simple wavelet with high-freq processing (recommended)
                    self.wavelet_skip_connections = nn.ModuleList([
                        WaveletSkipSimple(hidden_size, patch_size) 
                        for d in range(depth//2)
                    ])
                elif wavelet_type == 'enhanced':
                    # Multi-level wavelet decomposition
                    self.wavelet_skip_connections = nn.ModuleList([
                        WaveletSkipNTDEnhanced(hidden_size, patch_size, num_levels=2) 
                        for d in range(depth//2)
                    ])
                elif wavelet_type == 'residual':
                    # Residual connection with separate low/high processing
                    self.wavelet_skip_connections = nn.ModuleList([
                        WaveletSkipResidual(hidden_size, patch_size) 
                        for d in range(depth//2)
                    ])
                elif wavelet_type == 'ntd':
                    # Full NTD wavelet with separate processors
                    self.wavelet_skip_connections = nn.ModuleList([
                        WaveletSkipNTD(hidden_size, patch_size, use_wavelet=True, process_high_freq=True) 
                        for d in range(depth//2)
                    ])
                else:
                    raise ValueError(f"Unknown wavelet_type: {wavelet_type}. Choose from: simple, enhanced, residual, ntd")
            else:
                # Use standard skip connections
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
        self.final_layer = FinalLayer(decoder_hidden_size, patch_size, self.out_channels, cls_token_dim, self.cls_token_loss)
        self.initialize_weights()

    @staticmethod
    def _build_heterogeneous_mlp_ratios(depth, base_ratio=4.0):
        """
        2A: Generate varying MLP expansion ratios across depth.
        Shallow layers get larger ratios (broader features),
        deep layers get smaller ratios (more focused features).
        Periodic variation adds extra heterogeneity within each trend.
        Total parameter count stays roughly the same as uniform base_ratio.
        """
        ratios = []
        for i in range(depth):
            # Linearly decreasing base: base_ratio → base_ratio - 1.0
            trend = base_ratio - 1.0 * (i / max(depth - 1, 1))
            # Periodic variation: ±0.5 with period 4
            variation = [0, 0.5, -0.5, 0.25][i % 4]
            ratio = max(trend + variation, 2.0)  # floor at 2.0
            ratios.append(round(ratio, 2))
        return ratios

    @staticmethod
    def _build_alternating_heads(depth, base_heads=16):
        """
        2B: Alternate attention head counts across blocks.
        Even blocks: base_heads      (e.g. 16 heads, head_dim=72)
        Odd blocks:  base_heads // 2 (e.g.  8 heads, head_dim=144)
        Different head dims → different attention granularity → natural diversity.
        Total QKV parameter count is identical (only partitioning changes).
        """
        half = max(base_heads // 2, 1)
        return [base_heads if i % 2 == 0 else half for i in range(depth)]

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        if self.cls_token_loss:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5), cls_token=1, extra_tokens=1
                )
        else:
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

        # --- 2C: Depth-Aware Initialization ---
        # Scale attn/MLP weights by a depth-dependent factor.
        # Deeper layers get smaller init → they start closer to identity longer,
        # breaking the symmetry of "all blocks wake up at the same speed".
        # adaLN stays zero-init (preserves initial identity), only attn/MLP affected.
        if self.depth_aware_init:
            for i, block in enumerate(self.blocks):
                depth_scale = 1.0 - 0.3 * (i / max(self.depth - 1, 1))  # 1.0 → 0.7
                for name, param in block.named_parameters():
                    if 'adaLN' not in name and 'weight' in name and param.dim() >= 2:
                        param.data *= depth_scale

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        if self.auc_mse_prediction:
            nn.init.constant_(self.mid_final_layer.linear.weight, 0)
            nn.init.constant_(self.mid_final_layer.linear.bias, 0)
        if self.cls_token_loss:
            nn.init.constant_(self.final_layer.linear_cls.weight, 0)
            nn.init.constant_(self.final_layer.linear_cls.bias, 0)


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
    
    def set_important_blocks_for_diversity(self, block_indices):
        """
        Set important block indices that must be included in diversity loss computation.
        These blocks will always be selected (not subject to random sampling).
        
        Args:
            block_indices: list of int, block indices to always include
                          (e.g., blocks used for projection or other critical features)
        
        Example:
            # Ensure blocks 3, 7, and 11 are always included in diversity loss
            model.set_important_blocks_for_diversity([3, 7, 11])
            
            # For projection at specific layers (e.g., final layer)
            model.set_important_blocks_for_diversity([model.depth - 1])
        """
        self._important_blocks = block_indices if block_indices else None
    
    def forward(self, x, t, y, disperse_loss_block=None, ad=None, cls_token=None, teacher_block=None):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        if cls_token is not None:
            x = self.x_embedder(x)
            cls_token = self.cls_projectors2(cls_token)
            cls_token = self.wg_norm(cls_token)
            cls_token = cls_token.unsqueeze(1)  # [b, length, d]
            x = torch.cat((cls_token, x), dim=1)
            x = x + self.pos_embed   
        else:         
            x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        N, T, D = x.shape

        # timestep and class embedding
        t_embed = self.t_embedder(t)                   # (N, D)
        y, labels_train = self.y_embedder(y, self.training)    # (N, D)
        c = t_embed + y                               # (N, D)

        block_feas = {}
        acts = []
        zs_multiple_blocks = []
        zs_multiple_diff_blocks = []
        skips = []
        aux_outputs = {}  # for block-wise auxiliary heads
        total_layers = len(self.blocks)
        # Sample layer drop pattern if enabled
        layer_drop_mask = None
        if self.layer_drop and self.training and self.layer_drop_module is not None:
            layer_drop_mask = self.layer_drop_module.sample_drop_pattern(batch_size=N)
            layer_drop_mask = layer_drop_mask.to(x.device)

        # --- Block Shuffling: compute execution order ---
        block_order = list(range(len(self.blocks)))
        if self.block_shuffling and self.training and torch.rand(1).item() < self.block_shuffling_prob:
            gs = self.block_shuffling_group_size
            for g_start in range(0, len(self.blocks), gs):
                g_end = min(g_start + gs, len(self.blocks))
                group = block_order[g_start:g_end]
                perm = torch.randperm(len(group)).tolist()
                block_order[g_start:g_end] = [group[p] for p in perm]

        for step, i in enumerate(block_order):
            block = self.blocks[i]
            # if self.training:
            if i >= self.depth//2:
                if self.cross_layer_connection:
                    if self.wavelet_skip_connection:
                        # Use wavelet-based skip connection
                        wavelet_skip = self.wavelet_skip_connections[i-self.depth//2]
                        skip = skips.pop()
                        x = wavelet_skip(x, skip)
                    else:
                        skip_linear = self.skip_linears[i-self.depth//2]
                        skip_norm = self.skip_norms[i-self.depth//2]
                        skip = skips.pop()
                        cat = torch.cat([x, skip], dim=-1)
                        cat = skip_norm(cat)
                        x = skip_linear(cat, x)
            if i < self.depth //2:
                skips.append(x)
                    # x += acts[i - (int((len(self.blocks)) / 2))]
            
            # --- Conditioning: per-block offset OR group transform ---
            c_block = c
            if self.block_group_conditioning:
                group_id = min(i // max(self.depth // self.num_cond_groups, 1),
                               self.num_cond_groups - 1)
                c_block = self.cond_transforms[group_id](c)
            elif self.per_block_conditioning:
                c_block = c + self.block_cond_offsets[i]

            # Apply layer drop
            if self.layer_drop and self.training and layer_drop_mask is not None:
                # Check if any sample in the batch should drop this layer
                batch_drop_mask = layer_drop_mask[:, i]
                if batch_drop_mask.any():
                    # For samples that drop this layer, keep input unchanged
                    x_before = x.clone()
                    x_after = block(x, c_block)

                    # Create a mask for blending (avoid inplace operation)
                    # batch_drop_mask: True means DROP, False means KEEP
                    # We want to keep x_before for dropped samples, x_after for others
                    blend_mask = (~batch_drop_mask).float().view(N, 1, 1)  # (N, 1, 1)
                    x = x_before * (1 - blend_mask) + x_after * blend_mask
                else:
                    # No samples drop this layer
                    x = block(x, c_block)
            elif self.skipped_layers_eval is not None and (i + 1) == self.skipped_layers_eval:
                x = x
            else:
                x = block(x, c_block)

            # --- Gradient Isolation ---
            if self.gradient_isolation and self.training and i in self.gradient_isolation_layers:
                x = gradient_isolate(x, self.gradient_isolation_alpha)

            # --- Block-wise Auxiliary Heads ---
            if self.block_aux_heads_enabled and self.training and self.block_aux_head_projs is not None:
                if i in self.block_aux_head_layers:
                    aux_out = self.block_aux_head_projs[str(i)](x)  # (N, T, out_dim)
                    aux_outputs[i] = aux_out

            # if self.skipped_layers_eval is not None and (i + 1) == self.skipped_layers_eval:
            #     x = x
            # else:
            #     x = block(x, c)                     # (N, T, D)
            acts.append(x)
            if self.projection:
                if (i + 1) == self.encoder_depth:
                    if self.irepa_conv_projection:
                        # Convert to spatial format for conv projection
                        # x: (N, T, D) -> (N, D, H, W)
                        # H = W = int(T ** 0.5)
                        # x_spatial = x.reshape(N, H, W, D).permute(0, 3, 1, 2)  # (N, D, H, W)
                        # zs = []
                        # for projector in self.projectors:
                        #     z_spatial = projector(x_spatial)  # (N, z_dim, H, W)
                        #     z = z_spatial.permute(0, 2, 3, 1).reshape(N, T, -1)  # (N, T, z_dim)
                        #     zs.append(z)
                        zs = [projector(x) for projector in self.projectors] if not self.eval_mode else None
                    else:
                        # Standard MLP projection
                        zs = [projector(x.reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors] if not self.eval_mode else None
                    if self.sara:
                        zs_sara = [similarity(z) for z in zs]
                    # Apply spatial normalization to target representations if enabled
                    if self.irepa_spatial_norm:
                        zs = [self.spatial_norm(z) for z in zs]
            if self.auc_mse_prediction and (i + 1) == self.encoder_depth:
                mid_x = self.mid_final_layer(x, c)
            # multiple projection
            if self.multiple_projection:
                if (i + 1) in self.multiple_projection_blocks:
                    projector_index = self.multiple_projection_blocks.index(i+1)
                    zs_current_block = self.multiple_projectors[projector_index]((x.reshape(-1, D)).reshape(N, T, -1))
                    zs_multiple_blocks.append(zs_current_block)
            if self.diff_projection:
                if (i + 1) in self.diff_projection_blocks:
                    projector_index = self.diff_projection_blocks.index(i+1)
                    zs_current_block = self.multiple_diff_projectors[projector_index]((x.reshape(-1, D)).reshape(N, T, -1))
                    zs_multiple_diff_blocks.append(zs_current_block)
            if disperse_loss_block is not None:
                if (i + 1) == disperse_loss_block:
                    disperse_feature = x
                    # disperse_feature = disperse_feature.reshape(x.shape[0], -1)
            if ad is not None and self.sra:
                if (i + 1) == ad:
                    if self.training:
                        xr = self.ap_head(x)
                    else:
                        xr = x
            if self.teacher_representation_loss and (i + 1) == teacher_block:
                if self.training:
                    x_teacher = self.teacher_head(x)
                else:
                    x_teacher = x
            if self.cka or self.feature_dissimilarity or self.block_difference_loss or self.block_diversity_loss:
                block_feas[i] = x 
        if self.cls_token_loss:
            x, cls_token = self.final_layer(x, c, cls=cls_token)
        else:
            x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        if self.auc_mse_prediction:
            mid_x = self.unpatchify(mid_x)
        # denoising loss
        result = {'x': x}
        # all activations
        result['acts'] = acts
        # projection loss
        if self.projection:
            result['zs'] = zs
        if self.sara:
            result['zs_sara'] = zs_sara
        # disperse loss
        if disperse_loss_block is not None:
            result['disperse_feature'] = disperse_feature
        # sra loss
        if ad is not None and self.sra:
            result['xr'] = xr
            result['labels_train'] = labels_train
        # cka similarity
        if self.cka or self.feature_dissimilarity or self.block_difference_loss or self.block_diversity_loss:
            result['block_feas'] = block_feas
        if self.cls_token_loss:
            result['cls_token'] = cls_token
        if self.teacher_representation_loss and teacher_block is not None:
            result['x_teacher'] = x_teacher
        if self.multiple_projection:
            result['zs_multiple_blocks'] = zs_multiple_blocks
        if self.diff_projection:
            result['zs_multiple_diff_blocks'] = zs_multiple_diff_blocks
        if self.auc_mse_prediction:
            result['mid_x'] = mid_x
        # block-wise auxiliary head outputs
        if self.block_aux_heads_enabled and len(aux_outputs) > 0:
            result['aux_outputs'] = aux_outputs
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
    return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def SiT_S_8(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SiT_models = {
    'SiT-XL/2': SiT_XL_2,  'SiT-XL/4': SiT_XL_4,  'SiT-XL/8': SiT_XL_8,
    'SiT-L/2':  SiT_L_2,   'SiT-L/4':  SiT_L_4,   'SiT-L/8':  SiT_L_8,
    'SiT-B/2':  SiT_B_2,   'SiT-B/4':  SiT_B_4,   'SiT-B/8':  SiT_B_8,
    'SiT-S/2':  SiT_S_2,   'SiT-S/4':  SiT_S_4,   'SiT-S/8':  SiT_S_8,
}