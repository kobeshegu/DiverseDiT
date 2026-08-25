import argparse
import copy
from copy import deepcopy
import logging
import math
import os
from pathlib import Path
from collections import OrderedDict
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.sit import SiT_models
from models.trajectory import (
    ContextualDepthJEPAPredictor,
    MaskedJEPAPredictor,
    TrajectoryDINOLoss,
    TrajectoryEncoder,
    TrajectoryPatchEncoder,
    TrajectorySampler,
    flatten_trajectory,
    interpolate_trajectory,
    masked_jepa_loss,
    repeat_labels_for_trajectory,
    sample_block_patch_mask,
    sample_patch_mask,
    sample_stratified_timesteps,
    symmetric_infonce_loss,
    trajectory_hierarchical_contrastive_loss,
    trajectory_patch_infonce_loss,
    trajectory_patch_loss,
    vicreg_loss,
)
from loss import SILoss
from utils import load_encoders

from dataset import CustomDataset
from diffusers.models import AutoencoderKL
# import wandb_utils
import wandb
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


def safe_scalar(x, accelerator=None):
    '''
    avoid int.mean() and .detach() error
    if x is a tensor and accelerator is not None, then gather x
    if x has a mean method, then mean x
    if x has a detach method, then detach x and return the item
    otherwise, return x
    '''
    if hasattr(x, 'device') and accelerator is not None:
        x = accelerator.gather(x)
    if hasattr(x, 'mean'):
        x = x.mean()
    if hasattr(x, 'detach'):
        return x.detach().item()
    return x

@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device
    
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z 


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def diversity_warmup(step, warmup_steps=10000):
    """
    Linear warmup coefficient for diversity mechanisms.
    Returns 0→1 over warmup_steps, then stays at 1.
    Allows the network to learn basic denoising first before
    diversity pressure is fully applied.
    """
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, step / warmup_steps)


def cosine_decay_scale(step, start_step=-1, end_step=-1, min_scale=0.0):
    """Decay from 1 to min_scale between start_step and end_step."""
    if start_step < 0 or end_step < 0:
        return 1.0
    if step <= start_step:
        return 1.0
    if step >= end_step:
        return min_scale
    progress = (step - start_step) / (end_step - start_step)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_scale + (1 - min_scale) * cosine


def parse_float_list(value):
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    return [float(x.strip()) for x in value.split(',') if x.strip()]


def parse_int_list(value):
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(x.strip()) for x in value.split(',') if x.strip()]


def parse_depth_pairs(value):
    pairs = []
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        parts = item.split(':')
        if len(parts) != 2:
            raise ValueError(
                f"Invalid depth pair '{item}'; expected SOURCE:TARGET"
            )
        pairs.append((int(parts[0]), int(parts[1])))
    return pairs


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )
    # Import DDPKwargs for handling unused parameters in DDP
    from accelerate import DistributedDataParallelKwargs
    # Configure DDP to handle unused parameters
    # This is needed when some model components (like wavelet_skip_connections, irepa_projection) 
    # may not be used depending on configuration
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs]
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        # EMA modules are not wrapped by DDP, so all ranks must initialize them identically.
        set_seed(args.seed)
    
    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    use_repa = args.enc_type is not None and args.enc_type.strip().lower() not in {"", "none", "null"}
    if not use_repa and args.proj_coeff != 0:
        raise ValueError("--proj-coeff must be 0 when --enc-type=none")
    if args.self_flow and use_repa:
        raise ValueError("Self-Flow baseline must use --enc-type=none")
    if accelerator.is_main_process:
        logger.info(
            f"REPA {'enabled' if use_repa else 'disabled'} "
            f"(enc_type={args.enc_type}, proj_coeff={args.proj_coeff})"
        )
    if use_repa:
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type,
            device,
            args.resolution,
            checkpoint_dir=args.encoder_checkpoint_dir,
            )
    else:
        encoders, encoder_types, architectures = [], [], []
    z_dims = [encoder.embed_dim for encoder in encoders]
    if args.self_flow:
        model_hidden_sizes = {
            "SiT-S/2": 384, "SiT-S/4": 384, "SiT-S/8": 384,
            "SiT-B/2": 768, "SiT-B/4": 768, "SiT-B/8": 768,
            "SiT-L/2": 1024, "SiT-L/4": 1024, "SiT-L/8": 1024,
            "SiT-XL/2": 1152, "SiT-XL/4": 1152, "SiT-XL/8": 1152,
        }
        z_dims = [model_hidden_sizes[args.model]]
    # Parse comma-separated layer lists
    gradient_isolation_layers = None
    if args.gradient_isolation and args.gradient_isolation_layers:
        gradient_isolation_layers = [int(x.strip()) for x in args.gradient_isolation_layers.split(',')]
    block_aux_head_layers = None
    if args.block_aux_heads and args.block_aux_head_layers:
        block_aux_head_layers = [int(x.strip()) for x in args.block_aux_head_layers.split(',')]

    # Any of the new losses that need block_feas collection
    needs_block_feas = (
        args.block_diversity_loss or args.block_contrastive_loss
        or args.block_barlow_twins_loss or args.block_vicreg_loss
    )

    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=(args.cfg_prob > 0),
        z_dims=z_dims,
        encoder_depth=args.encoder_depth,
        skip_layer_connection=args.skip_layer_connection,
        cross_layer_connection=args.cross_layer_connection,
        block_diversity_loss=needs_block_feas,
        # layer drop
        layer_drop=args.layer_drop,
        layer_drop_rate=args.layer_drop_rate,
        layer_drop_strategy=args.layer_drop_strategy,
        layer_drop_type=args.layer_drop_type,
        # gradient isolation
        gradient_isolation=args.gradient_isolation,
        gradient_isolation_alpha=args.gradient_isolation_alpha,
        gradient_isolation_layers=gradient_isolation_layers,
        # block shuffling
        block_shuffling=args.block_shuffling,
        block_shuffling_prob=args.block_shuffling_prob,
        block_shuffling_group_size=args.block_shuffling_group_size,
        # per-block conditioning
        per_block_conditioning=args.per_block_conditioning,
        # block-wise auxiliary heads
        block_aux_heads=args.block_aux_heads,
        block_aux_head_layers=block_aux_head_layers,
        # structured heterogeneity
        residual_scaling=args.residual_scaling,
        block_group_conditioning=args.block_group_conditioning,
        num_cond_groups=args.num_cond_groups,
        heterogeneous_mlp=args.heterogeneous_mlp,
        alternating_heads=args.alternating_heads,
        depth_aware_init=args.depth_aware_init,
        **block_kwargs
    )

    model = model.to(device)
    if args.self_flow:
        if not (0 < args.self_flow_mask_ratio <= 0.5):
            raise ValueError("--self-flow-mask-ratio must be in (0, 0.5]")
        if not (1 <= args.encoder_depth < args.self_flow_teacher_depth <= model.depth):
            raise ValueError(
                "Self-Flow requires 1 <= --encoder-depth < "
                "--self-flow-teacher-depth <= model depth"
            )
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    model_num_patches = model.x_embedder.num_patches
    patch_objectives = {"patch", "patch_infonce", "hierarchical"}
    jepa_objectives = {"masked_jepa", "contextual_jepa"}
    token_objectives = patch_objectives | jepa_objectives
    repr_depths = []
    repr_depth_pairs = []
    repr_feature_depths = []
    trajectory_encoder = None
    trajectory_encoder_ema = None
    trajectory_dino_loss = None
    trajectory_sampler = None
    if args.traj_loss:
        traj_anchors = parse_float_list(args.traj_anchors)
        if args.traj_num_steps != len(traj_anchors):
            raise ValueError(
                f"--traj-num-steps={args.traj_num_steps} must match "
                f"--traj-anchors length={len(traj_anchors)}"
            )
        if args.traj_objective in patch_objectives and args.traj_num_steps < 2:
            raise ValueError(
                f"--traj-objective={args.traj_objective} requires at least two trajectory anchors"
            )
        if args.traj_objective == "hierarchical" and args.traj_num_steps != 3:
            raise ValueError("--traj-objective=hierarchical requires exactly three anchors")
        if args.traj_objective in {"patch_infonce", "hierarchical"}:
            if args.traj_patch_num_samples <= 0:
                raise ValueError("--traj-patch-num-samples must be positive")
            if args.traj_patch_temperature <= 0:
                raise ValueError("--traj-patch-temperature must be positive")
            if args.traj_global_temperature <= 0:
                raise ValueError("--traj-global-temperature must be positive")
            if not (-1.0 <= args.traj_patch_negative_sim_threshold <= 1.0):
                raise ValueError(
                    "--traj-patch-negative-sim-threshold must be in [-1, 1]"
                )
            if args.traj_patch_min_negatives < 0:
                raise ValueError("--traj-patch-min-negatives must be non-negative")
        if args.traj_objective == "hierarchical":
            hierarchy_coeffs = {
                "--traj-patch-nce-coeff": args.traj_patch_nce_coeff,
                "--traj-global-nce-coeff": args.traj_global_nce_coeff,
                "--traj-patch-positive-coeff": args.traj_patch_positive_coeff,
                "--traj-patch-std-coeff": args.traj_patch_std_coeff,
                "--traj-patch-cov-coeff": args.traj_patch_cov_coeff,
            }
            for name, value in hierarchy_coeffs.items():
                if value < 0:
                    raise ValueError(f"{name} must be non-negative")
        if args.traj_objective in jepa_objectives:
            if args.traj_objective == "contextual_jepa":
                repr_depth_pairs = parse_depth_pairs(args.repr_depth_pairs)
                if not repr_depth_pairs:
                    raise ValueError(
                        "--repr-depth-pairs must contain at least one SOURCE:TARGET pair"
                    )
                if len(repr_depth_pairs) != len(set(repr_depth_pairs)):
                    raise ValueError("--repr-depth-pairs must not contain duplicates")
                repr_depths = sorted({depth for pair in repr_depth_pairs for depth in pair})
            else:
                repr_depths = parse_int_list(args.repr_depths)
                if not repr_depths:
                    raise ValueError("--repr-depths must contain at least one block depth")
                repr_depth_pairs = [(depth, depth) for depth in repr_depths]
            invalid_depths = [depth for depth in repr_depths if not (1 <= depth <= model.depth)]
            if invalid_depths:
                raise ValueError(
                    f"JEPA configuration contains invalid depths {invalid_depths}; "
                    f"model depth is {model.depth}"
                )
            repr_feature_depths = sorted(set(repr_depths))
            if not (0 < args.repr_mask_ratio < 1):
                raise ValueError("--repr-mask-ratio must be in (0, 1)")
            if args.traj_objective == "contextual_jepa":
                if not (0 <= args.repr_timestep_min < args.repr_timestep_max <= 1):
                    raise ValueError(
                        "contextual JEPA requires "
                        "0 <= --repr-timestep-min < --repr-timestep-max <= 1"
                    )
                if args.repr_mask_num_blocks <= 0:
                    raise ValueError("--repr-mask-num-blocks must be positive")
            else:
                if not (0 <= args.repr_timestep <= 1):
                    raise ValueError("--repr-timestep must be in [0, 1]")
                if not (0 <= args.repr_timestep_jitter <= 1):
                    raise ValueError("--repr-timestep-jitter must be in [0, 1]")
            if args.repr_relational_coeff < 0 or args.repr_variance_coeff < 0:
                raise ValueError("masked JEPA loss coefficients must be non-negative")
            if args.repr_predictor_hidden_dim <= 0:
                raise ValueError("--repr-predictor-hidden-dim must be positive")
        if args.traj_decay_start >= 0 or args.traj_decay_end >= 0:
            if not (0 <= args.traj_decay_start < args.traj_decay_end):
                raise ValueError(
                    "trajectory decay requires "
                    "0 <= --traj-decay-start < --traj-decay-end"
                )
        if not (0 <= args.traj_min_loss_scale <= 1):
            raise ValueError("--traj-min-loss-scale must be in [0, 1]")
        if not (0 <= args.traj_ema_decay < 1):
            raise ValueError("--traj-ema-decay must be in [0, 1)")
        model_hidden_size = getattr(model, "hidden_size", model.pos_embed.shape[-1])
        if args.traj_objective == "contextual_jepa":
            trajectory_encoder = ContextualDepthJEPAPredictor(
                in_dim=model_hidden_size,
                depth_pairs=repr_depth_pairs,
                hidden_dim=args.repr_predictor_hidden_dim,
            ).to(device)
        elif args.traj_objective == "masked_jepa":
            trajectory_encoder = MaskedJEPAPredictor(
                in_dim=model_hidden_size,
                hidden_dim=args.repr_predictor_hidden_dim,
            ).to(device)
        elif args.traj_objective in patch_objectives:
            trajectory_encoder = TrajectoryPatchEncoder(
                in_dim=model_hidden_size,
                embed_dim=args.traj_embed_dim,
                out_dim=args.traj_out_dim,
            ).to(device)
        else:
            trajectory_encoder = TrajectoryEncoder(
                in_dim=model_hidden_size,
                embed_dim=args.traj_embed_dim,
                out_dim=args.traj_out_dim,
                num_layers=args.traj_encoder_layers,
                num_heads=args.traj_encoder_heads,
                mlp_ratio=args.traj_encoder_mlp_ratio,
                dropout=args.traj_encoder_dropout,
            ).to(device)
        if args.traj_objective not in jepa_objectives:
            trajectory_encoder_ema = deepcopy(trajectory_encoder).to(device)
            requires_grad(trajectory_encoder_ema, False)
            trajectory_encoder_ema.eval()
        trajectory_sampler = TrajectorySampler(
            anchors=traj_anchors,
            base_jitter=args.traj_base_jitter,
            view_jitter=args.traj_view_jitter,
            min_gap=args.traj_min_gap,
            sampler=args.traj_sampler,
            semantic_bins=args.traj_semantic_bins,
            semantic_mix=args.traj_semantic_mix,
            semantic_temperature=args.traj_semantic_temperature,
            semantic_momentum=args.traj_semantic_momentum,
            semantic_warmup_steps=args.traj_semantic_warmup_steps,
            semantic_min_t=args.traj_semantic_min_t,
            semantic_max_t=args.traj_semantic_max_t,
        )
        if args.traj_objective == "dino":
            trajectory_dino_loss = TrajectoryDINOLoss(
                out_dim=args.traj_out_dim,
                student_temp=args.traj_student_temp,
                teacher_temp=args.traj_teacher_temp,
                center_momentum=args.traj_center_momentum,
            ).to(device)
    pretrained_vae_path = f"{args.pretrained_model_path}/sd-vae-ft-{args.vae}"
    vae = AutoencoderKL.from_pretrained(pretrained_vae_path).to(device)
    requires_grad(ema, False)
    
    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)

    # create loss function
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type,
        encoders=encoders,
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting,
        block_diversity_loss=args.block_diversity_loss,
        projection=True,  # default = True for REPA
        encoder_depth=args.encoder_depth,
        # new diversity losses
        block_contrastive_loss=args.block_contrastive_loss,
        block_contrastive_temperature=args.block_contrastive_temperature,
        block_barlow_twins_loss=args.block_barlow_twins_loss,
        block_barlow_lambda=args.block_barlow_lambda,
        block_vicreg_loss=args.block_vicreg_loss,
        block_vicreg_lambda=args.block_vicreg_lambda,
        block_vicreg_mu=args.block_vicreg_mu,
        block_vicreg_nu=args.block_vicreg_nu,
        self_flow=args.self_flow,
        self_flow_mask_ratio=args.self_flow_mask_ratio,
        self_flow_teacher_depth=args.self_flow_teacher_depth,
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    trainable_params = list(model.parameters())
    if args.traj_loss:
        trainable_params += list(trajectory_encoder.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    

    if args.seed is not None:
        # Use rank-specific randomness for data order, noise, and timestep sampling.
        set_seed(args.seed + accelerator.process_index)
    
    # Setup data:
    train_dataset = CustomDataset(args.data_dir)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            weights_only=False,
            )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        if args.traj_loss and 'trajectory_encoder' in ckpt:
            trajectory_encoder.load_state_dict(ckpt['trajectory_encoder'])
            if trajectory_encoder_ema is not None:
                trajectory_encoder_ema.load_state_dict(
                    ckpt.get('trajectory_encoder_ema', ckpt['trajectory_encoder'])
                )
            if 'trajectory_sampler' in ckpt:
                trajectory_sampler.load_state_dict(ckpt['trajectory_sampler'])
            if trajectory_dino_loss is not None and 'trajectory_dino_loss' in ckpt:
                trajectory_dino_loss.load_state_dict(ckpt['trajectory_dino_loss'])
        try:
            optimizer.load_state_dict(ckpt['opt'])
        except ValueError:
            if args.traj_loss:
                logger.warning(
                    "Optimizer state does not match trajectory parameters; "
                    "continuing with a freshly initialized optimizer."
                )
            else:
                raise
        global_step = ckpt['steps']

    if args.traj_loss:
        model, trajectory_encoder, optimizer, train_dataloader = accelerator.prepare(
            model, trajectory_encoder, optimizer, train_dataloader
        )
    else:
        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )

    if accelerator.is_main_process and args.report_to != "none":
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="REPA", 
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )
        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    gt_xs = ys = xT = None
    if not args.skip_training_samples:
        # Fixed inputs used only for periodic qualitative samples.
        sample_batch_size = 64 // accelerator.num_processes
        gt_raw_images, gt_xs, _ = next(iter(train_dataloader))
        assert gt_raw_images.shape[-1] == args.resolution
        gt_xs = gt_xs[:sample_batch_size]
        gt_xs = sample_posterior(
            gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
            )
        ys = torch.randint(1000, size=(gt_xs.shape[0],), device=device)
        xT = torch.randn((ys.shape[0], 4, latent_size, latent_size), device=device)
        
    for epoch in range(args.epochs):
        model.train()
        for raw_image, x, y in train_dataloader:
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)
            z = None
            if args.legacy:
                # In our early experiments, we accidentally apply label dropping twice: 
                # once in train.py and once in sit.py. 
                # We keep this option for exact reproducibility with previous runs.
                drop_ids = torch.rand(y.shape[0], device=y.device) < args.cfg_prob
                labels = torch.where(drop_ids, args.num_classes, y)
            else:
                labels = y
            zs = None
            if use_repa:
                raw_image = raw_image.to(device)
                zs = []
                with torch.no_grad():
                    with accelerator.autocast():
                        for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                            raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                            z = encoder.forward_features(raw_image_)
                            if 'mocov3' in encoder_type: z = z[:, 1:]
                            if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                            zs.append(z)

            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)

            with accelerator.accumulate(model):
                model_kwargs = dict(y=labels)
                losses = loss_fn(
                    model,
                    x,
                    model_kwargs,
                    zs=zs,
                    teacher_model=ema if args.self_flow else None,
                )
                denoising_loss = losses.get('denoising_loss', 0)
                proj_loss = losses.get('proj_loss', 0)
                denoising_loss_mean = denoising_loss.mean()
                proj_loss_mean = proj_loss.mean()
                self_flow_rep_loss = losses.get('self_flow_rep_loss', 0)
                self_flow_rep_loss_mean = (
                    self_flow_rep_loss.mean()
                    if hasattr(self_flow_rep_loss, "mean")
                    else self_flow_rep_loss
                )
                block_diversity_loss = losses.get('block_diversity_loss', 0)

                # Diversity warmup: ramp up diversity pressure over warmup_steps
                warmup = diversity_warmup(global_step, args.diversity_warmup_steps)

                # block diversity loss coefficient (adaptive + warmup)
                if block_diversity_loss > 0.5:
                    block_diversity_loss_coeff = 1.0
                elif block_diversity_loss > 0.1:
                    block_diversity_loss_coeff = (block_diversity_loss - 0.1) / 0.5
                else:
                    block_diversity_loss_coeff = 0
                block_diversity_loss_coeff *= warmup

                # new diversity losses (all scaled by warmup)
                block_contrastive_loss_val = losses.get('block_contrastive_loss', 0)
                block_barlow_twins_loss_val = losses.get('block_barlow_twins_loss', 0)
                block_vicreg_loss_val = losses.get('block_vicreg_loss', 0)
                block_aux_loss = losses.get('block_aux_loss', 0)
                traj_loss_val = torch.zeros((), device=device, dtype=denoising_loss_mean.dtype)
                traj_pos_cos = torch.zeros_like(traj_loss_val)
                traj_neg_cos = torch.zeros_like(traj_loss_val)
                traj_student_std = torch.zeros_like(traj_loss_val)
                traj_teacher_std = torch.zeros_like(traj_loss_val)
                traj_teacher_entropy = torch.zeros_like(traj_loss_val)
                traj_patch_sim = torch.zeros_like(traj_loss_val)
                traj_patch_std = torch.zeros_like(traj_loss_val)
                traj_patch_cov = torch.zeros_like(traj_loss_val)
                traj_patch_nce = torch.zeros_like(traj_loss_val)
                traj_patch_nce_raw = torch.zeros_like(traj_loss_val)
                traj_patch_valid_negatives = torch.zeros_like(traj_loss_val)
                traj_mid_patch_pos_cos = torch.zeros_like(traj_loss_val)
                traj_mid_patch_neg_cos = torch.zeros_like(traj_loss_val)
                traj_high_global_nce = torch.zeros_like(traj_loss_val)
                traj_high_global_nce_raw = torch.zeros_like(traj_loss_val)
                traj_high_global_pos_cos = torch.zeros_like(traj_loss_val)
                traj_high_global_neg_cos = torch.zeros_like(traj_loss_val)
                traj_high_global_valid_negatives = torch.zeros_like(traj_loss_val)
                repr_masked_loss = torch.zeros_like(traj_loss_val)
                repr_masked_cos = torch.zeros_like(traj_loss_val)
                repr_unmasked_cos = torch.zeros_like(traj_loss_val)
                repr_relational_loss = torch.zeros_like(traj_loss_val)
                repr_variance_loss = torch.zeros_like(traj_loss_val)
                repr_student_std = torch.zeros_like(traj_loss_val)
                repr_teacher_std = torch.zeros_like(traj_loss_val)
                repr_student_effective_rank = torch.zeros_like(traj_loss_val)
                repr_teacher_effective_rank = torch.zeros_like(traj_loss_val)
                repr_timestep_mean = torch.zeros_like(traj_loss_val)
                repr_timestep_min = torch.zeros_like(traj_loss_val)
                repr_timestep_max = torch.zeros_like(traj_loss_val)
                repr_depth_metrics = {}
                traj_warmup = diversity_warmup(global_step, args.traj_warmup_steps)
                traj_decay_scale = cosine_decay_scale(
                    global_step,
                    args.traj_decay_start,
                    args.traj_decay_end,
                    args.traj_min_loss_scale,
                )
                traj_loss_weight = (
                    args.traj_loss_coeff * traj_warmup * traj_decay_scale
                )
                traj_computed = False

                if (
                    args.traj_loss
                    and traj_loss_weight > 0
                    and args.traj_loss_frequency > 0
                    and global_step % args.traj_loss_frequency == 0
                ):
                    traj_computed = True
                    traj_bsz = max(1, int(x.shape[0] * args.traj_batch_ratio))
                    traj_bsz = min(traj_bsz, x.shape[0])
                    if traj_bsz < x.shape[0]:
                        traj_indices = torch.randperm(x.shape[0], device=x.device)[:traj_bsz]
                        x_traj = x[traj_indices]
                        y_traj = labels[traj_indices]
                    else:
                        x_traj = x
                        y_traj = labels

                    if args.traj_objective in jepa_objectives:
                        if args.traj_objective == "contextual_jepa":
                            t_repr = sample_stratified_timesteps(
                                x_traj.shape[0],
                                args.repr_timestep_min,
                                args.repr_timestep_max,
                                x_traj.device,
                                x_traj.dtype,
                            )
                        else:
                            t_repr = torch.full(
                                (x_traj.shape[0],),
                                args.repr_timestep,
                                device=x_traj.device,
                                dtype=x_traj.dtype,
                            )
                            if args.repr_timestep_jitter > 0:
                                t_repr = t_repr + torch.empty_like(t_repr).uniform_(
                                    -args.repr_timestep_jitter,
                                    args.repr_timestep_jitter,
                                )
                        t_repr = t_repr.clamp(0, 1).unsqueeze(1)
                        repr_timestep_mean = t_repr.mean().detach()
                        repr_timestep_min = t_repr.min().detach()
                        repr_timestep_max = t_repr.max().detach()
                        t_a = t_b = t_repr
                        eps_a = torch.randn_like(x_traj)
                        eps_b = eps_a
                    else:
                        t_a, t_b = trajectory_sampler.sample_pair(
                            x_traj.shape[0],
                            device=x_traj.device,
                            dtype=x_traj.dtype,
                            step=global_step,
                        )
                        if args.traj_shared_noise_within_path:
                            eps_a = torch.randn_like(x_traj)
                            eps_b = torch.randn_like(x_traj)
                        else:
                            eps_shape = (
                                x_traj.shape[0],
                                args.traj_num_steps,
                            ) + tuple(x_traj.shape[1:])
                            eps_a = torch.randn(
                                eps_shape,
                                device=x_traj.device,
                                dtype=x_traj.dtype,
                            )
                            eps_b = torch.randn(
                                eps_shape,
                                device=x_traj.device,
                                dtype=x_traj.dtype,
                            )

                    if args.traj_objective in jepa_objectives:
                        t_student = t_a
                        t_teacher = t_b
                        student_noise = eps_a
                        teacher_noise = eps_b
                    elif args.traj_objective in patch_objectives:
                        t_student = t_a[:, :-1]
                        t_teacher = t_b[:, -1:]
                        student_noise = eps_a if eps_a.ndim == x_traj.ndim else eps_a[:, :-1]
                        teacher_noise = eps_b if eps_b.ndim == x_traj.ndim else eps_b[:, -1:]
                    else:
                        t_student = t_a
                        t_teacher = t_b
                        student_noise = eps_a
                        teacher_noise = eps_b

                    x_a = interpolate_trajectory(x_traj, student_noise, t_student, loss_fn)
                    x_b = interpolate_trajectory(x_traj, teacher_noise, t_teacher, loss_fn)
                    y_student = repeat_labels_for_trajectory(y_traj, t_student.shape[1])
                    y_teacher = repeat_labels_for_trajectory(y_traj, t_teacher.shape[1])
                    repr_mask = None
                    if args.traj_objective in jepa_objectives:
                        if args.traj_objective == "contextual_jepa":
                            repr_mask = sample_block_patch_mask(
                                x_traj.shape[0],
                                model_num_patches,
                                args.repr_mask_ratio,
                                x_traj.device,
                                num_blocks=args.repr_mask_num_blocks,
                            )
                        else:
                            repr_mask = sample_patch_mask(
                                x_traj.shape[0],
                                model_num_patches,
                                args.repr_mask_ratio,
                                x_traj.device,
                            )

                    student_outputs = model(
                        flatten_trajectory(x_a),
                        t_student.reshape(-1),
                        y_student,
                        return_features=True,
                        feature_depth=(
                            None if args.traj_objective in jepa_objectives
                            else args.traj_depth
                        ),
                        feature_depths=(
                            repr_feature_depths
                            if args.traj_objective in jepa_objectives
                            else None
                        ),
                        input_mask=repr_mask,
                    )

                    with torch.no_grad():
                        teacher_outputs = ema(
                            flatten_trajectory(x_b),
                            t_teacher.reshape(-1),
                            y_teacher,
                            return_features=True,
                            feature_depth=(
                                None if args.traj_objective in jepa_objectives
                                else args.traj_depth
                            ),
                            feature_depths=(
                                repr_feature_depths
                                if args.traj_objective in jepa_objectives
                                else None
                            ),
                        )
                    if args.traj_objective in jepa_objectives:
                        h_a_by_depth = student_outputs["features"]
                        h_b_by_depth = teacher_outputs["features"]
                    else:
                        h_a = student_outputs['features'].reshape(
                            x_traj.shape[0],
                            t_student.shape[1],
                            student_outputs['features'].shape[1],
                            student_outputs['features'].shape[2],
                        )
                        h_b = teacher_outputs['features'].reshape(
                            x_traj.shape[0],
                            t_teacher.shape[1],
                            teacher_outputs['features'].shape[1],
                            teacher_outputs['features'].shape[2],
                        )

                    if args.traj_objective in jepa_objectives:
                        depth_losses = []
                        last_prediction = None
                        last_teacher = None
                        for source_depth, target_depth in repr_depth_pairs:
                            pair_key = f"{source_depth}_to_{target_depth}"
                            if args.traj_objective == "contextual_jepa":
                                prediction = trajectory_encoder(
                                    h_a_by_depth[source_depth],
                                    source_depth,
                                    target_depth,
                                )
                            else:
                                prediction = trajectory_encoder(
                                    h_a_by_depth[source_depth]
                                )
                            teacher_feature = h_b_by_depth[target_depth]
                            depth_loss, depth_components = masked_jepa_loss(
                                prediction,
                                teacher_feature,
                                repr_mask,
                                relational_coeff=args.repr_relational_coeff,
                                variance_coeff=args.repr_variance_coeff,
                            )
                            depth_losses.append(depth_loss)
                            repr_depth_metrics[pair_key] = depth_components
                            last_prediction = prediction
                            last_teacher = teacher_feature

                        traj_loss_val = torch.stack(depth_losses).mean()
                        component_keys = (
                            "masked_loss",
                            "masked_cosine",
                            "unmasked_cosine",
                            "relational",
                            "variance",
                            "student_std",
                            "teacher_std",
                            "student_effective_rank",
                            "teacher_effective_rank",
                        )
                        averaged_components = {
                            key: torch.stack(
                                [
                                    repr_depth_metrics[
                                        f"{source_depth}_to_{target_depth}"
                                    ][key]
                                    for source_depth, target_depth in repr_depth_pairs
                                ]
                            ).mean()
                            for key in component_keys
                        }
                        repr_masked_loss = averaged_components["masked_loss"]
                        repr_masked_cos = averaged_components["masked_cosine"]
                        repr_unmasked_cos = averaged_components["unmasked_cosine"]
                        repr_relational_loss = averaged_components["relational"]
                        repr_variance_loss = averaged_components["variance"]
                        repr_student_std = averaged_components["student_std"]
                        repr_teacher_std = averaged_components["teacher_std"]
                        repr_student_effective_rank = averaged_components[
                            "student_effective_rank"
                        ]
                        repr_teacher_effective_rank = averaged_components[
                            "teacher_effective_rank"
                        ]
                        z_a = last_prediction.unsqueeze(1)
                        z_b = last_teacher.unsqueeze(1)
                    elif args.traj_objective == "patch":
                        z_a = trajectory_encoder(h_a, t_student, predict=True)
                        with torch.no_grad():
                            z_b = trajectory_encoder_ema(h_b, t_teacher, predict=False)
                        traj_loss_val, patch_components = trajectory_patch_loss(
                            z_a,
                            z_b,
                            sim_coeff=args.traj_patch_sim_coeff,
                            std_coeff=args.traj_patch_std_coeff,
                            cov_coeff=args.traj_patch_cov_coeff,
                        )
                        traj_patch_sim = patch_components["sim"]
                        traj_patch_std = patch_components["std"]
                        traj_patch_cov = patch_components["cov"]
                        trajectory_sampler.update_semantic_scores(h_a.detach(), t_student)
                    elif args.traj_objective == "patch_infonce":
                        z_a = trajectory_encoder(h_a, t_student, predict=True)
                        with torch.no_grad():
                            z_b = trajectory_encoder_ema(h_b, t_teacher, predict=False)
                        traj_loss_val, patch_components = trajectory_patch_infonce_loss(
                            z_a,
                            z_b,
                            accelerator=accelerator,
                            num_patches=args.traj_patch_num_samples,
                            temperature=args.traj_patch_temperature,
                            negative_sim_threshold=args.traj_patch_negative_sim_threshold,
                            min_negatives=args.traj_patch_min_negatives,
                            nce_coeff=args.traj_patch_nce_coeff,
                            std_coeff=args.traj_patch_std_coeff,
                            cov_coeff=args.traj_patch_cov_coeff,
                        )
                        traj_patch_nce = patch_components["nce"]
                        traj_patch_nce_raw = patch_components["nce_raw"]
                        traj_patch_sim = patch_components["sim"]
                        traj_patch_std = patch_components["std"]
                        traj_patch_cov = patch_components["cov"]
                        traj_patch_valid_negatives = patch_components["valid_negatives"]
                        trajectory_sampler.update_semantic_scores(h_a.detach(), t_student)
                    elif args.traj_objective == "hierarchical":
                        z_a = trajectory_encoder(h_a, t_student, predict=True)
                        with torch.no_grad():
                            z_b = trajectory_encoder_ema(h_b, t_teacher, predict=False)
                        traj_loss_val, hierarchy_components = (
                            trajectory_hierarchical_contrastive_loss(
                                z_a,
                                z_b,
                                accelerator=accelerator,
                                num_patches=args.traj_patch_num_samples,
                                patch_temperature=args.traj_patch_temperature,
                                global_temperature=args.traj_global_temperature,
                                negative_sim_threshold=args.traj_patch_negative_sim_threshold,
                                min_negatives=args.traj_patch_min_negatives,
                                patch_nce_coeff=args.traj_patch_nce_coeff,
                                global_nce_coeff=args.traj_global_nce_coeff,
                                positive_coeff=args.traj_patch_positive_coeff,
                                std_coeff=args.traj_patch_std_coeff,
                                cov_coeff=args.traj_patch_cov_coeff,
                            )
                        )
                        traj_patch_nce = hierarchy_components["patch_nce"]
                        traj_patch_nce_raw = hierarchy_components["patch_nce_raw"]
                        traj_patch_sim = hierarchy_components["patch_sim"]
                        traj_patch_std = hierarchy_components["std"]
                        traj_patch_cov = hierarchy_components["cov"]
                        traj_patch_valid_negatives = hierarchy_components[
                            "patch_valid_negatives"
                        ]
                        traj_mid_patch_pos_cos = hierarchy_components[
                            "patch_positive_cosine"
                        ]
                        traj_mid_patch_neg_cos = hierarchy_components[
                            "patch_negative_cosine"
                        ]
                        traj_high_global_nce = hierarchy_components["global_nce"]
                        traj_high_global_nce_raw = hierarchy_components[
                            "global_nce_raw"
                        ]
                        traj_high_global_pos_cos = hierarchy_components[
                            "global_positive_cosine"
                        ]
                        traj_high_global_neg_cos = hierarchy_components[
                            "global_negative_cosine"
                        ]
                        traj_high_global_valid_negatives = hierarchy_components[
                            "global_valid_negatives"
                        ]
                        trajectory_sampler.update_semantic_scores(h_a.detach(), t_student)
                    elif args.traj_objective == "dino":
                        z_a = trajectory_encoder(h_a, t_student, normalize=False)
                        with torch.no_grad():
                            z_b = trajectory_encoder_ema(h_b, t_teacher, normalize=False)
                        trajectory_sampler.update_semantic_scores(h_b, t_teacher)
                        traj_loss_val = trajectory_dino_loss(z_a, z_b, accelerator=accelerator)
                    elif args.traj_objective == "vicreg":
                        z_a = trajectory_encoder(h_a, t_student, normalize=False)
                        with torch.no_grad():
                            z_b = trajectory_encoder_ema(h_b, t_teacher, normalize=False)
                        trajectory_sampler.update_semantic_scores(h_b, t_teacher)
                        traj_loss_val = vicreg_loss(
                            z_a,
                            z_b.detach(),
                            sim_coeff=args.traj_vicreg_sim_coeff,
                            std_coeff=args.traj_vicreg_std_coeff,
                            cov_coeff=args.traj_vicreg_cov_coeff,
                        )
                    elif args.traj_objective == "infonce":
                        z_a = trajectory_encoder(h_a, t_student, normalize=False)
                        with torch.no_grad():
                            z_b = trajectory_encoder_ema(h_b, t_teacher, normalize=False)
                        trajectory_sampler.update_semantic_scores(h_b, t_teacher)
                        traj_loss_val = symmetric_infonce_loss(
                            z_a,
                            z_b.detach(),
                            temperature=args.traj_infonce_temperature,
                        )
                    else:
                        raise ValueError(f"Unsupported trajectory objective: {args.traj_objective}")

                    with torch.no_grad():
                        z_a_norm = F.normalize(z_a.float(), dim=-1)
                        z_b_norm = F.normalize(z_b.float(), dim=-1)
                        if args.traj_objective in token_objectives:
                            z_b_norm = z_b_norm.expand(-1, z_a_norm.shape[1], -1, -1)
                        traj_pos_cos = (z_a_norm * z_b_norm).sum(dim=-1).mean()
                        if z_a.shape[0] > 1:
                            traj_neg_cos = (z_a_norm * z_b_norm.roll(1, dims=0)).sum(dim=-1).mean()
                        if args.traj_objective in token_objectives:
                            student_pooled = z_a.float().mean(dim=2)
                            teacher_pooled = z_b.float().mean(dim=2)
                            traj_student_std = student_pooled.reshape(-1, student_pooled.shape[-1]).std(
                                dim=0, unbiased=False
                            ).mean()
                            traj_teacher_std = teacher_pooled.reshape(-1, teacher_pooled.shape[-1]).std(
                                dim=0, unbiased=False
                            ).mean()
                        else:
                            traj_student_std = z_a.float().std(dim=0, unbiased=False).mean()
                            traj_teacher_std = z_b.float().std(dim=0, unbiased=False).mean()
                        if trajectory_dino_loss is not None:
                            teacher_probs = F.softmax(
                                (z_b.float() - trajectory_dino_loss.center.float())
                                / trajectory_dino_loss.teacher_temp,
                                dim=-1,
                            )
                            traj_teacher_entropy = -(
                                teacher_probs * teacher_probs.clamp_min(1e-12).log()
                            ).sum(dim=-1).mean()

                loss = denoising_loss_mean + proj_loss_mean * args.proj_coeff \
                    + self_flow_rep_loss_mean * args.self_flow_rep_coeff \
                    + block_diversity_loss * block_diversity_loss_coeff \
                    + block_contrastive_loss_val * args.block_contrastive_loss_coeff * warmup \
                    + block_barlow_twins_loss_val * args.block_barlow_twins_loss_coeff * warmup \
                    + block_vicreg_loss_val * args.block_vicreg_loss_coeff * warmup \
                    + block_aux_loss * args.block_aux_head_coeff * warmup \
                    + traj_loss_val * traj_loss_weight
                    
                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = list(model.parameters())
                    if args.traj_loss:
                        params_to_clip += list(trajectory_encoder.parameters())
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model) # change ema function
                    if trajectory_encoder_ema is not None:
                        update_ema(
                            trajectory_encoder_ema,
                            trajectory_encoder,
                            decay=args.traj_ema_decay,
                        )
            
            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1                
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    checkpoint = {
                        "model": unwrapped_model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    if args.traj_loss:
                        checkpoint["trajectory_encoder"] = accelerator.unwrap_model(trajectory_encoder).state_dict()
                        if trajectory_encoder_ema is not None:
                            checkpoint["trajectory_encoder_ema"] = trajectory_encoder_ema.state_dict()
                        checkpoint["trajectory_sampler"] = trajectory_sampler.state_dict()
                        if trajectory_dino_loss is not None:
                            checkpoint["trajectory_dino_loss"] = trajectory_dino_loss.state_dict()
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if (
                not args.skip_training_samples
                and (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0))
            ):
                from samplers import euler_sampler
                with torch.no_grad():
                    samples = euler_sampler(
                        model, 
                        xT, 
                        ys,
                        num_steps=50, 
                        cfg_scale=4.0,
                        guidance_low=0.,
                        guidance_high=1.,
                        path_type=args.path_type,
                        heun=False,
                    ).to(torch.float32)
                    samples = vae.decode((samples -  latents_bias) / latents_scale).sample
                    gt_samples = vae.decode((gt_xs - latents_bias) / latents_scale).sample
                    samples = (samples + 1) / 2.
                    gt_samples = (gt_samples + 1) / 2.
                out_samples = accelerator.gather(samples.to(torch.float32))
                gt_samples = accelerator.gather(gt_samples.to(torch.float32))
                accelerator.log({"samples": wandb.Image(array2grid(out_samples)),
                                 "gt_samples": wandb.Image(array2grid(gt_samples))})
                logging.info("Generating EMA samples done.")
            # save logs for monitoring the training process and grad norm
            logs = {
                "loss": accelerator.gather(denoising_loss_mean).mean().detach().item(),
                "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                "block_diversity_loss": safe_scalar(block_diversity_loss, accelerator),
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item(),
                "diversity_warmup": warmup,
            }
            if args.self_flow:
                logs["self_flow_rep_loss"] = safe_scalar(
                    self_flow_rep_loss_mean, accelerator
                )
                logs["self_flow_mask_fraction"] = safe_scalar(
                    losses["self_flow_mask_fraction"], accelerator
                )
                logs["self_flow_timestep_gap"] = safe_scalar(
                    losses["self_flow_timestep_gap"], accelerator
                )
                logs["self_flow_teacher_timestep"] = safe_scalar(
                    losses["self_flow_teacher_timestep"], accelerator
                )
            if args.block_contrastive_loss:
                logs["block_contrastive_loss"] = safe_scalar(block_contrastive_loss_val, accelerator)
            if args.block_barlow_twins_loss:
                logs["block_barlow_twins_loss"] = safe_scalar(block_barlow_twins_loss_val, accelerator)
            if args.block_vicreg_loss:
                logs["block_vicreg_loss"] = safe_scalar(block_vicreg_loss_val, accelerator)
            if args.block_aux_heads:
                logs["block_aux_loss"] = safe_scalar(block_aux_loss, accelerator)
            if args.traj_loss:
                logs["traj_loss"] = safe_scalar(traj_loss_val, accelerator)
                logs["traj_loss_weight"] = traj_loss_weight
                logs["traj_warmup"] = traj_warmup
                logs["traj_decay_scale"] = traj_decay_scale
                logs["traj_computed"] = float(traj_computed)
                logs["traj_pos_cos"] = safe_scalar(traj_pos_cos, accelerator)
                logs["traj_neg_cos"] = safe_scalar(traj_neg_cos, accelerator)
                logs["traj_student_std"] = safe_scalar(traj_student_std, accelerator)
                logs["traj_teacher_std"] = safe_scalar(traj_teacher_std, accelerator)
                logs["traj_teacher_entropy"] = safe_scalar(traj_teacher_entropy, accelerator)
                if args.traj_objective in patch_objectives:
                    logs["traj_patch_sim"] = safe_scalar(traj_patch_sim, accelerator)
                    logs["traj_patch_std"] = safe_scalar(traj_patch_std, accelerator)
                    logs["traj_patch_cov"] = safe_scalar(traj_patch_cov, accelerator)
                if args.traj_objective in {"patch_infonce", "hierarchical"}:
                    logs["traj_patch_nce"] = safe_scalar(traj_patch_nce, accelerator)
                    logs["traj_patch_nce_raw"] = safe_scalar(traj_patch_nce_raw, accelerator)
                    logs["traj_patch_valid_negatives"] = safe_scalar(
                        traj_patch_valid_negatives,
                        accelerator,
                    )
                if args.traj_objective == "hierarchical":
                    logs["traj_mid_patch_pos_cos"] = safe_scalar(
                        traj_mid_patch_pos_cos,
                        accelerator,
                    )
                    logs["traj_mid_patch_neg_cos"] = safe_scalar(
                        traj_mid_patch_neg_cos,
                        accelerator,
                    )
                    logs["traj_high_global_nce"] = safe_scalar(
                        traj_high_global_nce,
                        accelerator,
                    )
                    logs["traj_high_global_nce_raw"] = safe_scalar(
                        traj_high_global_nce_raw,
                        accelerator,
                    )
                    logs["traj_high_global_pos_cos"] = safe_scalar(
                        traj_high_global_pos_cos,
                        accelerator,
                    )
                    logs["traj_high_global_neg_cos"] = safe_scalar(
                        traj_high_global_neg_cos,
                        accelerator,
                    )
                    logs["traj_high_global_valid_negatives"] = safe_scalar(
                        traj_high_global_valid_negatives,
                        accelerator,
                    )
                if args.traj_objective in jepa_objectives:
                    logs["repr_mask_ratio"] = args.repr_mask_ratio
                    logs["repr_timestep_mean"] = safe_scalar(
                        repr_timestep_mean,
                        accelerator,
                    )
                    logs["repr_timestep_min"] = safe_scalar(
                        repr_timestep_min,
                        accelerator,
                    )
                    logs["repr_timestep_max"] = safe_scalar(
                        repr_timestep_max,
                        accelerator,
                    )
                    logs["repr_masked_loss"] = safe_scalar(
                        repr_masked_loss,
                        accelerator,
                    )
                    logs["repr_masked_cos"] = safe_scalar(
                        repr_masked_cos,
                        accelerator,
                    )
                    logs["repr_unmasked_cos"] = safe_scalar(
                        repr_unmasked_cos,
                        accelerator,
                    )
                    logs["repr_relational_loss"] = safe_scalar(
                        repr_relational_loss,
                        accelerator,
                    )
                    logs["repr_variance_loss"] = safe_scalar(
                        repr_variance_loss,
                        accelerator,
                    )
                    logs["repr_student_std"] = safe_scalar(
                        repr_student_std,
                        accelerator,
                    )
                    logs["repr_teacher_std"] = safe_scalar(
                        repr_teacher_std,
                        accelerator,
                    )
                    logs["repr_student_effective_rank"] = safe_scalar(
                        repr_student_effective_rank,
                        accelerator,
                    )
                    logs["repr_teacher_effective_rank"] = safe_scalar(
                        repr_teacher_effective_rank,
                        accelerator,
                    )
                    for source_depth, target_depth in repr_depth_pairs:
                        pair_key = f"{source_depth}_to_{target_depth}"
                        depth_components = repr_depth_metrics.get(pair_key)
                        pair_cosine = (
                            depth_components["masked_cosine"]
                            if depth_components is not None
                            else traj_loss_val.new_zeros(())
                        )
                        logs[f"repr_{pair_key}_masked_cos"] = safe_scalar(
                            pair_cosine,
                            accelerator,
                        )
                        if (
                            args.traj_objective == "masked_jepa"
                            and source_depth == target_depth
                        ):
                            logs[f"repr_d{source_depth}_masked_cos"] = safe_scalar(
                                pair_cosine,
                                accelerator,
                            )
                if args.traj_sampler == "semantic":
                    logs["traj_semantic_updates"] = trajectory_sampler.semantic_updates
                    if trajectory_sampler.semantic_scores is not None:
                        logs["traj_semantic_score_max"] = safe_scalar(trajectory_sampler.semantic_scores.max(), accelerator)
                        logs["traj_semantic_score_mean"] = safe_scalar(trajectory_sampler.semantic_scores.mean(), accelerator)
            logging.info(f"losses: {logs}")
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--skip-training-samples", action="store_true",
                        help="disable periodic qualitative sampling during training")
    parser.add_argument("--resume-step", type=int, default=0)

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    # vae
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")  # Choice doesn't affect training
    parser.add_argument("--pretrained-model-path", type=str, default="/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/public/yangmengping/pretrained_models")
    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--encoder-checkpoint-dir", type=str, default="ckpts",
                        help="directory containing pretrained representation encoder checkpoints")
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--self-flow", action="store_true",
                        help="enable official dual-timestep Self-Flow training")
    parser.add_argument("--self-flow-mask-ratio", type=float, default=0.25,
                        help="fraction of image tokens assigned the second timestep")
    parser.add_argument("--self-flow-rep-coeff", type=float, default=0.8,
                        help="cosine representation alignment coefficient")
    parser.add_argument("--self-flow-teacher-depth", type=int, default=8,
                        help="1-based EMA teacher block depth")
    ##### added
    # skip-layer connection to improve the model's ability to capture long-range dependencies, improving the representation diversity
    parser.add_argument("--skip-layer-connection", action="store_true", help="skip-layer connection like unet")
    parser.add_argument("--cross-layer-connection", action="store_true",
                        help="alias/variant for skip-layer cross connection")
    # block diversity loss block_diversity_loss
    parser.add_argument("--block-diversity-loss", action="store_true", help="block diversity difference loss")
    parser.add_argument("--block-diversity-loss-coeff", type=float, default=0.001, help="coefficient for block difference loss")

    ##### new diversity methods
    # --- Layer Drop (Stochastic Depth) ---
    parser.add_argument("--layer-drop", action="store_true", help="enable stochastic depth layer drop")
    parser.add_argument("--layer-drop-rate", type=float, default=0.1, help="max layer drop probability")
    parser.add_argument("--layer-drop-strategy", type=str, default="linear", choices=["uniform", "linear", "cosine"],
                        help="layer drop probability schedule")
    parser.add_argument("--layer-drop-type", type=str, default="random", choices=["random", "batch"],
                        help="per-sample or whole-batch layer drop")

    # --- Block Contrastive Loss (InfoNCE) ---
    parser.add_argument("--block-contrastive-loss", action="store_true", help="enable InfoNCE contrastive loss between blocks")
    parser.add_argument("--block-contrastive-loss-coeff", type=float, default=0.01, help="coefficient for contrastive loss")
    parser.add_argument("--block-contrastive-temperature", type=float, default=0.1, help="InfoNCE temperature")

    # --- Barlow Twins Loss ---
    parser.add_argument("--block-barlow-twins-loss", action="store_true", help="enable Barlow Twins cross-correlation loss")
    parser.add_argument("--block-barlow-twins-loss-coeff", type=float, default=0.01, help="coefficient for Barlow Twins loss")
    parser.add_argument("--block-barlow-lambda", type=float, default=0.005, help="off-diagonal weight in Barlow Twins")

    # --- VICReg Loss ---
    parser.add_argument("--block-vicreg-loss", action="store_true", help="enable VICReg diversity loss")
    parser.add_argument("--block-vicreg-loss-coeff", type=float, default=0.01, help="coefficient for VICReg loss")
    parser.add_argument("--block-vicreg-lambda", type=float, default=25.0, help="VICReg variance weight")
    parser.add_argument("--block-vicreg-mu", type=float, default=25.0, help="VICReg invariance weight")
    parser.add_argument("--block-vicreg-nu", type=float, default=1.0, help="VICReg covariance weight")

    # --- Gradient Isolation ---
    parser.add_argument("--gradient-isolation", action="store_true", help="enable gradient isolation between block groups")
    parser.add_argument("--gradient-isolation-alpha", type=float, default=0.0, help="gradient scale at barriers (0=full stop)")
    parser.add_argument("--gradient-isolation-layers", type=str, default=None,
                        help="comma-separated layer indices for barriers, e.g. '9,18'")

    # --- Block Shuffling ---
    parser.add_argument("--block-shuffling", action="store_true", help="enable random block order shuffling")
    parser.add_argument("--block-shuffling-prob", type=float, default=0.1, help="probability of shuffling per forward pass")
    parser.add_argument("--block-shuffling-group-size", type=int, default=4, help="shuffle within groups of this size")

    # --- Per-Block Conditioning ---
    parser.add_argument("--per-block-conditioning", action="store_true",
                        help="learnable per-block conditioning offset for diversity")

    # --- Block-wise Auxiliary Heads ---
    parser.add_argument("--block-aux-heads", action="store_true", help="enable block-wise auxiliary prediction heads")
    parser.add_argument("--block-aux-head-layers", type=str, default=None,
                        help="comma-separated layer indices for aux heads, e.g. '7,14,21'")
    parser.add_argument("--block-aux-head-coeff", type=float, default=0.1, help="coefficient for auxiliary head loss")

    ##### Structured Heterogeneity (architectural diversity)
    # --- 1A: Per-Block Residual Scaling ---
    parser.add_argument("--residual-scaling", action="store_true",
                        help="learnable per-block residual scale with depth-dependent init")
    # --- 1B: Block-Group Conditioning Transform ---
    parser.add_argument("--block-group-conditioning", action="store_true",
                        help="lightweight affine transform per block group for conditioning diversity")
    parser.add_argument("--num-cond-groups", type=int, default=4,
                        help="number of conditioning groups (default: 4)")
    # --- 2A: Heterogeneous MLP Ratio ---
    parser.add_argument("--heterogeneous-mlp", action="store_true",
                        help="use varying MLP expansion ratios across blocks")
    # --- 2B: Alternating Attention Heads ---
    parser.add_argument("--alternating-heads", action="store_true",
                        help="alternate attention head counts (full/half) across blocks")
    # --- 2C: Depth-Aware Initialization ---
    parser.add_argument("--depth-aware-init", action="store_true",
                        help="depth-scaled weight init to break block symmetry")
    # --- Diversity Warmup ---
    parser.add_argument("--diversity-warmup-steps", type=int, default=10000,
                        help="linear warmup steps for all diversity mechanisms (0 = no warmup)")

    ##### Trajectory-DINO / TrajFlow
    parser.add_argument("--traj-loss", action="store_true",
                        help="enable trajectory-level self-supervised auxiliary loss")
    parser.add_argument("--traj-loss-coeff", type=float, default=0.05,
                        help="coefficient for trajectory auxiliary loss")
    parser.add_argument("--traj-warmup-steps", type=int, default=10000,
                        help="linear warmup steps for trajectory loss")
    parser.add_argument("--traj-decay-start", type=int, default=-1,
                        help="optimizer step to start cosine trajectory-loss decay")
    parser.add_argument("--traj-decay-end", type=int, default=-1,
                        help="optimizer step to finish cosine trajectory-loss decay")
    parser.add_argument("--traj-min-loss-scale", type=float, default=0.0,
                        help="minimum trajectory-loss scale after cosine decay")
    parser.add_argument("--traj-loss-frequency", type=int, default=8,
                        help="compute trajectory loss every N optimizer steps")
    parser.add_argument("--traj-batch-ratio", type=float, default=0.25,
                        help="fraction of local batch used for trajectory branch")
    parser.add_argument("--traj-num-steps", type=int, default=3,
                        help="number of timesteps in each trajectory")
    parser.add_argument("--traj-anchors", type=str, default="0.85,0.50,0.15",
                        help="comma-separated high/mid/low timestep anchors")
    parser.add_argument("--traj-base-jitter", type=float, default=0.10,
                        help="per-image jitter shared by the two positive schedules")
    parser.add_argument("--traj-view-jitter", type=float, default=0.03,
                        help="per-view jitter around each base trajectory schedule")
    parser.add_argument("--traj-min-gap", type=float, default=0.12,
                        help="minimum descending gap between trajectory timesteps")
    parser.add_argument("--traj-sampler", type=str, default="jittered",
                        choices=["jittered", "semantic"],
                        help="trajectory timestep sampler")
    parser.add_argument("--traj-semantic-bins", type=int, default=32,
                        help="number of bins for semantic-emergence guided sampling")
    parser.add_argument("--traj-semantic-mix", type=float, default=0.5,
                        help="mixture weight for semantic sampling vs stage-uniform sampling")
    parser.add_argument("--traj-semantic-temperature", type=float, default=0.2,
                        help="softmax temperature for semantic sampling scores")
    parser.add_argument("--traj-semantic-momentum", type=float, default=0.95,
                        help="EMA momentum for semantic velocity bin scores")
    parser.add_argument("--traj-semantic-warmup-steps", type=int, default=10000,
                        help="steps before semantic sampling is used")
    parser.add_argument("--traj-semantic-min-t", type=float, default=0.05,
                        help="lower timestep bound for semantic sampler bins")
    parser.add_argument("--traj-semantic-max-t", type=float, default=0.95,
                        help="upper timestep bound for semantic sampler bins")
    parser.add_argument("--traj-shared-noise-within-path", action=argparse.BooleanOptionalAction, default=True,
                        help="use one noise tensor for all timesteps inside each trajectory")
    parser.add_argument("--traj-depth", type=int, default=8,
                        help="1-based SiT block depth used for trajectory features")
    parser.add_argument("--traj-embed-dim", type=int, default=768,
                        help="hidden dimension of the trajectory temporal encoder")
    parser.add_argument("--traj-out-dim", type=int, default=256,
                        help="output dimension for trajectory SSL head")
    parser.add_argument("--traj-encoder-layers", type=int, default=2,
                        help="number of temporal Transformer layers")
    parser.add_argument("--traj-encoder-heads", type=int, default=8,
                        help="number of temporal attention heads")
    parser.add_argument("--traj-encoder-mlp-ratio", type=float, default=4.0,
                        help="MLP ratio in the temporal Transformer")
    parser.add_argument("--traj-encoder-dropout", type=float, default=0.0,
                        help="dropout in the temporal Transformer")
    parser.add_argument("--traj-objective", type=str, default="dino",
                        choices=["patch", "patch_infonce", "hierarchical",
                                 "masked_jepa", "contextual_jepa",
                                 "dino", "vicreg", "infonce"],
                        help="trajectory SSL objective")
    parser.add_argument("--traj-ema-decay", type=float, default=0.9999,
                        help="EMA decay for the trajectory teacher head")
    parser.add_argument("--traj-patch-sim-coeff", type=float, default=1.0,
                        help="patch-wise cosine alignment coefficient")
    parser.add_argument("--traj-patch-std-coeff", type=float, default=1.0,
                        help="online patch representation variance coefficient")
    parser.add_argument("--traj-patch-cov-coeff", type=float, default=0.04,
                        help="online patch representation covariance coefficient")
    parser.add_argument("--traj-patch-nce-coeff", type=float, default=1.0,
                        help="normalized cross-noise patch InfoNCE coefficient")
    parser.add_argument("--traj-global-nce-coeff", type=float, default=0.5,
                        help="high-noise global InfoNCE coefficient")
    parser.add_argument("--traj-patch-positive-coeff", type=float, default=0.25,
                        help="mid-noise positive patch cosine coefficient")
    parser.add_argument("--traj-patch-temperature", type=float, default=0.1,
                        help="cross-noise patch InfoNCE temperature")
    parser.add_argument("--traj-global-temperature", type=float, default=0.2,
                        help="high-noise global InfoNCE temperature")
    parser.add_argument("--traj-patch-num-samples", type=int, default=16,
                        help="number of spatial patches sampled per image for InfoNCE")
    parser.add_argument("--traj-patch-negative-sim-threshold", type=float, default=0.95,
                        help="filter teacher negatives above this cosine similarity")
    parser.add_argument("--traj-patch-min-negatives", type=int, default=32,
                        help="minimum negatives before disabling similarity filtering")
    parser.add_argument("--traj-student-temp", type=float, default=0.1,
                        help="DINO student temperature")
    parser.add_argument("--traj-teacher-temp", type=float, default=0.04,
                        help="DINO teacher temperature")
    parser.add_argument("--traj-center-momentum", type=float, default=0.9,
                        help="DINO center EMA momentum")
    parser.add_argument("--traj-vicreg-sim-coeff", type=float, default=25.0,
                        help="VICReg invariance coefficient")
    parser.add_argument("--traj-vicreg-std-coeff", type=float, default=25.0,
                        help="VICReg variance coefficient")
    parser.add_argument("--traj-vicreg-cov-coeff", type=float, default=1.0,
                        help="VICReg covariance coefficient")
    parser.add_argument("--traj-infonce-temperature", type=float, default=0.1,
                        help="InfoNCE temperature for trajectory baseline")
    parser.add_argument("--repr-depths", type=str, default="8,12",
                        help="comma-separated 1-based SiT depths for masked JEPA targets")
    parser.add_argument("--repr-depth-pairs", type=str, default="8:12,12:12",
                        help="comma-separated SOURCE:TARGET pairs for contextual JEPA")
    parser.add_argument("--repr-mask-ratio", type=float, default=0.4,
                        help="fraction of student patch tokens masked for JEPA prediction")
    parser.add_argument("--repr-mask-num-blocks", type=int, default=4,
                        help="number of compact target regions in contextual JEPA masks")
    parser.add_argument("--repr-timestep", type=float, default=0.5,
                        help="center timestep for same-state masked JEPA prediction")
    parser.add_argument("--repr-timestep-jitter", type=float, default=0.1,
                        help="uniform jitter around the masked JEPA center timestep")
    parser.add_argument("--repr-timestep-min", type=float, default=0.2,
                        help="minimum stratified timestep for contextual JEPA")
    parser.add_argument("--repr-timestep-max", type=float, default=0.8,
                        help="maximum stratified timestep for contextual JEPA")
    parser.add_argument("--repr-predictor-hidden-dim", type=int, default=768,
                        help="hidden dimension of the masked JEPA student predictor")
    parser.add_argument("--repr-relational-coeff", type=float, default=0.1,
                        help="image-relation preservation coefficient for masked JEPA")
    parser.add_argument("--repr-variance-coeff", type=float, default=0.1,
                        help="masked-token variance coefficient for masked JEPA")
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
