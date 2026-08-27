import argparse
import copy
from copy import deepcopy
import logging
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


def safe_mean(x):
    """Mean a tensor loss while leaving numeric zero-valued placeholders intact."""
    return x.mean() if hasattr(x, 'mean') else x


def linear_warmup(step, warmup_steps):
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, step / warmup_steps)


def cosine_decay_scale(step, start_step=-1, end_step=-1, min_scale=0.0):
    if start_step < 0 or end_step < 0:
        return 1.0
    if step <= start_step:
        return 1.0
    if step >= end_step:
        return min_scale
    progress = (step - start_step) / (end_step - start_step)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_scale + (1 - min_scale) * cosine

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


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    if args.trajectory_factorization:
        if not 0.0 < args.factor_batch_ratio <= 1.0:
            raise ValueError("--factor-batch-ratio must be in (0, 1]")
        if args.factor_loss_frequency <= 0:
            raise ValueError("--factor-loss-frequency must be positive")
        if args.factor_decay_start >= 0 or args.factor_decay_end >= 0:
            if not 0 <= args.factor_decay_start < args.factor_decay_end:
                raise ValueError(
                    "factor decay requires 0 <= start < end, or both values -1"
                )
        if not 0.0 <= args.factor_min_loss_scale <= 1.0:
            raise ValueError("--factor-min-loss-scale must be in [0, 1]")
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
        set_seed(args.seed + accelerator.process_index)
    
    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    use_external_encoder = (
        args.enc_type is not None and args.enc_type.lower() != 'none'
    )
    if use_external_encoder:
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, args.resolution
            )
    else:
        encoders, encoder_types, architectures = [], [], []
    z_dims = [encoder.embed_dim for encoder in encoders]
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = (args.cfg_prob > 0),
        z_dims = z_dims,
        encoder_depth=args.encoder_depth,
        skip_layer_connection=args.skip_layer_connection,
        block_diversity_loss=args.block_diversity_loss,
        trajectory_factorization=args.trajectory_factorization,
        factor_dim=args.factor_dim,
        factor_projector_dim=args.factor_projector_dim,
        factor_source_depth=args.factor_source_depth,
        factor_target_depth=args.factor_target_depth,
        factor_transition=args.factor_transition,
        **block_kwargs
    )

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
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
        block_diversity_loss = args.block_diversity_loss,
        projection=use_external_encoder and args.proj_coeff > 0,
        encoder_depth=args.encoder_depth,
        trajectory_factorization=args.trajectory_factorization,
        factor_pair_cross_noise_prob=args.factor_pair_cross_noise_prob,
        factor_min_delta_t=args.factor_min_delta_t,
        factor_max_delta_t=args.factor_max_delta_t,
        factor_transition=args.factor_transition,
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    
    # Setup data:
    if args.batch_size % accelerator.num_processes != 0:
        raise ValueError("--batch-size must be divisible by the number of processes")
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
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    if global_step == 0:
        # DDP broadcasts the online model during prepare; synchronize each
        # process-local EMA copy with that broadcast state.
        update_ema(ema, model, decay=0)

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
        sample_batch_size = max(1, 64 // accelerator.num_processes)
        gt_raw_images, gt_xs, _ = next(iter(train_dataloader))
        assert gt_raw_images.shape[-1] == args.resolution
        gt_xs = gt_xs[:sample_batch_size]
        gt_xs = sample_posterior(
            gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
            )
        ys = torch.randint(args.num_classes, size=(sample_batch_size,), device=device)
        xT = torch.randn((ys.size(0), 4, latent_size, latent_size), device=device)
        
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
            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
                zs = []
                with accelerator.autocast():
                    if use_external_encoder:
                        raw_image = raw_image.to(device)
                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                        z = encoder.forward_features(raw_image_)
                        if 'mocov3' in encoder_type: z = z = z[:, 1:] 
                        if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                        zs.append(z)

            with accelerator.accumulate(model):
                model_kwargs = dict(y=labels)
                factor_warmup = linear_warmup(
                    global_step, args.factor_warmup_steps
                )
                factor_decay = cosine_decay_scale(
                    global_step,
                    args.factor_decay_start,
                    args.factor_decay_end,
                    args.factor_min_loss_scale,
                )
                factor_has_objective = any(coeff > 0 for coeff in (
                    args.factor_inv_coeff,
                    args.factor_recom_coeff,
                    args.factor_transition_coeff if args.factor_transition else 0,
                    args.factor_decorrelation_coeff,
                    args.factor_variance_coeff,
                ))
                factorization_active = (
                    args.trajectory_factorization
                    and global_step % args.factor_loss_frequency == 0
                    and factor_warmup * factor_decay > 0
                    and (args.factor_paired_view_only or factor_has_objective)
                )
                factor_loss_scale = (
                    factor_warmup * factor_decay
                    if factorization_active and factor_has_objective else 0.0
                )
                losses = loss_fn(
                    model,
                    x,
                    model_kwargs,
                    zs=zs,
                    factorization_active=factorization_active,
                    factor_batch_ratio=args.factor_batch_ratio,
                )
                denoising_loss = losses.get('denoising_loss', 0)
                proj_loss = losses.get('proj_loss', 0)
                denoising_loss_mean = denoising_loss.mean()
                proj_loss_mean = proj_loss.mean()
                block_diversity_loss = losses.get('block_diversity_loss', 0)
                factor_inv_loss = losses.get('factor_inv_loss', 0)
                factor_recom_loss = losses.get('factor_recom_loss', 0)
                factor_transition_loss = losses.get('factor_transition_loss', 0)
                factor_decorrelation_loss = losses.get('factor_decorrelation_loss', 0)
                factor_variance_loss = losses.get('factor_variance_loss', 0)

                loss = denoising_loss_mean + proj_loss_mean * args.proj_coeff \
                    + block_diversity_loss * args.block_diversity_loss_coeff \
                    + safe_mean(factor_inv_loss) * args.factor_inv_coeff * factor_loss_scale \
                    + safe_mean(factor_recom_loss) * args.factor_recom_coeff * factor_loss_scale \
                    + safe_mean(factor_transition_loss) * args.factor_transition_coeff * factor_loss_scale \
                    + safe_mean(factor_decorrelation_loss) * args.factor_decorrelation_coeff * factor_loss_scale \
                    + safe_mean(factor_variance_loss) * args.factor_variance_coeff * factor_loss_scale
                    
                ## optimization
                accelerator.backward(loss)
                grad_norm = torch.zeros((), device=device)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model) # change ema function
            
            if not accelerator.sync_gradients:
                continue

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
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if (
                not args.skip_training_samples
                and (global_step == 1 or (
                    global_step % args.sampling_steps == 0 and global_step > 0
                ))
            ):
                from samplers import euler_sampler
                with torch.no_grad():
                    samples = euler_sampler(
                        ema,
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
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }
            if args.trajectory_factorization:
                for metric in (
                    'factor_inv_loss', 'factor_recom_loss', 'factor_transition_loss',
                    'factor_decorrelation_loss', 'factor_variance_loss',
                    'persistent_similarity', 'evolving_similarity',
                    'recomposition_gap', 'persistent_std', 'evolving_std',
                    'mean_delta_t', 'cross_noise_fraction', 'factor_batch_fraction',
                ):
                    if metric in losses:
                        logs[metric] = safe_scalar(losses[metric], accelerator)
                logs['factor_loss_scale'] = factor_loss_scale
                logs['factor_warmup'] = factor_warmup
                logs['factor_decay'] = factor_decay
                logs['factorization_active'] = float(factorization_active)
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
                        help="disable expensive qualitative sampling during training")
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
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b',
                        help="external REPA encoder, or 'none' for self-supervised training")
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)
    ##### added 
    # skip-layer connection to improve the model's ability to capture long-range dependencies, improving the representation diversity
    parser.add_argument("--skip-layer-connection", action="store_true", help="skip-layer connection like unet")
    # block diversity loss block_diversity_loss
    parser.add_argument("--block-diversity-loss", action="store_true", help="block diversity difference loss")
    parser.add_argument("--block-diversity-loss-coeff", type=float, default=0.001, help="coefficient for block difference loss")
    # Persistent--Evolving trajectory factorization. Auxiliary heads are used
    # only for training and add no denoising-time forward dependency.
    parser.add_argument("--trajectory-factorization", action="store_true",
                        help="learn persistent/evolving trajectory representations")
    parser.add_argument("--factor-dim", type=int, default=256)
    parser.add_argument("--factor-projector-dim", type=int, default=1024)
    parser.add_argument("--factor-source-depth", type=int, default=None,
                        help="1-indexed factorization layer; defaults to encoder depth")
    parser.add_argument("--factor-target-depth", type=int, default=None,
                        help="1-indexed reconstruction target layer; defaults to final block")
    parser.add_argument("--factor-pair-cross-noise-prob", type=float, default=0.5,
                        help="fraction of pairs that use independent noise realizations")
    parser.add_argument("--factor-min-delta-t", type=float, default=0.15)
    parser.add_argument("--factor-max-delta-t", type=float, default=0.7)
    parser.add_argument("--factor-inv-coeff", type=float, default=0.1)
    parser.add_argument("--factor-recom-coeff", type=float, default=0.1)
    parser.add_argument("--factor-transition", action="store_true",
                        help="predict evolving-code motion conditioned on signed delta-t")
    parser.add_argument("--factor-transition-coeff", type=float, default=0.05)
    parser.add_argument("--factor-decorrelation-coeff", type=float, default=0.0)
    parser.add_argument("--factor-variance-coeff", type=float, default=0.0)
    parser.add_argument("--factor-warmup-steps", type=int, default=10000)
    parser.add_argument("--factor-decay-start", type=int, default=-1)
    parser.add_argument("--factor-decay-end", type=int, default=-1)
    parser.add_argument("--factor-min-loss-scale", type=float, default=0.0)
    parser.add_argument("--factor-loss-frequency", type=int, default=1,
                        help="activate paired TFCR training every N optimizer steps")
    parser.add_argument("--factor-batch-ratio", type=float, default=0.5,
                        help="fraction of source batch receiving a second trajectory view")
    parser.add_argument("--factor-paired-view-only", action="store_true",
                        help="keep paired views active as a no-auxiliary-loss compute control")
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
