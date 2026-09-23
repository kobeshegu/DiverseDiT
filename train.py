import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json
import shutil
import subprocess
import sys
from datetime import datetime

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


def delayed_linear_warmup(step, start_step, warmup_steps):
    """Zero before ``start_step``, then linearly ramp from zero to one."""
    if step < start_step:
        return 0.0
    return linear_warmup(step - start_step, warmup_steps)


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


def archive_training_code(save_dir, archive_name="code"):
    """
    Copy the source files needed to reproduce this run into the experiment dir.
    Large artifacts such as results, samples, checkpoints, and .git are omitted.
    """
    repo_dir = Path(__file__).resolve().parent
    save_dir = Path(save_dir)
    archive_dir = save_dir / archive_name
    if archive_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = save_dir / f"{archive_name}_{timestamp}"
    archive_dir.mkdir(parents=True, exist_ok=False)

    root_files = [
        "train.py",
        "generate.py",
        "train_t2i.py",
        "generate_t2i.py",
        "dataset.py",
        "loss.py",
        "samplers.py",
        "samplers_t2i.py",
        "utils.py",
        "npz_convert.py",
        "evaluator.py",
        "requirements.txt",
        "README.md",
        "AGENTS.md",
    ]
    root_dirs = [
        "models",
        "scripts",
        "preprocessing",
        "dinov2",
        "analysis",
        "docs",
        "tests",
    ]
    ignore = shutil.ignore_patterns(
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".ipynb_checkpoints",
        ".git",
        "results",
        "sampled_images",
        "wandb",
        ".efc_*",
    )

    for rel_path in root_files:
        src = repo_dir / rel_path
        if src.is_file():
            dst = archive_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    for rel_path in root_dirs:
        src = repo_dir / rel_path
        if src.is_dir():
            shutil.copytree(src, archive_dir / rel_path, ignore=ignore)

    (archive_dir / "launch_command.txt").write_text(
        " ".join([sys.executable] + sys.argv) + "\n",
        encoding="utf-8",
    )

    git_commands = {
        "git_head.txt": ["git", "rev-parse", "HEAD"],
        "git_branch.txt": ["git", "branch", "--show-current"],
        "git_status.txt": ["git", "status", "--short", "--branch"],
        "git_diff.patch": ["git", "diff", "--"],
    }
    for filename, command in git_commands.items():
        try:
            result = subprocess.run(
                command,
                cwd=repo_dir,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            (archive_dir / filename).write_text(result.stdout, encoding="utf-8")
        except OSError as exc:
            (archive_dir / filename).write_text(
                f"Failed to run {' '.join(command)}: {exc}\n",
                encoding="utf-8",
            )

    return archive_dir


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
    if args.factor_shared_clean:
        args.factor_clean_consensus = True
    if args.factor_shared_clean_coeff is not None:
        args.factor_clean_consensus = True
        args.factor_clean_consensus_coeff = args.factor_shared_clean_coeff

    if args.trajectory_factorization and args.trajectory_invariance:
        raise ValueError(
            "--trajectory-factorization and --trajectory-invariance are "
            "mutually exclusive"
        )
    if args.proj_use_factor_schedule:
        if (
            args.enc_type is None
            or args.enc_type.lower() in {"", "none", "null"}
            or args.proj_coeff <= 0
        ):
            raise ValueError(
                "--proj-use-factor-schedule requires a non-none encoder and "
                "positive --proj-coeff"
            )
        if args.factor_warmup_steps < 0:
            raise ValueError("scheduled REPA warmup must be non-negative")
        if args.factor_regularization_start_steps < 0:
            raise ValueError(
                "factor regularization start step must be non-negative"
            )
        if args.factor_decay_start >= 0 or args.factor_decay_end >= 0:
            if not 0 <= args.factor_decay_start < args.factor_decay_end:
                raise ValueError(
                    "scheduled REPA decay requires 0 <= start < end, or -1/-1"
                )
    if (
        args.factor_clean_consensus
        or args.factor_selective_invariance
        or args.factor_shared_repa
        or args.factor_shared_self_distill
        or args.factor_shared_contrastive
        or args.factor_shared_relation
        or args.factor_evolving_separation
        or args.factor_native_parameterization
        or args.factor_native_shuffle_source
        or args.factor_semantic_conditioning
        or args.factor_semantic_shuffle_targets
        or args.factor_pair_interaction
        or args.factor_pair_random_align
        or args.factor_pair_byol_align
        or args.factor_self_flow_full_align
        or args.factor_self_flow_source_align
        or args.factor_self_flow_shuffle_teacher
        or args.factor_shared_repa_coeff > 0
        or args.factor_shared_self_distill_coeff > 0
        or args.factor_shared_variance_coeff > 0
        or args.factor_shared_contrastive_coeff > 0
        or args.factor_shared_relation_coeff > 0
        or args.factor_evolving_separation_coeff > 0
        or args.factor_native_source_coeff > 0
        or args.factor_native_noise_coeff > 0
        or args.factor_native_antithetic_coeff > 0
        or args.factor_native_base_coeff > 0
        or args.factor_semantic_repa_coeff > 0
        or args.factor_semantic_source_consistency_coeff > 0
        or args.factor_semantic_decorrelation_coeff > 0
        or args.factor_pair_random_align_coeff > 0
        or args.factor_pair_random_variance_coeff > 0
        or args.factor_pair_byol_align_coeff > 0
        or args.factor_pair_byol_variance_coeff > 0
        or args.factor_self_flow_full_coeff > 0
        or args.factor_self_flow_source_coeff > 0
    ) and not args.trajectory_factorization:
        raise ValueError(
            "factor shared-target objectives require "
            "--trajectory-factorization"
        )
    if args.trajectory_factorization:
        coefficient_names = (
            "factor_inv_coeff", "factor_persistent_coeff",
            "factor_evolving_coeff", "factor_recom_coeff",
            "factor_velocity_recom_coeff",
            "factor_adv_persistent_time_coeff",
            "factor_adv_persistent_orbit_coeff",
            "factor_probe_evolving_time_coeff",
            "factor_probe_evolving_orbit_coeff",
            "factor_transition_coeff", "factor_decorrelation_coeff",
            "factor_variance_coeff",
            "factor_clean_consensus_coeff", "factor_selective_coeff",
            "factor_selective_orth_coeff",
            "factor_selective_variance_coeff",
            "factor_shared_repa_coeff",
            "factor_shared_self_distill_coeff",
            "factor_shared_variance_coeff",
            "factor_shared_contrastive_coeff",
            "factor_shared_relation_coeff",
            "factor_evolving_separation_coeff",
            "factor_native_source_coeff",
            "factor_native_noise_coeff",
            "factor_native_antithetic_coeff",
            "factor_native_base_coeff",
            "factor_semantic_repa_coeff",
            "factor_semantic_source_consistency_coeff",
            "factor_semantic_decorrelation_coeff",
            "factor_pair_random_align_coeff",
            "factor_pair_random_variance_coeff",
            "factor_pair_byol_align_coeff",
            "factor_pair_byol_variance_coeff",
            "factor_self_flow_full_coeff",
            "factor_self_flow_source_coeff",
        )
        if any(getattr(args, name) < 0 for name in coefficient_names):
            raise ValueError("TFCR loss coefficients must be non-negative")
        if not 0.0 < args.factor_batch_ratio <= 1.0:
            raise ValueError("--factor-batch-ratio must be in (0, 1]")
        if not 0.0 <= args.factor_pair_cross_noise_prob <= 1.0:
            raise ValueError("--factor-pair-cross-noise-prob must be in [0, 1]")
        if not 0.0 <= args.factor_orbit_noise_only_prob <= 1.0:
            raise ValueError("--factor-orbit-noise-only-prob must be in [0, 1]")
        if args.factor_pair_interaction_scale < 0:
            raise ValueError("--factor-pair-interaction-scale must be non-negative")
        if args.factor_pair_interaction_hidden_ratio <= 0:
            raise ValueError(
                "--factor-pair-interaction-hidden-ratio must be positive"
            )
        if not 0.0 <= args.factor_pair_interaction_self_prob <= 1.0:
            raise ValueError(
                "--factor-pair-interaction-self-prob must be in [0, 1]"
            )
        if args.factor_pair_random_align_dim <= 0:
            raise ValueError("--factor-pair-random-align-dim must be positive")
        if args.factor_pair_alignment_dim <= 0:
            raise ValueError("--factor-pair-alignment-dim must be positive")
        if args.factor_pair_alignment_predictor_dim <= 0:
            raise ValueError(
                "--factor-pair-alignment-predictor-dim must be positive"
            )
        if args.factor_pair_align_variance_target <= 0:
            raise ValueError(
                "--factor-pair-align-variance-target must be positive"
            )
        if not (
            0.0
            <= args.factor_min_delta_t
            <= args.factor_max_delta_t
            < 1.0
        ):
            raise ValueError(
                "factor timestep deltas must satisfy 0 <= min <= max < 1"
            )
        if args.factor_loss_frequency <= 0:
            raise ValueError("--factor-loss-frequency must be positive")
        if not 0.0 < args.factor_reliability_keep_ratio <= 1.0:
            raise ValueError("--factor-reliability-keep-ratio must be in (0, 1]")
        if not 0.0 <= args.factor_reliability_floor <= 1.0:
            raise ValueError("--factor-reliability-floor must be in [0, 1]")
        if args.factor_clean_consensus_temperature <= 0:
            raise ValueError(
                "--factor-clean-consensus-temperature must be positive"
            )
        if args.factor_selective_dim <= 0:
            raise ValueError("--factor-selective-dim must be positive")
        if args.factor_selective_variance_target <= 0:
            raise ValueError(
                "--factor-selective-variance-target must be positive"
            )
        if args.factor_shared_target_temperature <= 0:
            raise ValueError(
                "--factor-shared-target-temperature must be positive"
            )
        if args.factor_shared_snr_power < 0:
            raise ValueError("--factor-shared-snr-power must be non-negative")
        if args.factor_shared_variance_target <= 0:
            raise ValueError(
                "--factor-shared-variance-target must be positive"
            )
        if args.factor_shared_contrastive_temperature <= 0:
            raise ValueError(
                "--factor-shared-contrastive-temperature must be positive"
            )
        if args.factor_evolving_separation_margin < 0:
            raise ValueError(
                "--factor-evolving-separation-margin must be non-negative"
            )
        if (
            args.factor_clean_consensus_coeff > 0
            and not args.factor_clean_consensus
        ):
            raise ValueError(
                "--factor-clean-consensus-coeff requires "
                "--factor-clean-consensus"
            )
        if args.factor_shared_repa_coeff > 0 and not args.factor_shared_repa:
            raise ValueError(
                "--factor-shared-repa-coeff requires --factor-shared-repa"
            )
        semantic_coefficients = (
            args.factor_semantic_repa_coeff,
            args.factor_semantic_source_consistency_coeff,
            args.factor_semantic_decorrelation_coeff,
        )
        if (
            any(coefficient > 0 for coefficient in semantic_coefficients)
            and not args.factor_semantic_conditioning
        ):
            raise ValueError(
                "semantic factorization coefficients require "
                "--factor-semantic-conditioning"
            )
        if args.factor_semantic_conditioning:
            if (
                args.enc_type is None
                or args.enc_type.lower() in {"", "none", "null"}
            ):
                raise ValueError(
                    "--factor-semantic-conditioning requires a non-none "
                    "clean representation encoder"
                )
            if args.proj_coeff != 0:
                raise ValueError(
                    "semantic conditioning aligns only the source branch; "
                    "set --proj-coeff 0 to disable full-hidden REPA"
                )
            if args.factor_loss_frequency != 1:
                raise ValueError(
                    "semantic conditioning requires --factor-loss-frequency 1"
                )
            if args.cfg_prob > 0 and not args.factor_share_cfg_dropout:
                raise ValueError(
                    "semantic paired views require --factor-share-cfg-dropout"
                )
            if args.factor_native_parameterization:
                raise ValueError(
                    "semantic conditioning and native parameterization are "
                    "mutually exclusive"
                )
        if args.factor_semantic_injection_scale < 0:
            raise ValueError(
                "--factor-semantic-injection-scale must be non-negative"
            )
        if (
            args.factor_pair_random_align_coeff > 0
            and not args.factor_pair_random_align
        ):
            raise ValueError(
                "--factor-pair-random-align-coeff requires "
                "--factor-pair-random-align"
            )
        if (
            args.factor_pair_random_variance_coeff > 0
            and not args.factor_pair_random_align
        ):
            raise ValueError(
                "--factor-pair-random-variance-coeff requires "
                "--factor-pair-random-align"
            )
        if (
            args.factor_pair_byol_align_coeff > 0
            and not args.factor_pair_byol_align
        ):
            raise ValueError(
                "--factor-pair-byol-align-coeff requires "
                "--factor-pair-byol-align"
            )
        if (
            args.factor_pair_byol_variance_coeff > 0
            and not args.factor_pair_byol_align
        ):
            raise ValueError(
                "--factor-pair-byol-variance-coeff requires "
                "--factor-pair-byol-align"
            )
        if (
            args.factor_self_flow_full_coeff > 0
            and not args.factor_self_flow_full_align
        ):
            raise ValueError(
                "--factor-self-flow-full-coeff requires "
                "--factor-self-flow-full-align"
            )
        if (
            args.factor_self_flow_source_coeff > 0
            and not args.factor_self_flow_source_align
        ):
            raise ValueError(
                "--factor-self-flow-source-coeff requires "
                "--factor-self-flow-source-align"
            )
        if (
            args.factor_self_flow_shuffle_teacher
            and not (
                args.factor_self_flow_full_align
                or args.factor_self_flow_source_align
            )
        ):
            raise ValueError(
                "--factor-self-flow-shuffle-teacher requires an EMA "
                "self-flow alignment objective"
            )
        if (
            args.factor_self_flow_source_align
            and args.factor_native_parameterization
        ):
            raise ValueError(
                "source self-flow alignment and native parameterization are "
                "mutually exclusive"
            )
        if (
            args.factor_semantic_shuffle_targets
            and not args.factor_semantic_conditioning
        ):
            raise ValueError(
                "--factor-semantic-shuffle-targets requires "
                "--factor-semantic-conditioning"
            )
        if (
            args.factor_shared_repa
            and args.factor_shared_repa_coeff > 0
            and (
                args.enc_type is None
                or args.enc_type.lower() in {"", "none", "null"}
            )
        ):
            raise ValueError(
                "--factor-shared-repa needs a non-none --enc-type so the "
                "paired views can share an external clean-image target"
            )
        if (
            args.factor_shared_self_distill_coeff > 0
            and not args.factor_shared_self_distill
        ):
            raise ValueError(
                "--factor-shared-self-distill-coeff requires "
                "--factor-shared-self-distill"
            )
        if (
            args.factor_shared_variance_coeff > 0
            and not (
                args.factor_shared_self_distill
                or args.factor_shared_contrastive
                or args.factor_shared_relation
            )
        ):
            raise ValueError(
                "--factor-shared-variance-coeff requires "
                "a full-feature shared target objective"
            )
        if (
            args.factor_shared_contrastive_coeff > 0
            and not args.factor_shared_contrastive
        ):
            raise ValueError(
                "--factor-shared-contrastive-coeff requires "
                "--factor-shared-contrastive"
            )
        if (
            args.factor_shared_relation_coeff > 0
            and not args.factor_shared_relation
        ):
            raise ValueError(
                "--factor-shared-relation-coeff requires "
                "--factor-shared-relation"
            )
        if (
            args.factor_evolving_separation_coeff > 0
            and not args.factor_evolving_separation
        ):
            raise ValueError(
                "--factor-evolving-separation-coeff requires "
                "--factor-evolving-separation"
            )
        if (
            args.factor_shared_self_distill_shuffle_targets
            and not args.factor_shared_self_distill
        ):
            raise ValueError(
                "--factor-shared-self-distill-shuffle-targets requires "
                "--factor-shared-self-distill"
            )
        selective_coefficients = (
            args.factor_selective_coeff,
            args.factor_selective_orth_coeff,
            args.factor_selective_variance_coeff,
        )
        if (
            any(coefficient > 0 for coefficient in selective_coefficients)
            and not args.factor_selective_invariance
        ):
            raise ValueError(
                "factor selective coefficients require "
                "--factor-selective-invariance"
            )
        if (
            (
                args.factor_clean_consensus
                or args.factor_selective_invariance
                or args.factor_shared_repa
                or args.factor_shared_self_distill
                or args.factor_shared_contrastive
                or args.factor_shared_relation
                or args.factor_evolving_separation
                or args.factor_semantic_conditioning
                or args.factor_self_flow_full_align
                or args.factor_self_flow_source_align
            )
            and args.cfg_prob > 0
            and not args.factor_share_cfg_dropout
        ):
            raise ValueError(
                "paired shared-target objectives require "
                "--factor-share-cfg-dropout when classifier-free dropout is "
                "enabled"
            )
        if (
            args.factor_clean_consensus_shuffle_targets
            and not args.factor_clean_consensus
        ):
            raise ValueError(
                "--factor-clean-consensus-shuffle-targets requires "
                "--factor-clean-consensus"
            )
        if (
            args.factor_selective_shuffle_targets
            and not args.factor_selective_invariance
        ):
            raise ValueError(
                "--factor-selective-shuffle-targets requires "
                "--factor-selective-invariance"
            )
        if args.factor_selective_shuffle_utility and (
            not args.factor_selective_invariance
            or args.factor_selective_weighting != "task"
        ):
            raise ValueError(
                "--factor-selective-shuffle-utility requires task-weighted "
                "--factor-selective-invariance"
            )
        if (
            args.factor_velocity_recom_coeff > 0
            and not args.factor_velocity_recomposition
        ):
            raise ValueError(
                "--factor-velocity-recom-coeff requires "
                "--factor-velocity-recomposition"
            )
        if args.factor_native_parameterization:
            if args.factor_orbit_mode != "antithetic":
                raise ValueError(
                    "--factor-native-parameterization requires "
                    "--factor-orbit-mode antithetic"
                )
            if args.factor_loss_frequency != 1:
                raise ValueError(
                    "native parameterization requires --factor-loss-frequency 1"
                )
            if args.cfg_prob > 0 and not args.factor_share_cfg_dropout:
                raise ValueError(
                    "native antithetic pairs require "
                    "--factor-share-cfg-dropout when CFG dropout is enabled"
                )
            if args.factor_velocity_recomposition:
                raise ValueError(
                    "native parameterization and legacy velocity "
                    "recomposition are mutually exclusive"
                )
        native_coefficients = (
            args.factor_native_source_coeff,
            args.factor_native_noise_coeff,
            args.factor_native_antithetic_coeff,
            args.factor_native_base_coeff,
        )
        if any(coefficient > 0 for coefficient in native_coefficients):
            if not args.factor_native_parameterization:
                raise ValueError(
                    "native source/noise coefficients require "
                    "--factor-native-parameterization"
                )
        if (
            args.factor_native_shuffle_source
            and not args.factor_native_parameterization
        ):
            raise ValueError(
                "--factor-native-shuffle-source requires "
                "--factor-native-parameterization"
            )
        adversarial_coefficients = (
            args.factor_adv_persistent_time_coeff,
            args.factor_adv_persistent_orbit_coeff,
            args.factor_probe_evolving_time_coeff,
            args.factor_probe_evolving_orbit_coeff,
        )
        if any(coefficient > 0 for coefficient in adversarial_coefficients):
            if not args.factor_adversarial:
                raise ValueError(
                    "factor adversarial/probe coefficients require "
                    "--factor-adversarial"
                )
        if args.factor_adversarial:
            if args.factor_orbit_mode != "orthogonal":
                raise ValueError(
                    "--factor-adversarial requires --factor-orbit-mode orthogonal"
                )
            if not 0.0 < args.factor_orbit_noise_only_prob < 1.0:
                raise ValueError(
                    "--factor-adversarial requires both time-only and "
                    "noise-only pairs"
                )
            if args.cfg_prob > 0 and not args.factor_share_cfg_dropout:
                raise ValueError(
                    "--factor-adversarial requires --factor-share-cfg-dropout "
                    "when classifier-free dropout is enabled"
                )
            if args.factor_adversarial_timestep_bins < 2:
                raise ValueError(
                    "--factor-adversarial-timestep-bins must be at least 2"
                )
            if not 0.0 <= args.factor_adversarial_grl_scale <= 1.0:
                raise ValueError(
                    "--factor-adversarial-grl-scale must be in [0, 1]"
                )
            if (
                args.factor_adversarial_start_steps < 0
                or args.factor_adversarial_warmup_steps < 0
            ):
                raise ValueError(
                    "factor adversarial start/warmup steps must be non-negative"
                )
        elif args.factor_adversarial_shuffle_labels:
            raise ValueError(
                "--factor-adversarial-shuffle-labels requires "
                "--factor-adversarial"
            )
        if args.factor_warmup_steps < 0:
            raise ValueError("--factor-warmup-steps must be non-negative")
        if args.factor_regularization_start_steps < 0:
            raise ValueError(
                "--factor-regularization-start-steps must be non-negative"
            )
        if args.factor_decay_start >= 0 or args.factor_decay_end >= 0:
            if not 0 <= args.factor_decay_start < args.factor_decay_end:
                raise ValueError(
                    "factor decay requires 0 <= start < end, or both values -1"
                )
        if not 0.0 <= args.factor_min_loss_scale <= 1.0:
            raise ValueError("--factor-min-loss-scale must be in [0, 1]")
    if args.trajectory_invariance:
        if args.invariant_dim <= 0 or args.invariant_projector_dim <= 0:
            raise ValueError("invariant dimensions must be positive")
        coefficient_names = (
            "invariant_time_coeff", "invariant_noise_coeff",
            "invariant_image_variance_coeff",
            "invariant_spatial_variance_coeff", "invariant_covariance_coeff",
            "invariant_relation_coeff", "invariant_basis_coeff",
        )
        if any(getattr(args, name) < 0 for name in coefficient_names):
            raise ValueError("invariance loss coefficients must be non-negative")
        has_objective = any(
            getattr(args, name) > 0 for name in coefficient_names
        )
        if args.invariant_view_control_only and has_objective:
            raise ValueError(
                "--invariant-view-control-only requires every invariant loss "
                "coefficient to be zero"
            )
        if not args.invariant_view_control_only and not has_objective:
            raise ValueError(
                "--trajectory-invariance needs a non-zero invariant loss "
                "coefficient or --invariant-view-control-only"
            )
        if (
            args.invariant_projector_type == "mlp"
            and args.invariant_basis_coeff != 0
        ):
            raise ValueError(
                "--invariant-basis-coeff must be zero for the MLP readout"
            )
        if not 0.0 < args.invariant_batch_ratio <= 1.0:
            raise ValueError("--invariant-batch-ratio must be in (0, 1]")
        if not (
            0.0
            <= args.invariant_min_delta_t
            <= args.invariant_max_delta_t
            < 1.0
        ):
            raise ValueError(
                "invariant timestep deltas must satisfy 0 <= min <= max < 1"
            )
        if not 0.0 < args.invariant_max_t <= 1.0:
            raise ValueError("--invariant-max-t must be in (0, 1]")
        if args.invariant_max_delta_t > args.invariant_max_t:
            raise ValueError(
                "--invariant-max-delta-t cannot exceed --invariant-max-t"
            )
        if args.invariant_snr_power < 0:
            raise ValueError("--invariant-snr-power must be non-negative")
        if args.invariant_loss_frequency <= 0:
            raise ValueError("--invariant-loss-frequency must be positive")
        if args.invariant_warmup_steps < 0:
            raise ValueError("--invariant-warmup-steps must be non-negative")
        if args.invariant_decay_start >= 0 or args.invariant_decay_end >= 0:
            if not 0 <= args.invariant_decay_start < args.invariant_decay_end:
                raise ValueError(
                    "invariant decay requires 0 <= start < end, or both values -1"
                )
        if not 0.0 <= args.invariant_min_loss_scale <= 1.0:
            raise ValueError("--invariant-min-loss-scale must be in [0, 1]")
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
        code_archive_dir = None
        if not args.skip_code_archive:
            code_archive_dir = archive_training_code(
                save_dir,
                archive_name=args.code_archive_dir,
            )
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
        if code_archive_dir is not None:
            logger.info(f"Archived training code at {code_archive_dir}")
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
        path_type=args.path_type,
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = (args.cfg_prob > 0),
        class_dropout_prob=args.cfg_prob,
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
        factor_velocity_recomposition=args.factor_velocity_recomposition,
        factor_native_parameterization=args.factor_native_parameterization,
        factor_semantic_conditioning=args.factor_semantic_conditioning,
        factor_self_flow_conditioning=args.factor_self_flow_source_align,
        factor_semantic_injection_scale=(
            args.factor_semantic_injection_scale
        ),
        factor_pair_interaction=args.factor_pair_interaction,
        factor_pair_interaction_depth=args.factor_pair_interaction_depth,
        factor_pair_interaction_scale=args.factor_pair_interaction_scale,
        factor_pair_interaction_hidden_ratio=(
            args.factor_pair_interaction_hidden_ratio
        ),
        factor_pair_interaction_self_prob=(
            args.factor_pair_interaction_self_prob
        ),
        factor_pair_interaction_detach_context=(
            args.factor_pair_interaction_detach_context
        ),
        factor_pair_byol_alignment=args.factor_pair_byol_align,
        factor_pair_alignment_dim=args.factor_pair_alignment_dim,
        factor_pair_alignment_predictor_dim=(
            args.factor_pair_alignment_predictor_dim
        ),
        factor_adversarial=args.factor_adversarial,
        factor_adversarial_timestep_bins=(
            args.factor_adversarial_timestep_bins
        ),
        factor_selective_invariance=args.factor_selective_invariance,
        factor_selective_dim=args.factor_selective_dim,
        factor_selective_source_depth=(
            args.factor_selective_source_depth
        ),
        factor_shared_source_depth=args.factor_shared_source_depth,
        trajectory_invariance=args.trajectory_invariance,
        invariant_dim=args.invariant_dim,
        invariant_projector_dim=args.invariant_projector_dim,
        invariant_source_depth=args.invariant_source_depth,
        invariant_projector_type=args.invariant_projector_type,
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
        factor_orbit_mode=args.factor_orbit_mode,
        factor_orbit_noise_only_prob=args.factor_orbit_noise_only_prob,
        factor_min_delta_t=args.factor_min_delta_t,
        factor_max_delta_t=args.factor_max_delta_t,
        factor_transition=args.factor_transition,
        factor_native_parameterization=args.factor_native_parameterization,
        factor_native_shuffle_source=args.factor_native_shuffle_source,
        factor_semantic_conditioning=args.factor_semantic_conditioning,
        factor_semantic_shuffle_targets=(
            args.factor_semantic_shuffle_targets
        ),
        factor_self_flow_full_align=args.factor_self_flow_full_align,
        factor_self_flow_source_align=args.factor_self_flow_source_align,
        factor_self_flow_shuffle_teacher=(
            args.factor_self_flow_shuffle_teacher
        ),
        factor_reliable_target=args.factor_reliable_target,
        factor_reliability_keep_ratio=args.factor_reliability_keep_ratio,
        factor_reliability_floor=args.factor_reliability_floor,
        factor_adversarial=args.factor_adversarial,
        factor_adversarial_timestep_bins=(
            args.factor_adversarial_timestep_bins
        ),
        factor_adversarial_shuffle_labels=(
            args.factor_adversarial_shuffle_labels
        ),
        factor_clean_consensus=args.factor_clean_consensus,
        factor_clean_consensus_temperature=(
            args.factor_clean_consensus_temperature
        ),
        factor_clean_consensus_shuffle_targets=(
            args.factor_clean_consensus_shuffle_targets
        ),
        factor_selective_invariance=args.factor_selective_invariance,
        factor_selective_weighting=args.factor_selective_weighting,
        factor_selective_shuffle_targets=(
            args.factor_selective_shuffle_targets
        ),
        factor_selective_shuffle_utility=(
            args.factor_selective_shuffle_utility
        ),
        factor_selective_variance_target=(
            args.factor_selective_variance_target
        ),
        factor_shared_repa=args.factor_shared_repa,
        factor_shared_self_distill=args.factor_shared_self_distill,
        factor_shared_target_temperature=(
            args.factor_shared_target_temperature
        ),
        factor_shared_snr_power=args.factor_shared_snr_power,
        factor_shared_self_distill_shuffle_targets=(
            args.factor_shared_self_distill_shuffle_targets
        ),
        factor_shared_variance_target=args.factor_shared_variance_target,
        factor_shared_contrastive=args.factor_shared_contrastive,
        factor_shared_contrastive_temperature=(
            args.factor_shared_contrastive_temperature
        ),
        factor_shared_relation=args.factor_shared_relation,
        factor_evolving_separation=args.factor_evolving_separation,
        factor_evolving_separation_margin=(
            args.factor_evolving_separation_margin
        ),
        factor_pair_random_align=args.factor_pair_random_align,
        factor_pair_random_align_dim=args.factor_pair_random_align_dim,
        factor_pair_random_align_seed=args.factor_pair_random_align_seed,
        factor_pair_byol_align=args.factor_pair_byol_align,
        factor_pair_align_variance_target=(
            args.factor_pair_align_variance_target
        ),
        trajectory_invariance=args.trajectory_invariance,
        invariant_min_delta_t=args.invariant_min_delta_t,
        invariant_max_delta_t=args.invariant_max_delta_t,
        invariant_max_t=args.invariant_max_t,
        invariant_snr_power=args.invariant_snr_power,
        invariant_variance_target=args.invariant_variance_target,
        invariant_spatial_variance_target=(
            args.invariant_spatial_variance_target
        ),
    )
    factor_coefficients = {
        'factor_inv_loss': args.factor_inv_coeff,
        'factor_persistent_loss': args.factor_persistent_coeff,
        'factor_evolving_loss': args.factor_evolving_coeff,
        'factor_recom_loss': args.factor_recom_coeff,
        'factor_velocity_recom_loss': args.factor_velocity_recom_coeff,
        'factor_native_source_loss': args.factor_native_source_coeff,
        'factor_native_noise_loss': args.factor_native_noise_coeff,
        'factor_native_antithetic_loss': (
            args.factor_native_antithetic_coeff
        ),
        'factor_native_base_loss': args.factor_native_base_coeff,
        'factor_semantic_repa_loss': args.factor_semantic_repa_coeff,
        'factor_semantic_source_consistency_loss': (
            args.factor_semantic_source_consistency_coeff
        ),
        'factor_semantic_decorrelation_loss': (
            args.factor_semantic_decorrelation_coeff
        ),
        'factor_pair_random_align_loss': args.factor_pair_random_align_coeff,
        'factor_pair_random_variance_loss': (
            args.factor_pair_random_variance_coeff
        ),
        'factor_pair_byol_align_loss': args.factor_pair_byol_align_coeff,
        'factor_pair_byol_variance_loss': (
            args.factor_pair_byol_variance_coeff
        ),
        'factor_self_flow_full_loss': args.factor_self_flow_full_coeff,
        'factor_self_flow_source_loss': args.factor_self_flow_source_coeff,
        'factor_adv_persistent_time_loss': (
            args.factor_adv_persistent_time_coeff
        ),
        'factor_adv_persistent_orbit_loss': (
            args.factor_adv_persistent_orbit_coeff
        ),
        'factor_probe_evolving_time_loss': (
            args.factor_probe_evolving_time_coeff
        ),
        'factor_probe_evolving_orbit_loss': (
            args.factor_probe_evolving_orbit_coeff
        ),
        'factor_transition_loss': (
            args.factor_transition_coeff if args.factor_transition else 0.0
        ),
        'factor_decorrelation_loss': args.factor_decorrelation_coeff,
        'factor_variance_loss': args.factor_variance_coeff,
        'factor_clean_consensus_loss': args.factor_clean_consensus_coeff,
        'factor_selective_loss': args.factor_selective_coeff,
        'factor_selective_orth_loss': args.factor_selective_orth_coeff,
        'factor_selective_variance_loss': (
            args.factor_selective_variance_coeff
        ),
        'factor_shared_repa_loss': args.factor_shared_repa_coeff,
        'factor_shared_self_distill_loss': (
            args.factor_shared_self_distill_coeff
        ),
        'factor_shared_variance_loss': args.factor_shared_variance_coeff,
        'factor_shared_contrastive_loss': (
            args.factor_shared_contrastive_coeff
        ),
        'factor_shared_relation_loss': args.factor_shared_relation_coeff,
        'factor_evolving_separation_loss': (
            args.factor_evolving_separation_coeff
        ),
    }
    factor_has_objective = any(
        coefficient > 0 for coefficient in factor_coefficients.values()
    )
    invariant_coefficients = {
        'invariant_time_loss': args.invariant_time_coeff,
        'invariant_noise_loss': args.invariant_noise_coeff,
        'invariant_image_variance_loss': (
            args.invariant_image_variance_coeff
        ),
        'invariant_spatial_variance_loss': (
            args.invariant_spatial_variance_coeff
        ),
        'invariant_covariance_loss': args.invariant_covariance_coeff,
        'invariant_relation_loss': args.invariant_relation_coeff,
        'invariant_basis_loss': args.invariant_basis_coeff,
    }
    invariant_has_objective = any(
        coefficient > 0 for coefficient in invariant_coefficients.values()
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
                # A controlled trajectory intervention must not accidentally
                # change conditioning. Q-series runs always share the mask;
                # TFCR does so only behind an explicit compatibility switch.
                share_cfg_dropout = (
                    (args.trajectory_invariance and not args.legacy)
                    or (
                        args.trajectory_factorization
                        and args.factor_share_cfg_dropout
                    )
                )
                if share_cfg_dropout and args.cfg_prob > 0:
                    model_kwargs['force_drop_ids'] = (
                        torch.rand(y.shape[0], device=y.device) < args.cfg_prob
                    )
                factor_warmup = linear_warmup(
                    global_step, args.factor_warmup_steps
                )
                factor_regularization_warmup = delayed_linear_warmup(
                    global_step,
                    args.factor_regularization_start_steps,
                    args.factor_warmup_steps,
                )
                factor_decay = cosine_decay_scale(
                    global_step,
                    args.factor_decay_start,
                    args.factor_decay_end,
                    args.factor_min_loss_scale,
                )
                factorization_active = (
                    args.trajectory_factorization
                    and global_step % args.factor_loss_frequency == 0
                    and (
                        args.factor_native_parameterization
                        or args.factor_semantic_conditioning
                        or args.factor_pair_interaction
                        or args.factor_pair_random_align
                        or args.factor_pair_byol_align
                        or args.factor_self_flow_full_align
                        or args.factor_self_flow_source_align
                        or (
                            args.factor_paired_view_only
                            and factor_warmup * factor_decay > 0
                        )
                        or factor_regularization_warmup * factor_decay > 0
                    )
                    and (
                        args.factor_paired_view_only
                        or args.factor_native_parameterization
                        or args.factor_semantic_conditioning
                        or args.factor_pair_interaction
                        or args.factor_pair_random_align
                        or args.factor_pair_byol_align
                        or args.factor_self_flow_full_align
                        or args.factor_self_flow_source_align
                        or factor_has_objective
                    )
                )
                factor_loss_scale = (
                    factor_regularization_warmup * factor_decay
                    if factorization_active and factor_has_objective else 0.0
                )
                proj_loss_scale = (
                    factor_warmup * factor_decay
                    if args.proj_use_factor_schedule else 1.0
                )
                factor_adversarial_grl_ramp = (
                    delayed_linear_warmup(
                        global_step,
                        args.factor_adversarial_start_steps,
                        args.factor_adversarial_warmup_steps,
                    )
                    if args.factor_adversarial else 0.0
                )
                factor_adversarial_grl_scale = (
                    args.factor_adversarial_grl_scale
                    * factor_adversarial_grl_ramp
                )
                invariant_warmup = linear_warmup(
                    global_step, args.invariant_warmup_steps
                )
                invariant_decay = cosine_decay_scale(
                    global_step,
                    args.invariant_decay_start,
                    args.invariant_decay_end,
                    args.invariant_min_loss_scale,
                )
                invariance_active = (
                    args.trajectory_invariance
                    and global_step % args.invariant_loss_frequency == 0
                    and invariant_warmup * invariant_decay > 0
                    and (
                        args.invariant_view_control_only
                        or invariant_has_objective
                    )
                )
                invariant_loss_scale = (
                    invariant_warmup * invariant_decay
                    if invariance_active and invariant_has_objective else 0.0
                )
                losses = loss_fn(
                    model,
                    x,
                    model_kwargs,
                    zs=zs,
                    factorization_active=factorization_active,
                    factor_batch_ratio=args.factor_batch_ratio,
                    factor_adversarial_grl_scale=(
                        factor_adversarial_grl_scale
                    ),
                    invariance_active=invariance_active,
                    invariant_batch_ratio=args.invariant_batch_ratio,
                    ema_model=(
                        ema if (
                            args.factor_self_flow_full_align
                            or args.factor_self_flow_source_align
                        ) else None
                    ),
                )
                denoising_loss = losses.get('denoising_loss', 0)
                proj_loss = losses.get('proj_loss', 0)
                denoising_loss_mean = denoising_loss.mean()
                proj_loss_mean = proj_loss.mean()
                block_diversity_loss = losses.get('block_diversity_loss', 0)
                factor_regularization = sum(
                    safe_mean(losses.get(name, 0)) * coefficient
                    for name, coefficient in factor_coefficients.items()
                )
                invariant_regularization = sum(
                    safe_mean(losses.get(name, 0)) * coefficient
                    for name, coefficient in invariant_coefficients.items()
                )

                loss = (
                    denoising_loss_mean
                    + proj_loss_mean * args.proj_coeff * proj_loss_scale
                    + block_diversity_loss * args.block_diversity_loss_coeff
                    + factor_regularization * factor_loss_scale
                    + invariant_regularization * invariant_loss_scale
                )
                    
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
                        "format_version": 2,
                        "model": unwrapped_model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                        "tfcr_objective": (
                            (
                                "selective_semantic_factorization_v1"
                                if args.factor_semantic_conditioning
                                else "antithetic_native_parameterization_v1"
                                if args.factor_native_parameterization
                                else "paired_shared_target_v1"
                                if (
                                    args.factor_clean_consensus
                                    or args.factor_selective_invariance
                                    or args.factor_shared_repa
                                    or args.factor_shared_self_distill
                                    or args.factor_shared_contrastive
                                    or args.factor_shared_relation
                                    or args.factor_evolving_separation
                                    or args.factor_self_flow_full_align
                                    or args.factor_self_flow_source_align
                                )
                                else (
                                    "adversarial_orbit_purification_v3"
                                    if args.factor_adversarial
                                    else (
                                        "orbit_consensus_task_v2"
                                        if (
                                            args.factor_orbit_mode != "legacy"
                                            or args.factor_reliable_target
                                            or args.factor_velocity_recomposition
                                        )
                                        else "balanced_additive_v1"
                                    )
                                )
                            )
                            if args.trajectory_factorization else None
                        ),
                        "representation_objective": (
                            "dino_source_film_evolving_v1"
                            if args.factor_semantic_conditioning
                            else "ema_source_film_self_flow_v1"
                            if args.factor_self_flow_source_align
                            else "ema_full_hidden_self_flow_v1"
                            if args.factor_self_flow_full_align
                            else "native_source_evolution_v1"
                            if args.factor_native_parameterization
                            else "full_feature_shared_target_v1"
                            if (
                                args.factor_shared_self_distill
                                or args.factor_shared_contrastive
                                or args.factor_shared_relation
                            )
                            else (
                                "velocity_guided_selective_invariance_v1"
                                if args.factor_selective_invariance
                                else (
                                    "trajectory_orbit_subspace_v2"
                                    if args.trajectory_invariance else None
                                )
                            )
                        ),
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
                "proj_loss_scale": proj_loss_scale,
                "block_diversity_loss": safe_scalar(block_diversity_loss, accelerator),
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }
            if args.trajectory_factorization:
                for metric in (
                    'factor_inv_loss', 'factor_persistent_loss',
                    'factor_evolving_loss', 'factor_recom_loss',
                    'factor_transition_loss',
                    'factor_decorrelation_loss', 'factor_variance_loss',
                    'persistent_similarity', 'evolving_similarity',
                    'recomposition_gap', 'persistent_usage_gap',
                    'evolving_usage_gap', 'common_energy_fraction',
                    'residual_energy_fraction', 'persistent_std', 'evolving_std',
                    'mean_delta_t', 'cross_noise_fraction', 'factor_batch_fraction',
                    'factor_time_only_fraction', 'factor_noise_only_fraction',
                    'factor_joint_intervention_fraction',
                    'factor_time_intervention_delta',
                    'factor_target_reliability', 'factor_target_gate_mean',
                    'factor_target_selected_fraction',
                    'factor_velocity_recom_loss',
                    'factor_velocity_recom_error',
                    'factor_velocity_reconstruction_loss',
                    'factor_velocity_persistent_loss',
                    'factor_velocity_evolving_loss',
                    'factor_native_source_loss',
                    'factor_native_noise_loss',
                    'factor_native_antithetic_loss',
                    'factor_native_base_loss',
                    'factor_native_source_error',
                    'factor_native_noise_error',
                    'factor_native_base_error',
                    'factor_native_source_pair_gap',
                    'factor_native_noise_antisymmetry_error',
                    'factor_native_source_weight',
                    'factor_native_noise_weight',
                    'factor_semantic_repa_loss',
                    'factor_semantic_source_consistency_loss',
                    'factor_semantic_decorrelation_loss',
                    'factor_semantic_source_similarity',
                    'factor_semantic_source_evolving_cosine_sq',
                    'factor_semantic_source_std',
                    'factor_semantic_evolving_std',
                    'factor_semantic_modulation_rms',
                    'factor_adv_persistent_time_loss',
                    'factor_adv_persistent_orbit_loss',
                    'factor_probe_evolving_time_loss',
                    'factor_probe_evolving_orbit_loss',
                    'factor_adv_persistent_time_accuracy',
                    'factor_adv_persistent_orbit_accuracy',
                    'factor_probe_evolving_time_accuracy',
                    'factor_probe_evolving_orbit_accuracy',
                    'factor_time_separation_gap',
                    'factor_orbit_separation_gap',
                    'factor_time_majority_accuracy',
                    'factor_orbit_majority_accuracy',
                    'factor_clean_consensus_loss',
                    'factor_clean_pair_gap',
                    'factor_clean_source_error',
                    'factor_clean_consensus_source_error',
                    'factor_clean_confidence_max',
                    'factor_clean_confidence_entropy',
                    'factor_selective_loss',
                    'factor_selective_orth_loss',
                    'factor_selective_variance_loss',
                    'factor_selective_similarity',
                    'factor_selective_source_ratio',
                    'factor_selective_between_energy',
                    'factor_selective_within_energy',
                    'factor_selective_stability',
                    'factor_selective_utility',
                    'factor_selective_weight_max',
                    'factor_selective_effective_dims',
                    'factor_selective_image_std',
                    'factor_shared_repa_loss',
                    'factor_shared_self_distill_loss',
                    'factor_shared_variance_loss',
                    'factor_shared_similarity',
                    'factor_shared_source_ratio',
                    'factor_shared_between_energy',
                    'factor_shared_within_energy',
                    'factor_shared_image_std',
                    'factor_shared_confidence_max',
                    'factor_shared_confidence_entropy',
                    'factor_shared_contrastive_loss',
                    'factor_shared_contrastive_accuracy',
                    'factor_shared_positive_similarity',
                    'factor_shared_negative_similarity',
                    'factor_shared_relation_loss',
                    'factor_shared_global_relation_loss',
                    'factor_shared_local_relation_loss',
                    'factor_evolving_separation_loss',
                    'factor_evolving_pair_distance',
                    'factor_self_flow_full_loss',
                    'factor_self_flow_source_loss',
                    'factor_self_flow_full_similarity',
                    'factor_self_flow_source_similarity',
                    'factor_self_flow_teacher_pair_similarity',
                    'factor_self_flow_teacher_std',
                    'factor_self_flow_source_std',
                    'factor_self_flow_confidence_max',
                    'factor_pair_interaction_rms',
                    'factor_pair_random_align_loss',
                    'factor_pair_random_variance_loss',
                    'factor_pair_random_similarity',
                    'factor_pair_random_source_ratio',
                    'factor_pair_random_between_energy',
                    'factor_pair_random_within_energy',
                    'factor_pair_random_image_std',
                    'factor_pair_byol_align_loss',
                    'factor_pair_byol_variance_loss',
                    'factor_pair_byol_similarity',
                    'factor_pair_byol_source_ratio',
                    'factor_pair_byol_between_energy',
                    'factor_pair_byol_within_energy',
                    'factor_pair_byol_image_std',
                ):
                    if metric in losses:
                        logs[metric] = safe_scalar(losses[metric], accelerator)
                logs['factor_loss_scale'] = factor_loss_scale
                logs['factor_warmup'] = factor_regularization_warmup
                logs['factor_proj_warmup'] = factor_warmup
                logs['factor_decay'] = factor_decay
                logs['factorization_active'] = float(factorization_active)
                if args.factor_adversarial:
                    logs['factor_adversarial_grl_scale'] = (
                        factor_adversarial_grl_scale
                    )
                    logs['factor_adversarial_grl_ramp'] = (
                        factor_adversarial_grl_ramp
                    )
            if args.trajectory_invariance:
                for metric in (
                    'invariant_time_loss', 'invariant_noise_loss',
                    'invariant_image_variance_loss',
                    'invariant_spatial_variance_loss',
                    'invariant_covariance_loss', 'invariant_relation_loss',
                    'invariant_basis_loss',
                    'invariant_similarity', 'invariant_time_similarity',
                    'invariant_noise_similarity', 'invariant_image_std',
                    'invariant_spatial_std', 'invariant_covariance_offdiag',
                    'invariant_relation_gap',
                    'invariant_time_relation_gap',
                    'invariant_noise_relation_gap',
                    'invariant_within_energy', 'invariant_between_energy',
                    'invariant_source_ratio', 'invariant_mean_time_delta',
                    'invariant_time_reliability',
                    'invariant_noise_reliability',
                    'invariant_batch_fraction', 'invariant_views_per_group',
                ):
                    if metric in losses:
                        logs[metric] = safe_scalar(
                            losses[metric], accelerator
                        )
                logs['invariant_regularization'] = safe_scalar(
                    invariant_regularization, accelerator
                )
                logs['invariant_loss_scale'] = invariant_loss_scale
                logs['invariant_warmup'] = invariant_warmup
                logs['invariant_decay'] = invariant_decay
                logs['invariance_active'] = float(invariance_active)
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
    parser.add_argument("--skip-code-archive", action="store_true",
                        help="do not copy source code into the experiment directory")
    parser.add_argument("--code-archive-dir", type=str, default="code",
                        help="subdirectory under the experiment directory for source code archival")
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
    parser.add_argument(
        "--proj-use-factor-schedule", action="store_true",
        help="apply the TFCR warmup/decay schedule to standard REPA",
    )
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)
    ##### added 
    # skip-layer connection to improve the model's ability to capture long-range dependencies, improving the representation diversity
    parser.add_argument("--skip-layer-connection", action="store_true", help="skip-layer connection like unet")
    # block diversity loss block_diversity_loss
    parser.add_argument("--block-diversity-loss", action="store_true", help="block diversity difference loss")
    parser.add_argument("--block-diversity-loss-coeff", type=float, default=0.001, help="coefficient for block difference loss")
    # Persistent--Evolving trajectory factorization. Historical heads remain
    # training-only; native parameterization is an explicit opt-in sampling path.
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
    parser.add_argument(
        "--factor-orbit-mode",
        choices=[
            "legacy", "orthogonal", "time-only", "noise-only", "antithetic"
        ],
        default="legacy",
        help=(
            "legacy changes time and optionally noise; orthogonal changes "
            "exactly one of time/noise per pair; antithetic uses a shared "
            "timestep and epsilon/-epsilon"
        ),
    )
    parser.add_argument(
        "--factor-orbit-noise-only-prob", type=float, default=0.5,
        help="noise-only fraction when --factor-orbit-mode=orthogonal",
    )
    parser.add_argument(
        "--factor-share-cfg-dropout", action="store_true",
        help="reuse one classifier-free dropout decision across paired views",
    )
    parser.add_argument("--factor-min-delta-t", type=float, default=0.15)
    parser.add_argument("--factor-max-delta-t", type=float, default=0.7)
    parser.add_argument("--factor-inv-coeff", type=float, default=0.1)
    parser.add_argument("--factor-persistent-coeff", type=float, default=0.05,
                        help="weight for predicting the pair-symmetric target")
    parser.add_argument("--factor-evolving-coeff", type=float, default=0.05,
                        help="weight for predicting the current-view residual")
    parser.add_argument("--factor-recom-coeff", type=float, default=0.1)
    parser.add_argument(
        "--factor-regularization-start-steps",
        type=int,
        default=0,
        help=(
            "delay decomposition regularization while leaving scheduled "
            "REPA controlled by --factor-warmup-steps from step 0"
        ),
    )
    parser.add_argument(
        "--factor-reliable-target", action="store_true",
        help=(
            "gate the pair consensus by its between-source/within-orbit "
            "reliability"
        ),
    )
    parser.add_argument(
        "--factor-reliability-keep-ratio", type=float, default=1.0,
        help="fraction of the most reliable target channels retained",
    )
    parser.add_argument(
        "--factor-reliability-floor", type=float, default=0.0,
        help="minimum reliability eligible for top-channel selection",
    )
    parser.add_argument(
        "--factor-velocity-recomposition", action="store_true",
        help="decode swapped persistent/current evolving codes to velocity",
    )
    parser.add_argument(
        "--factor-velocity-recom-coeff", type=float, default=0.0,
        help="weight for task-sufficient swapped velocity reconstruction",
    )
    parser.add_argument(
        "--factor-native-parameterization", action="store_true",
        help=(
            "make the main velocity output a path-aware recomposition of "
            "decoded clean-source and noise factors"
        ),
    )
    parser.add_argument(
        "--factor-native-shuffle-source", action="store_true",
        help="negative control: shuffle only the clean-source branch targets",
    )
    parser.add_argument(
        "--factor-native-source-coeff", type=float, default=0.0,
        help="weight for SNR-aware ground-truth x0 factor supervision",
    )
    parser.add_argument(
        "--factor-native-noise-coeff", type=float, default=0.0,
        help="weight for SNR-aware ground-truth epsilon factor supervision",
    )
    parser.add_argument(
        "--factor-native-antithetic-coeff", type=float, default=0.0,
        help="weight for epsilon-branch antisymmetry across paired views",
    )
    parser.add_argument(
        "--factor-native-base-coeff", type=float, default=0.0,
        help="small auxiliary velocity loss retaining the standard SiT head",
    )
    parser.add_argument(
        "--factor-semantic-conditioning", action="store_true",
        help=(
            "align a low-dimensional source code to clean semantics and "
            "FiLM-condition the later velocity blocks"
        ),
    )
    parser.add_argument(
        "--factor-semantic-injection-scale", type=float, default=1.0,
        help="strength of source-conditioned FiLM; zero is a causal ablation",
    )
    parser.add_argument(
        "--factor-semantic-shuffle-targets", action="store_true",
        help="negative control: preserve pairs but shuffle clean source identity",
    )
    parser.add_argument(
        "--factor-semantic-repa-coeff", type=float, default=0.0,
        help="weight for source-subspace alignment to clean encoder tokens",
    )
    parser.add_argument(
        "--factor-semantic-source-consistency-coeff", type=float, default=0.0,
        help="optional direct cross-view source-code consistency weight",
    )
    parser.add_argument(
        "--factor-semantic-decorrelation-coeff", type=float, default=0.0,
        help="weak source/evolving code squared-cosine penalty",
    )
    parser.add_argument(
        "--factor-adversarial", action="store_true",
        help="remove timestep/orbit nuisances from persistent factor with GRL",
    )
    parser.add_argument(
        "--factor-adversarial-timestep-bins", type=int, default=8,
        help="number of discrete timestep classes used by nuisance probes",
    )
    parser.add_argument(
        "--factor-adversarial-grl-scale", type=float, default=0.1,
        help="maximum reversed-gradient multiplier on persistent features",
    )
    parser.add_argument(
        "--factor-adversarial-start-steps", type=int, default=20000,
        help="delay before adversarial gradients reach persistent features",
    )
    parser.add_argument(
        "--factor-adversarial-warmup-steps", type=int, default=30000,
        help="linear ramp duration for the reversed-gradient multiplier",
    )
    parser.add_argument(
        "--factor-adversarial-shuffle-labels", action="store_true",
        help="shuffle nuisance labels as an adversarial regularization control",
    )
    parser.add_argument(
        "--factor-adv-persistent-time-coeff", type=float, default=0.0,
    )
    parser.add_argument(
        "--factor-adv-persistent-orbit-coeff", type=float, default=0.0,
    )
    parser.add_argument(
        "--factor-probe-evolving-time-coeff", type=float, default=0.0,
    )
    parser.add_argument(
        "--factor-probe-evolving-orbit-coeff", type=float, default=0.0,
    )
    parser.add_argument(
        "--factor-clean-consensus", action="store_true",
        help=(
            "recover x0 from paired velocity predictions and align both "
            "views to their confidence-weighted clean-state consensus"
        ),
    )
    parser.add_argument(
        "--factor-clean-consensus-temperature", type=float, default=0.25,
        help="softmin temperature used to choose the reliable clean estimate",
    )
    parser.add_argument(
        "--factor-clean-consensus-shuffle-targets", action="store_true",
        help="negative control: align clean estimates to another source",
    )
    parser.add_argument(
        "--factor-clean-consensus-coeff", type=float, default=0.0,
    )
    parser.add_argument(
        "--factor-shared-clean", action="store_true",
        help=(
            "alias for --factor-clean-consensus in the shared-target "
            "paired trajectory suite"
        ),
    )
    parser.add_argument(
        "--factor-shared-clean-coeff", type=float, default=None,
        help=(
            "alias for --factor-clean-consensus-coeff; enables "
            "--factor-clean-consensus when set"
        ),
    )
    parser.add_argument(
        "--factor-shared-repa", action="store_true",
        help=(
            "align paired trajectory views to the same external clean-image "
            "REPA target as a factor-scheduled shared target"
        ),
    )
    parser.add_argument("--factor-shared-repa-coeff", type=float, default=0.0)
    parser.add_argument(
        "--factor-shared-self-distill", action="store_true",
        help=(
            "align full paired hidden features to a stop-grad reliable "
            "view consensus without selecting invariant dimensions"
        ),
    )
    parser.add_argument(
        "--factor-shared-source-depth", type=int, default=None,
        help="1-indexed full-feature shared target layer; defaults to factor source",
    )
    parser.add_argument(
        "--factor-shared-target-temperature", type=float, default=0.25,
        help="softmax temperature for reliable view consensus weights",
    )
    parser.add_argument(
        "--factor-shared-snr-power", type=float, default=1.0,
        help="power applied to clean-source reliability weights for shared targets",
    )
    parser.add_argument(
        "--factor-shared-self-distill-shuffle-targets", action="store_true",
        help="negative control: align full features to another source consensus",
    )
    parser.add_argument(
        "--factor-shared-self-distill-coeff", type=float, default=0.0,
    )
    parser.add_argument(
        "--factor-shared-variance-target", type=float, default=1.0,
    )
    parser.add_argument(
        "--factor-shared-variance-coeff", type=float, default=0.0,
    )
    parser.add_argument(
        "--factor-shared-contrastive", action="store_true",
        help=(
            "treat paired trajectory views as positives and other sources in "
            "the batch as negatives"
        ),
    )
    parser.add_argument(
        "--factor-shared-contrastive-temperature", type=float, default=0.2,
    )
    parser.add_argument(
        "--factor-shared-contrastive-coeff", type=float, default=0.0,
    )
    parser.add_argument(
        "--factor-shared-relation", action="store_true",
        help=(
            "match source-level and local spatial similarity structure across "
            "paired trajectory views"
        ),
    )
    parser.add_argument(
        "--factor-shared-relation-coeff", type=float, default=0.0,
    )
    parser.add_argument(
        "--factor-evolving-separation", action="store_true",
        help=(
            "encourage evolving/private codes to separate paired trajectory "
            "views by a cosine-distance margin"
        ),
    )
    parser.add_argument(
        "--factor-evolving-separation-margin", type=float, default=0.5,
    )
    parser.add_argument(
        "--factor-evolving-separation-coeff", type=float, default=0.0,
    )
    parser.add_argument(
        "--factor-self-flow-full-align", action="store_true",
        help=(
            "align the online full hidden trajectory features to an EMA "
            "teacher consensus"
        ),
    )
    parser.add_argument(
        "--factor-self-flow-full-coeff", type=float, default=0.0,
    )
    parser.add_argument(
        "--factor-self-flow-source-align", action="store_true",
        help=(
            "align only the persistent/source branch to an EMA teacher "
            "consensus and leave the evolving branch unaligned"
        ),
    )
    parser.add_argument(
        "--factor-self-flow-source-coeff", type=float, default=0.0,
    )
    parser.add_argument(
        "--factor-self-flow-shuffle-teacher", action="store_true",
        help=(
            "negative control: shift EMA teacher source identity before "
            "self-flow alignment"
        ),
    )
    parser.add_argument(
        "--factor-selective-invariance", action="store_true",
        help=(
            "learn a low-rank A3 invariant subspace without constraining its "
            "orthogonal complement"
        ),
    )
    parser.add_argument("--factor-selective-dim", type=int, default=128)
    parser.add_argument(
        "--factor-selective-source-depth", type=int, default=None,
        help="1-indexed selective readout layer; defaults to factor source",
    )
    parser.add_argument(
        "--factor-selective-weighting",
        choices=["uniform", "stability", "task"],
        default="task",
        help=(
            "weight subspace directions uniformly, by cross-view source "
            "stability, or by stability times FM-gradient utility"
        ),
    )
    parser.add_argument(
        "--factor-selective-shuffle-targets", action="store_true",
        help="negative control: align the subspace to another source",
    )
    parser.add_argument(
        "--factor-selective-shuffle-utility", action="store_true",
        help="negative control: permute FM utility across subspace directions",
    )
    parser.add_argument(
        "--factor-selective-variance-target", type=float, default=1.0,
    )
    parser.add_argument("--factor-selective-coeff", type=float, default=0.0)
    parser.add_argument(
        "--factor-selective-orth-coeff", type=float, default=0.0,
    )
    parser.add_argument(
        "--factor-selective-variance-coeff", type=float, default=0.0,
    )
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
    parser.add_argument(
        "--factor-pair-interaction", action="store_true",
        help="inject shared residual context between paired trajectory views",
    )
    parser.add_argument("--factor-pair-interaction-depth", type=int, default=None)
    parser.add_argument(
        "--factor-pair-interaction-scale", type=float, default=1.0,
    )
    parser.add_argument(
        "--factor-pair-interaction-hidden-ratio", type=float, default=0.25,
    )
    parser.add_argument(
        "--factor-pair-interaction-self-prob", type=float, default=0.0,
        help="probability of using self context instead of pair consensus",
    )
    parser.add_argument(
        "--factor-pair-interaction-detach-context", action="store_true",
        help="stop gradients through the paired consensus context",
    )
    parser.add_argument(
        "--factor-pair-random-align", action="store_true",
        help="align fixed random projections of paired trajectory features",
    )
    parser.add_argument("--factor-pair-random-align-coeff", type=float, default=0.0)
    parser.add_argument("--factor-pair-random-variance-coeff", type=float, default=0.0)
    parser.add_argument("--factor-pair-random-align-dim", type=int, default=256)
    parser.add_argument("--factor-pair-random-align-seed", type=int, default=2027)
    parser.add_argument(
        "--factor-pair-byol-align", action="store_true",
        help="align paired trajectory readouts with a BYOL-style predictor",
    )
    parser.add_argument("--factor-pair-byol-align-coeff", type=float, default=0.0)
    parser.add_argument("--factor-pair-byol-variance-coeff", type=float, default=0.0)
    parser.add_argument("--factor-pair-alignment-dim", type=int, default=256)
    parser.add_argument(
        "--factor-pair-alignment-predictor-dim", type=int, default=1024,
    )
    parser.add_argument(
        "--factor-pair-align-variance-target", type=float, default=1.0,
    )
    # Teacher-free trajectory-orbit invariant subspace.  The full backbone is
    # not forced to be invariant; only this low-dimensional readout is aligned.
    parser.add_argument(
        "--trajectory-invariance", action="store_true",
        help="regularize a teacher-free trajectory-invariant subspace",
    )
    parser.add_argument("--invariant-dim", type=int, default=256)
    parser.add_argument("--invariant-projector-dim", type=int, default=1024)
    parser.add_argument(
        "--invariant-projector-type", choices=["linear", "mlp"],
        default="linear",
        help="linear is a true readout subspace; mlp is an ablation",
    )
    parser.add_argument(
        "--invariant-source-depth", type=int, default=None,
        help="1-indexed readout layer; defaults to encoder depth",
    )
    parser.add_argument("--invariant-min-delta-t", type=float, default=0.05)
    parser.add_argument("--invariant-max-delta-t", type=float, default=0.2)
    parser.add_argument(
        "--invariant-max-t", type=float, default=0.8,
        help="largest timestep supervised without an external teacher",
    )
    parser.add_argument(
        "--invariant-snr-power", type=float, default=1.0,
        help="power applied to clean-source reliability weights",
    )
    parser.add_argument("--invariant-time-coeff", type=float, default=0.1)
    parser.add_argument("--invariant-noise-coeff", type=float, default=0.1)
    parser.add_argument(
        "--invariant-image-variance-coeff", type=float, default=0.02,
        help="anti-collapse variance floor across source images",
    )
    parser.add_argument(
        "--invariant-spatial-variance-coeff", type=float, default=0.02,
        help="anti-collapse variance floor across spatial tokens",
    )
    parser.add_argument("--invariant-covariance-coeff", type=float, default=0.001)
    parser.add_argument(
        "--invariant-basis-coeff", type=float, default=0.01,
        help="orthogonality weight for a non-redundant linear basis",
    )
    parser.add_argument(
        "--invariant-relation-coeff", type=float, default=0.05,
        help="weight for local patch-relation consistency",
    )
    parser.add_argument("--invariant-variance-target", type=float, default=1.0)
    parser.add_argument(
        "--invariant-spatial-variance-target", type=float, default=0.5
    )
    parser.add_argument("--invariant-warmup-steps", type=int, default=10000)
    parser.add_argument("--invariant-decay-start", type=int, default=-1)
    parser.add_argument("--invariant-decay-end", type=int, default=-1)
    parser.add_argument("--invariant-min-loss-scale", type=float, default=0.0)
    parser.add_argument(
        "--invariant-loss-frequency", type=int, default=1,
        help="activate three-view invariant training every N optimizer steps",
    )
    parser.add_argument(
        "--invariant-batch-ratio", type=float, default=0.375,
        help="fraction receiving two extra views; 0.375 matches 0.75 two-view FLOPs",
    )
    parser.add_argument(
        "--invariant-view-control-only", action="store_true",
        help="keep three-view training active with zero auxiliary weights",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
