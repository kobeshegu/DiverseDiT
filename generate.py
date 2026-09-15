# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from models.sit import SiT_models
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from samplers import euler_sampler, euler_maruyama_sampler
from utils import load_legacy_checkpoints, download_model



def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:cd
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load model:
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    latent_size = args.resolution // 8
    projector_spec = args.projector_embed_dims.strip().lower()
    use_projection = args.projection and projector_spec not in {"", "none", "null"}
    z_dims = (
        [int(z_dim) for z_dim in args.projector_embed_dims.split(',')]
        if use_projection else []
    )
    model = SiT_models[args.model](
        path_type=args.path_type,
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = True,
        class_dropout_prob=0.1,
        z_dims=z_dims,
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
        factor_semantic_injection_scale=(
            args.factor_semantic_injection_scale
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
        trajectory_invariance=args.trajectory_invariance,
        invariant_dim=args.invariant_dim,
        invariant_projector_dim=args.invariant_projector_dim,
        invariant_source_depth=args.invariant_source_depth,
        invariant_projector_type=args.invariant_projector_type,
        **block_kwargs,
    ).to(device)
    # Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
    ckpt_path = args.ckpt
    if ckpt_path is None:
        args.ckpt = 'SiT-XL-2-256x256.pt'
        assert args.model == 'SiT-XL/2'
        assert use_projection
        assert len(args.projector_embed_dims.split(',')) == 1
        assert int(args.projector_embed_dims.split(',')[0]) == 768
        state_dict = download_model('last.pt')
    else:
        checkpoint = torch.load(
            ckpt_path,
            map_location=f'cuda:{device}',
            weights_only=False,
        )
        state_dict = checkpoint['ema']
        has_factorization_head = any(
            key.startswith('factorization_head.') for key in state_dict
        )
        if has_factorization_head != args.trajectory_factorization:
            raise ValueError(
                "--trajectory-factorization must match whether the checkpoint "
                "contains a factorization head"
            )
        has_velocity_recomposition = any(
            key.startswith("factorization_head.persistent_velocity_decoder.")
            for key in state_dict
        )
        if has_velocity_recomposition != args.factor_velocity_recomposition:
            raise ValueError(
                "--factor-velocity-recomposition must match whether the "
                "checkpoint contains the velocity recomposition head"
            )
        has_native_source_decoder = any(
            key.startswith("factorization_head.native_source_decoder.")
            for key in state_dict
        )
        has_native_noise_decoder = any(
            key.startswith("factorization_head.native_noise_decoder.")
            for key in state_dict
        )
        if has_native_source_decoder != has_native_noise_decoder:
            raise ValueError(
                "checkpoint contains only one native source/noise decoder"
            )
        has_native_parameterization = has_native_source_decoder
        if has_native_parameterization != args.factor_native_parameterization:
            raise ValueError(
                "--factor-native-parameterization must match whether the "
                "checkpoint contains native source/noise decoders"
            )
        has_semantic_conditioning = any(
            key == "factorization_head.semantic_source_gate"
            for key in state_dict
        )
        if has_semantic_conditioning != args.factor_semantic_conditioning:
            raise ValueError(
                "--factor-semantic-conditioning must match whether the "
                "checkpoint contains semantic source conditioning"
            )
        has_adversarial = any(
            key.startswith("factorization_head.persistent_time_discriminator.")
            for key in state_dict
        )
        if has_adversarial != args.factor_adversarial:
            raise ValueError(
                "--factor-adversarial must match whether the checkpoint "
                "contains nuisance heads"
            )
        if has_adversarial:
            checkpoint_timestep_bins = state_dict[
                "factorization_head.persistent_time_discriminator.3.weight"
            ].shape[0]
            if checkpoint_timestep_bins != args.factor_adversarial_timestep_bins:
                raise ValueError(
                    "--factor-adversarial-timestep-bins does not match "
                    f"checkpoint ({checkpoint_timestep_bins})"
                )
        has_selective_invariance = any(
            key.startswith("factor_selective_invariance_head.")
            for key in state_dict
        )
        if has_selective_invariance != args.factor_selective_invariance:
            raise ValueError(
                "--factor-selective-invariance must match whether the "
                "checkpoint contains its selective projector"
            )
        if has_selective_invariance:
            checkpoint_selective_dim = state_dict[
                "factor_selective_invariance_head.projector.weight"
            ].shape[0]
            if checkpoint_selective_dim != args.factor_selective_dim:
                raise ValueError(
                    "--factor-selective-dim does not match checkpoint "
                    f"({checkpoint_selective_dim})"
                )
        has_invariance_head = any(
            key.startswith('invariance_head.') for key in state_dict
        )
        if has_invariance_head != args.trajectory_invariance:
            raise ValueError(
                "--trajectory-invariance must match whether the checkpoint "
                "contains an invariance head"
            )
        if has_invariance_head:
            checkpoint_projector_type = (
                "linear"
                if "invariance_head.projector.weight" in state_dict
                else "mlp"
            )
            if checkpoint_projector_type != args.invariant_projector_type:
                raise ValueError(
                    "--invariant-projector-type does not match checkpoint "
                    f"({checkpoint_projector_type})"
                )
            if checkpoint_projector_type == "linear":
                checkpoint_invariant_dim = state_dict[
                    "invariance_head.projector.weight"
                ].shape[0]
                checkpoint_projector_dim = args.invariant_projector_dim
            else:
                checkpoint_invariant_dim = state_dict[
                    "invariance_head.projector.3.weight"
                ].shape[0]
                checkpoint_projector_dim = state_dict[
                    "invariance_head.projector.1.weight"
                ].shape[0]
            if checkpoint_invariant_dim != args.invariant_dim:
                raise ValueError(
                    "--invariant-dim does not match checkpoint "
                    f"({checkpoint_invariant_dim})"
                )
            if (
                checkpoint_projector_type == "mlp"
                and checkpoint_projector_dim != args.invariant_projector_dim
            ):
                raise ValueError(
                    "--invariant-projector-dim does not match checkpoint "
                    f"({checkpoint_projector_dim})"
                )
        if any(
            key.startswith('factorization_head.recomposer.')
            for key in state_dict
        ):
            raise ValueError(
                "checkpoint uses the pre-balanced TFCR head; sample it with "
                "the matching historical commit"
            )
    if args.legacy:
        state_dict = load_legacy_checkpoints(
            state_dict=state_dict, encoder_depth=args.encoder_depth
            )
    model.load_state_dict(state_dict)
    model.eval()  # important!
    pretrained_vae_path = f"{args.pretrained_model_path}/sd-vae-ft-{args.vae}"
    vae = AutoencoderKL.from_pretrained(pretrained_vae_path).to(device)
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.resolution}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}-{args.mode}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}/images"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"projector Parameters: {sum(p.numel() for p in model.projectors.parameters()):,}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # Sample images:
        sampling_kwargs = dict(
            model=model, 
            latents=z,
            y=y,
            num_steps=args.num_steps, 
            heun=args.heun,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            path_type=args.path_type,
        )
        with torch.no_grad():
            if args.mode == "sde":
                samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
            elif args.mode == "ode":
                samples = euler_sampler(**sampling_kwargs).to(torch.float32)
            else:
                raise NotImplementedError()

            latents_scale = torch.tensor(
                [0.18215, 0.18215, 0.18215, 0.18215, ]
                ).view(1, 4, 1, 1).to(device)
            latents_bias = -torch.tensor(
                [0., 0., 0., 0.,]
                ).view(1, 4, 1, 1).to(device)
            samples = vae.decode((samples -  latents_bias) / latents_scale).sample
            samples = (samples + 1) / 2.
            samples = torch.clamp(
                255. * samples, 0, 255
                ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    # if rank == 0:
    #     create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
    #     print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=0)

    # precision
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    # logging/saving:
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a SiT checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="samples")

    # model
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)

    # vae
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")  # Choice doesn't affect training
    parser.add_argument("--pretrained-model-path", type=str, default="/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/public/yangmengping/pretrained_models")
  
    # number of samples
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)

    # sampling related hyperparameters
    parser.add_argument("--mode", type=str, default="ode")
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument(
        "--projector-embed-dims",
        type=str,
        default="768,1024",
        help="comma-separated REPA dimensions, or 'none' for no-REPA checkpoints",
    )
    parser.add_argument("--projection", action=argparse.BooleanOptionalAction, default=True,
                        help="instantiate REPA projection heads from the checkpoint")
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False) # only for ode
    parser.add_argument("--guidance-low", type=float, default=0.)
    parser.add_argument("--guidance-high", type=float, default=1.)

    # will be deprecated
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False) # only for ode
    ##### added 
    # skip-layer connection to improve the model's ability to capture long-range dependencies, improving the representation diversity
    parser.add_argument("--skip-layer-connection", action="store_true", help="skip-layer connection like unet")
    # block diversity loss block_diversity_loss
    parser.add_argument("--block-diversity-loss", action="store_true", help="block diversity difference loss")
    parser.add_argument("--trajectory-factorization", action="store_true",
                        help="instantiate persistent/evolving heads from the checkpoint")
    parser.add_argument("--factor-dim", type=int, default=256)
    parser.add_argument("--factor-projector-dim", type=int, default=1024)
    parser.add_argument("--factor-source-depth", type=int, default=None)
    parser.add_argument("--factor-target-depth", type=int, default=None)
    parser.add_argument("--factor-transition", action="store_true")
    parser.add_argument(
        "--factor-velocity-recomposition", action="store_true",
        help="instantiate the task-sufficient decoder stored in the checkpoint",
    )
    parser.add_argument(
        "--factor-native-parameterization", action="store_true",
        help="sample through checkpoint x0/epsilon factor recomposition heads",
    )
    parser.add_argument(
        "--factor-semantic-conditioning", action="store_true",
        help="sample through checkpoint semantic source FiLM modules",
    )
    parser.add_argument(
        "--factor-semantic-injection-scale", type=float, default=1.0,
        help="semantic FiLM scale; set zero for a causal sampling ablation",
    )
    parser.add_argument(
        "--factor-adversarial", action="store_true",
        help="instantiate adversarial nuisance heads stored in the checkpoint",
    )
    parser.add_argument(
        "--factor-adversarial-timestep-bins", type=int, default=8,
    )
    parser.add_argument(
        "--factor-selective-invariance", action="store_true",
        help="instantiate the A3 selective projector stored in the checkpoint",
    )
    parser.add_argument("--factor-selective-dim", type=int, default=128)
    parser.add_argument("--factor-selective-source-depth", type=int, default=None)
    parser.add_argument(
        "--trajectory-invariance", action="store_true",
        help="instantiate the invariant projector stored in the checkpoint",
    )
    parser.add_argument("--invariant-dim", type=int, default=256)
    parser.add_argument("--invariant-projector-dim", type=int, default=1024)
    parser.add_argument("--invariant-source-depth", type=int, default=None)
    parser.add_argument(
        "--invariant-projector-type", choices=["linear", "mlp"],
        default="linear",
    )

    args = parser.parse_args()
    main(args)
