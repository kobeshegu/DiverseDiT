#!/usr/bin/env bash

# >>> conda initialize >>>
__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    . "/opt/conda/etc/profile.d/conda.sh"
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate /root/anaconda3/envs/repa

cd "/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/codes/DiverseDiT"

YOUR_DATA_DIR="/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/datasets/mengpingdata_0907"
YOUR_EXPERIMENT_NAME="traj_patch_sit_b2_no_repa_seed0"
PRETRAINED_MODEL_PATH="/inspire/l20d/project/sais-inspire-l20d/public/yangmengping/pretrained_models"


# step1: training
export MASTER_PORT=29501
CUDA_VISIBLE_DEVICES=0 accelerate launch train.py \
  --resolution=256 \
  --batch-size=256 \
  --report-to="none" \
  --allow-tf32 \
  --num-workers=16 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-B/2" \
  --enc-type="none" \
  --proj-coeff=0 \
  --encoder-depth=8 \
  --output-dir="results" \
  --exp-name="$YOUR_EXPERIMENT_NAME" \
  --data-dir="$YOUR_DATA_DIR" \
  --pretrained-model-path="$PRETRAINED_MODEL_PATH" \
  --max-train-steps=450000 \
  --checkpointing-steps=5000 \
  --skip-training-samples \
  --traj-loss \
  --traj-objective=patch \
  --traj-loss-coeff=0.1 \
  --traj-warmup-steps=10000 \
  --traj-loss-frequency=2 \
  --traj-batch-ratio=0.25 \
  --traj-num-steps=3 \
  --traj-anchors=0.85,0.50,0.15 \
  --traj-depth=8 \
  --traj-sampler=jittered \
  --traj-patch-sim-coeff=1.0 \
  --traj-patch-std-coeff=1.0 \
  --traj-patch-cov-coeff=0.04


# step2: generation configs
MODEL="SiT-B/2"
PER_PROC_BATCH_SIZE=32
NUM_FID_SAMPLES=50000
PATH_TYPE="linear"
MODE="sde"
NUM_STEPS=250
CFG_SCALE=1.8
GUIDANCE_HIGH=0.7
RESOLUTION=256
VAE="mse"
GLOBAL_SEED=0
SAMPLE_DIR="sampled_images/$YOUR_EXPERIMENT_NAME"
CKPT="results/$YOUR_EXPERIMENT_NAME/checkpoints/0450000.pt"
ENCODER_DEPTH=8


# step3: generate images
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --master_port=29501 generate.py \
  --model="$MODEL" \
  --num-fid-samples="$NUM_FID_SAMPLES" \
  --ckpt="$CKPT" \
  --path-type="$PATH_TYPE" \
  --encoder-depth="$ENCODER_DEPTH" \
  --projector-embed-dims=none \
  --per-proc-batch-size="$PER_PROC_BATCH_SIZE" \
  --mode="$MODE" \
  --num-steps="$NUM_STEPS" \
  --cfg-scale="$CFG_SCALE" \
  --guidance-high="$GUIDANCE_HIGH" \
  --sample-dir="$SAMPLE_DIR" \
  --resolution="$RESOLUTION" \
  --vae="$VAE" \
  --pretrained-model-path="$PRETRAINED_MODEL_PATH"


# step4: package samples and calculate evaluation metrics
python npz_convert.py \
  --model="$MODEL" \
  --ckpt="$CKPT" \
  --sample-dir="$SAMPLE_DIR" \
  --num-fid-samples="$NUM_FID_SAMPLES" \
  --resolution="$RESOLUTION" \
  --vae="$VAE" \
  --cfg-scale="$CFG_SCALE" \
  --global-seed="$GLOBAL_SEED" \
  --mode="$MODE"

conda activate /root/anaconda3/envs/scale_rae

python evaluator_tf.py \
  /inspire/l20d/project/sais-inspire-l20d/public/yangmengping/datasets/datasets/VIRTUAL_imagenet256_labeled.npz \
  sampled_images/$YOUR_EXPERIMENT_NAME/SiT-B-2-0450000-size-256-vae-mse-cfg-1.8-seed-0-sde.npz
