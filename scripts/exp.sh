# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        . "/opt/conda/etc/profile.d/conda.sh"
    else
        export PATH="/opt/conda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda info --envs
conda activate your_env_name


YOUR_DATA_DIR=""
YOUR_EXPERIMENT_NAME=""
PRETRAINED_MODEL_PATH=""


# step1: training
export MASTER_PORT=29501
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train.py \
  --resolution=256 \
  --batch-size=256 \
  --report-to="tensorboard" \
  --allow-tf32 \
  --num-workers=16 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-B/2" \
  --enc-type="dinov2-vit-b" \
  --proj-coeff=0.5 \
  --encoder-depth=8 \
  --output-dir="results" \
  --exp-name=$YOUR_EXPERIMENT_NAME \
  --data-dir=$YOUR_DATA_DIR \
  --pretrained-model-path $PRETRAINED_MODEL_PATH \
  --max-train-steps=450000 \
  --checkpointing-steps=5000 \


## step2: evaluation configs for generating images
MODEL="SiT-B/2"
PER_PROC_BATCH_SIZE=32
NUM_FID_SAMPLES=50000
PATH_TYPE="linear"
MODE="sde"
NUM_STEPS=250
# cfg_scale=1.0 (without CFG),by default we use cfg_scale=1.8 with guidance interval.
CFG_SCALE=1.8
CLS_CFG_SCALE=2.3
GUIDANCE_HIGH=0.7
RESOLUTION=256
VAE="mse"
GLOBAL_SEED=0
SAMPLE_DIR="sampled_images/$YOUR_EXPERIMENT_NAME"
CKPT="results/$YOUR_EXPERIMENT_NAME/checkpoints/0400000.pt"
ENCODER_DEPTH=8

# generating images
torchrun --nnodes=1 --nproc_per_node=8 --master_port=29501  generate.py \
  --model $MODEL \
  --num-fid-samples $NUM_FID_SAMPLES \
  --ckpt $CKPT \
  --path-type $PATH_TYPE \
  --encoder-depth $ENCODER_DEPTH \
  --projector-embed-dims 768 \
  --per-proc-batch-size $PER_PROC_BATCH_SIZE \
  --mode $MODE \
  --num-steps $NUM_STEPS \
  --cfg-scale $CFG_SCALE \
  --guidance-high $GUIDANCE_HIGH \
  --sample-dir $SAMPLE_DIR \
  --resolution $RESOLUTION \
  --vae $VAE \
  --pretrained-model-path $PRETRAINED_MODEL_PATH \


# step3: generating .npz files for quantitative evaluation
python npz_convert.py \
    --model $MODEL \
    --ckpt $CKPT \
    --sample-dir $SAMPLE_DIR \
    --num-fid-samples $NUM_FID_SAMPLES \
    --resolution $RESOLUTION \
    --vae $VAE \
    --cfg-scale $CFG_SCALE \
    --global-seed $GLOBAL_SEED \
    --mode $MODE



# step4: calculating FID, IS, etc, the results will be saved in your_path_here/your_experiment_name.txt
conda init
conda activate your_fid_env_name

python evaluator.py \
 your_path_here/VIRTUAL_imagenet256_labeled.npz \
 your_path_here/your_experiment_name.npz