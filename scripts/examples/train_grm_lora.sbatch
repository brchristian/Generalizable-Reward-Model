#!/bin/bash

#SBATCH -J grm_lora_training
#SBATCH -A your-slurm-account
#SBATCH -p gpu_long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# ============================================
# CONFIGURATION — edit these for your run
# ============================================
BASE_MODEL_PATH="Ray2333/GRM-Gemma2-2B-sftreg"
DATASET="Skywork/Skywork-Reward-Preference-80K-v0.2"
SEED=1
# ============================================

BASE_MODEL_NAME="${BASE_MODEL_PATH##*/}"
FULL_RUN_NAME="GRM_LoRA_on_${BASE_MODEL_NAME}_seed${SEED}"

# Load your environment
module purge
source ~/.bashrc
conda activate grm-training

# Set up Weights & Biases (optional)
export WANDB_ENTITY="your-wandb-entity"
export WANDB_PROJECT="your-wandb-project"
export HUGGINGFACE_HUB_TOKEN=${HF_TOKEN}

# Create logs directory if needed
mkdir -p logs

echo "------------------------------------------------"
echo "Run on host: $(hostname)"
echo "Operating system: $(uname -s)"
echo "Username: $(whoami)"
echo "Started at: $(date)"
echo "------------------------------------------------"
echo "Model: ${BASE_MODEL_PATH}"
echo "Seed: ${SEED}"
echo "------------------------------------------------"

cd reward_models

python run_grm_reward_train.py \
  --base_model "${BASE_MODEL_PATH}" \
  --dataset "${DATASET}" \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --num_train_epochs 2 \
  --max_length 1024 \
  --bf16 True \
  --gradient_checkpointing True \
  --attn_implementation eager \
  --use_lora True \
  --lora_r 32 \
  --lora_alpha 64 \
  --weight_ratio 0.01 \
  --layer_type mlp \
  --sft_only True \
  --reference_free True \
  --report_to wandb \
  --wandb_name "${FULL_RUN_NAME}" \
  --output_dir "../save_reward_models/${FULL_RUN_NAME}" \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 12 \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --logging_steps 100 \
  --load_best_model_at_end True \
  --metric_for_best_model eval_reward_accuracy \
  --greater_is_better True \
  --save_safetensors True \
  --seed ${SEED}
