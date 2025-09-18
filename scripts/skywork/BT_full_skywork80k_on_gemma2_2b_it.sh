#!/bin/bash

#SBATCH -J bt_full_skywork80k_on_gemma_2_2b_it
#SBATCH -D /well/summerfield/projects/base_model_values
#SBATCH -A summerfield.prj
#SBATCH -p gpu_long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 60:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --constraint=a100

# ============================================
# CONFIGURATION VARIABLES (edit these!)
# ============================================
BASE_MODEL_PATH="google/gemma-2-2b-it"
DATASET='Skywork/Skywork-Reward-Preference-80K-v0.2'
DATASET_SHORT_NAME="Skywork80k"
SEED=1
# ============================================


BASE_MODEL_NAME="${BASE_MODEL_PATH##*/}"  # Automatically extract model name after last slash
FULL_RUN_NAME="BT_full_${DATASET_SHORT_NAME}_on_${BASE_MODEL_NAME}_seed${SEED}"

# Load your environment
module purge
source ~/.bashrc
conda activate Ray2333_GRM

# Set up Weights & Biases
export WANDB_ENTITY="base-model-values"
export WANDB_PROJECT="BT_LoRA_unified"
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

# Navigate to the directory with your script (if needed)
# BMRC
cd /well/summerfield/projects/base_model_values/Ray233_RM_training/reward_models

python run_reward_models_train.py \
  --base_model "${BASE_MODEL_PATH}" \
  --dataset "${DATASET}" \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 64 \
  --learning_rate 5e-6 \
  --num_train_epochs 1 \
  --max_length 3000 \
  --use_lora False \
  --report_to wandb \
  --wandb_name "${FULL_RUN_NAME}" \
  --output_dir "../save_reward_models/${FULL_RUN_NAME}" \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 50 \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --logging_steps 100 \
  --load_best_model_at_end True \
  --metric_for_best_model eval_accuracy \
  --greater_is_better True \
  --save_safetensors True \
  --seed ${SEED} \
