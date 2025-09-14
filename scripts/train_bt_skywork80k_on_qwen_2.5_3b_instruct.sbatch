#!/bin/bash

#SBATCH -D /well/summerfield/projects/base_model_values
#SBATCH -J bt_lora_skywork_on_qwen_2.5_3b_instruct
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
MODEL_TYPE="BT" # or "BT" for Bradley-Terry
TRAINING_TYPE="LoRA" # or "Full" for full fine-tuning TODO: IMPLEMENT
DATASET="Skywork/Skywork-Reward-Preference-80K-v0.2"
DATASET_NAME="skywork80k"
BASE_MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
SEED=1
# ============================================


BASE_MODEL_NAME="${BASE_MODEL_PATH##*/}"  # Automatically extract model name after last slash
FULL_RUN_NAME="${MODEL_TYPE}_${TRAINING_TYPE}_${DATASET_NAME}_on_${BASE_MODEL_NAME}_seed${SEED}"

# Load your environment
module purge
source ~/.bashrc
conda activate Ray2333_GRM

# Set up Weights & Biases
export WANDB_ENTITY="base-model-values"
export WANDB_PROJECT="Ray2333_Gemma2-2B_repro"
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

# Determine script name and prefix based on training type
if [ "${MODEL_TYPE}" = "GRM" ]; then
    SCRIPT_NAME="run_grm_reward_train.py"
else
    SCRIPT_NAME="run_reward_models_train.py"
fi

# Navigate to the directory with your script (if needed)
# BMRC
cd /well/summerfield/projects/base_model_values/Ray233_RM_training/reward_models
# NOOPER
# cd /mnt/quick/brian/base_model_values/reward_model_training/Ray2333_GRM-gemma2-2B-rewardmodel-ft/Generalizable-Reward-Model/reward_models

# For automatically pushing to HuggingFace Hub (optional)
# --push_to_hub True \
# --hub_model_id "brianchristian/${FULL_RUN_NAME}" \
# --hub_private_repo True \
# --hub_strategy end \

python ${SCRIPT_NAME} \
  --base_model "${BASE_MODEL_PATH}" \
  --dataset "${DATASET}" \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --num_train_epochs 2 \
  --max_length 1024 \
  --bf16 True \
  --gradient_checkpointing True \
  --use_lora True \
  --lora_r 32 \
  --lora_alpha 64 \
  --weight_ratio 0.01 \
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
