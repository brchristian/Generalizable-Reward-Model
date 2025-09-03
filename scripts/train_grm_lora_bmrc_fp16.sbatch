#!/bin/bash

#SBATCH -D /well/summerfield/projects/base_model_values

#SBATCH -J train_grm_lora_bmrc
#SBATCH -A summerfield.prj
#SBATCH -p gpu_short
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# (Optional) constrain to newer GPUs if you want (uncomment one that matches your cluster)
# #SBATCH --constraint=a100
# #SBATCH --constraint=l40s

# Load your environment
module purge
# module load cuda/12.1  # only if your env needs a module
# module load anaconda   # or mamba
source ~/.bashrc
conda activate Ray2333_GRM   # env should have torch, transformers, accelerate, peft, etc.

# Set up Weights & Biases
export WANDB_ENTITY="base-model-values"
export WANDB_PROJECT="Ray2333_Gemma2-2B_repro"
export HUGGINGFACE_HUB_TOKEN=${HF_TOKEN}

# Create logs directory if needed
mkdir -p logs

echo "------------------------------------------------" 
echo "Run on host: "`hostname` 
echo "Operating system: "`uname -s` 
echo "Username: "`whoami` 
echo "Started at: "`date` 
echo "------------------------------------------------" 

# Navigate to the directory with your script (if needed)
cd /well/summerfield/projects/base_model_values/Ray233_RM_training/reward_models

python run_grm_reward_train.py \
  --bf16 False \
  --base_model "Ray2333/GRM-Gemma2-2B-sftreg" \
  --dataset "Skywork/Skywork-Reward-Preference-80K-v0.2" \
  --dataset_mode "400k" \
  --per_device_train_batch_size 1 \
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
  --wandb_name "GRM_fp16_seed1" \
  --output_dir ../save_reward_models/GRM_Gemma2-2B_ckpts \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 12 \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --logging_steps 100 \
  --load_best_model_at_end True \
  --metric_for_best_model eval_reward_accuracy \
  --greater_is_better False \
  --save_safetensors True