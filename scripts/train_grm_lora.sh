
# Set up Weights & Biases
export WANDB_ENTITY="base-model-values"
export WANDB_PROJECT="Ray2333_Gemma2-2B_repro"

devices=0,1,2
n_gpu=3
dataset_name='Skywork/Skywork-Reward-Preference-80K-v0.2'
dataset_mode='400k'
base_model='Ray2333/GRM-Gemma2-2B-sftreg'
attn_implementation='eager' # Eager recommended for Gemma 2, otherwise flash_attention_2 is default
wandb_name="GRM_seed1"
log_dir='../save_reward_models'
main_process_port=9995

learning_rate=1e-5
lora_r=32
lora_alpha=64
max_length=1024
num_train_epochs=2
gradient_accumulation_steps=4

weight_ratio=0.01
layer_type='mlp' 
sft_only=True
reference_free=True

# For saving checkpoints
output_dir="../save_reward_models/GRM_Gemma2-2B_ckpts"
save_strategy="steps"         # "epoch" is ok too; for analysis, steps is nicer
save_steps=1000               # adjust to taste
save_total_limit=12           # keep last N to cap disk
evaluation_strategy="steps"   # align eval with save so best-model works
eval_steps=1000
logging_steps=100
load_best_model_at_end=True
metric_for_best_model="eval_reward_accuracy"
greater_is_better=True
save_safetensors=True

cd ../reward_models
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port} run_grm_reward_train.py \
  --base_model ${base_model} --wandb_name ${wandb_name} --log_dir ${log_dir} \
  --attn_implementation ${attn_implementation} \
  --num_train_epochs ${num_train_epochs} --max_length ${max_length} \
  --use_lora True --gradient_accumulation_steps ${gradient_accumulation_steps} \
  --learning_rate ${learning_rate} \
  --dataset ${dataset_name} --dataset_mode ${dataset_mode} \
  --weight_ratio ${weight_ratio} --layer_type ${layer_type} \
  --reference_free ${reference_free} --sft_only ${sft_only} \
  --output_dir ${output_dir} \
  --save_strategy ${save_strategy} --save_steps ${save_steps} --save_total_limit ${save_total_limit} \
  --evaluation_strategy ${evaluation_strategy} --eval_steps ${eval_steps} \
  --logging_steps ${logging_steps} \
  --load_best_model_at_end ${load_best_model_at_end} \
  --metric_for_best_model ${metric_for_best_model} \
  --greater_is_better ${greater_is_better} \
  --save_safetensors ${save_safetensors} \
  --report_to wandb
