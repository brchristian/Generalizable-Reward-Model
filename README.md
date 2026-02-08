# Reward Model Training

Training code for reward models used in [Reward Models Inherit Value Biases from Pretraining](https://arxiv.org/abs/2601.20838) (Christian et al., ICLR 2026).

This is a fork of [Generalizable Reward Model (GRM)](https://github.com/YangRui2015/Generalizable-Reward-Model) by [Yang et al. (NeurIPS 2024)](https://arxiv.org/abs/2406.10216), with the following additions:

- **Dataset subsampling** via `--dataset_step_size` for controlled data ablations
- **Log-schedule checkpointing** (`--use_log_overlay`) that saves at powers of 2 overlaid on a fixed cadence, enabling analysis of training dynamics
- **Checkpoint-0 saving** to capture the model state before any training
- **Value head persistence** for GRM models (save/load `v_head.pt` alongside LoRA adapters)
- **HF Hub integration** with `--push_to_hub` and a `PromoteAndTagCallback` that promotes each checkpoint to the repo root with an immutable tag
- **Configurable attention** via `--attn_implementation` (default: `sdpa`)
- **`--max_steps` support** for step-based (rather than epoch-based) training

## Trained Models

Trained model checkpoints are available on Hugging Face Hub: [Oxford-HIPlab collection](https://huggingface.co/collections/Oxford-HIPlab/reward-models-inherit-value-biases-from-pretraining-iclr2026).

## Setup

```bash
conda env create -f environment.yml
conda activate grm-training
```

## Training

### Bradley-Terry (BT) Reward Model

```bash
cd reward_models

python run_reward_models_train.py \
  --base_model "Qwen/Qwen2.5-3B-Instruct" \
  --dataset "llm-blender/Unified-Feedback" \
  --dataset_step_size 64 \
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
  --report_to wandb \
  --wandb_name "BT_LoRA_example" \
  --output_dir "../save_reward_models/BT_LoRA_example" \
  --save_strategy steps \
  --save_steps 1000 \
  --eval_steps 1000 \
  --logging_steps 100 \
  --save_safetensors True \
  --seed 1
```

### GRM (Generalizable Reward Model)

```bash
cd reward_models

python run_grm_reward_train.py \
  --base_model "Ray2333/GRM-Gemma2-2B-sftreg" \
  --dataset "Skywork/Skywork-Reward-Preference-80K-v0.2" \
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
  --wandb_name "GRM_LoRA_example" \
  --output_dir "../save_reward_models/GRM_LoRA_example" \
  --save_strategy steps \
  --save_steps 1000 \
  --eval_steps 1000 \
  --logging_steps 100 \
  --save_safetensors True \
  --seed 1
```

See `scripts/examples/` for SLURM batch script templates.

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--dataset_step_size N` | Subsample the training set by taking every Nth example (e.g., 2 for 50%, 20 for 5%) |
| `--use_log_overlay` | Overlay log-scale (powers of 2) save/eval/log steps on top of the fixed `--save_steps` cadence |
| `--attn_implementation` | Attention implementation: `sdpa` (default), `eager`, `flash_attention_2` |
| `--push_to_hub` | Push checkpoints to HF Hub during training |
| `--hub_model_id` | HF Hub repo ID for pushing (e.g., `your-org/model-name`) |
| `--max_steps` | Total optimizer steps (overrides `--num_train_epochs` when set) |

## Uploading Checkpoints to HF Hub

To upload a full set of checkpoints as tagged revisions after training:

```bash
python scripts/checkpoint_automated_upload.py \
  --model MODEL_NAME \
  --repo-prefix your-hf-org
```

This creates one immutable tag per checkpoint (`step-0`, `step-1000`, ...) plus convenience tags (`best`, `final`).

## Citation

```bibtex
@inproceedings{christian2026reward,
  title={Reward Models Inherit Value Biases from Pretraining},
  author={Christian, Brian and Thompson, Jessica A. F. and Yang, Elle Michelle and Adam, Vincent and Kirk, Hannah Rose and Summerfield, Christopher and Dumbalska, Tsvetomira},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```

## Acknowledgments

This code is built on the [Generalizable Reward Model](https://github.com/YangRui2015/Generalizable-Reward-Model) codebase:

```bibtex
@inproceedings{yang2024regularizing,
  title={Regularizing Hidden States Enables Learning Generalizable Reward Model for LLMs},
  author={Yang, Rui and Ding, Ruomeng and Lin, Yong and Zhang, Huan and Zhang, Tong},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```

It also builds on [transformers](https://github.com/huggingface/transformers), [trl](https://github.com/huggingface/trl), and [RLHFlow](https://github.com/RLHFlow/RLHF-Reward-Modeling).
