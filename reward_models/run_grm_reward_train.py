from dataclasses import dataclass, field
from typing import List, Optional
from accelerate import Accelerator
import evaluate
import numpy as np
import os
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    TrainerCallback,
    AutoModelForCausalLM
)
from grm_reward_trainer import GRMDataCollatorWithPadding, GRMRewardTrainer
from load_datasets import load_train_eval_dataset
from utils import print_trainable_parameters, grm_compute_metrics
from grm_utils import AutoModelForCausalLMWithValueHead

from huggingface_hub import HfApi, create_commit, CommitOperationAdd, CommitOperationDelete
try:
    from huggingface_hub import HfHubHTTPError
except ImportError:
    from requests import HTTPError as HfHubHTTPError
import glob, shutil, tempfile

class PromoteAndTagCallback(TrainerCallback):
    """
    On each Trainer save:
      - copy the newest checkpoint's *loadable* files into a temp staging dir
      - replace the *repo root* with those files (atomic commit)
      - create a tag 'checkpoint-<step>' pointing at that commit
    """
    def __init__(self, repo_id: str, keep_root={"README.md","LICENSE",".gitattributes"}, hf_token=None):
        super().__init__()
        self.repo_id = repo_id
        self.keep_root = set(keep_root)
        self.api = HfApi(token=hf_token)
        # make sure repo exists
        try:
            self.api.repo_info(repo_id, repo_type="model")
        except HfHubHTTPError:
            self.api.create_repo(repo_id, repo_type="model", exist_ok=True, private=True)

        # Files to promote (cover full FT and LoRA)
        self.patterns = [
            "config.json","generation_config.json",
            "model.safetensors","pytorch_model.bin",
            "adapter_config.json","adapter_model.safetensors","peft_config.json",
            "tokenizer.json","tokenizer_config.json","tokenizer.model",
            "added_tokens.json",
            "special_tokens_map.json","spiece.model","vocab.json","merges.txt",
            "v_head.pt",
        ]

    def on_save(self, args, state, control, **kwargs):
        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt): return

        with tempfile.TemporaryDirectory() as staging:
            picked = []
            for pat in self.patterns:
                for src in glob.glob(os.path.join(ckpt, pat)):
                    dst = os.path.join(staging, os.path.basename(src))
                    shutil.copy2(src, dst)
                    picked.append(os.path.basename(src))
            if not picked:
                print(f"[PromoteAndTag] no loadable files found in {ckpt}")
                return

            # delete old root (except keepers), add new files
            ops = []
            try:
                info = self.api.repo_info(self.repo_id, repo_type="model")
                root_files = [f.rfilename for f in info.siblings if "/" not in f.rfilename]
                for rf in root_files:
                    if rf not in self.keep_root:
                        ops.append(CommitOperationDelete(path_in_repo=rf))
            except Exception as e:
                print(f"[PromoteAndTag] warn listing root: {e}")

            for fname in os.listdir(staging):
                ops.append(CommitOperationAdd(
                    path_in_repo=fname,
                    path_or_fileobj=os.path.join(staging, fname)
                ))

            commit = create_commit(
                repo_id=self.repo_id,
                repo_type="model",
                operations=ops,
                commit_message=f"Promote checkpoint-{state.global_step} to repo root",
                token=self.api.token,
            )
            tag = f"checkpoint-{state.global_step}"
            try:
                self.api.create_tag(self.repo_id, tag=tag, revision=commit.oid, repo_type="model")
                print(f"[PromoteAndTag] tagged {tag}")
            except Exception as e:
                print(f"[PromoteAndTag] warn create_tag: {e}")

@dataclass
class ScriptArguments:
    # training args
    per_device_train_batch_size: Optional[int] = field(default=1) 
    gradient_accumulation_steps: Optional[int] = field(default=16)
    learning_rate: Optional[float] = field(default=1e-5)
    num_train_epochs: Optional[int] = field(default=2, metadata={"help": "The number of training epochs for the reward model."})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "Total number of optimizer steps to train for. Overrides num_train_epochs."})
    optim: Optional[str] = field(default="adamw_hf",  metadata={"help": "The optimizer to use."})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "The lr scheduler"},)
    max_length: Optional[int] = field(default=1024) 
    gradient_checkpointing: Optional[bool] = field(default=True)
    bf16: Optional[bool] = field(default=True)
    attn_implementation: Optional[str] = field(default="sdpa")
    # data
    dataset: Optional[str] = field(default='llm-blender/Unified-Feedback')
    dataset_mode: Optional[str] = field(default='', metadata={"help": "use from '', '40k', and '400k' for the paper's experiments"},)
    dataset_step_size: Optional[int] = field(default=None, metadata={"help": "Step size for dataset subsampling (e.g., 2 for every 2nd sample, 20 for every 20th sample)"},)
    # lora
    use_lora: Optional[bool] = field(default=True)
    lora_target_modules: Optional[List[str]] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_r: Optional[int] = field(default=32)
    lora_alpha: Optional[int] = field(default=64)
    lora_dropout: Optional[float] = field(default=0.05)
    # eval
    per_device_eval_batch_size: Optional[int] = field(default=1)
    evaluation_strategy: Optional[str] = field(default="steps")
    eval_steps: Optional[int] = field(default=100)
    # model and loss
    base_model: Optional[str] =  field(default="google/gemma-2b-it")
    # log
    report_to: Optional[str] = field(default='none', metadata={'help': "use 'none', 'wandb'. "})
    log_dir: Optional[str] = field(default='./reward_models_train')
    wandb_name: Optional[str] = field(default="test",)
    save_strategy: Optional[str] = field(default="epoch")
    save_steps: Optional[int] = field(default=1000)
    debug: Optional[bool] = field(default=False, metadata={'help': 'if debug=True, only train with 100 samples'})
    # GRM
    weight_ratio: Optional[float] = field(default=0.01)
    beta: Optional[float] = field(default=0.1, metadata={'help': 'beta for DPO'})
    layer_type: Optional[str] = field(default='mlp') # mlp, linear
    num_layers: Optional[int] = field(default=1)
    num_neurons: Optional[int] = field(default=1024)
    reference_free: Optional[bool] = field(default=True)
    sft_only: Optional[bool] = field(default=True)
    no_logsigmoid_sft: Optional[bool] = field(default=False)
    # Checkpointing
    output_dir: Optional[str] = field(default=None, metadata={"help": "Overrides default output path"})
    save_total_limit: Optional[int] = field(default=12)
    logging_steps: Optional[int] = field(default=100)
    load_best_model_at_end: Optional[bool] = field(default=True)
    metric_for_best_model: Optional[str] = field(default="eval_loss")
    greater_is_better: Optional[bool] = field(default=False)
    save_safetensors: Optional[bool] = field(default=True)
    # Hub arguments
    push_to_hub: Optional[bool] = field(default=False)
    hub_model_id: Optional[str] = field(default=None)
    hub_private_repo: Optional[bool] = field(default=False)
    hub_strategy: Optional[str] = field(default="every_save")
    # Training seed
    seed: Optional[int] = field(default=42)
    


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
model_name_split = script_args.base_model.split("/")[-1]
if script_args.use_lora:
    output_name = f"{script_args.log_dir}/{model_name_split}_{script_args.wandb_name}_len{script_args.max_length}_lora{script_args.lora_r}_{script_args.learning_rate}_data{script_args.dataset.split('/')[-1]}"
else:
    output_name = f"{script_args.log_dir}/{model_name_split}_{script_args.wandb_name}_len{script_args.max_length}_fulltrain_{script_args.learning_rate}_data{script_args.dataset.split('/')[-1]}"

device = Accelerator().local_process_index 

# pick a tidy default if user didn't pass --output_dir
final_output_dir = script_args.output_dir or os.path.join(output_name, 'checkpoints')

training_args = TrainingArguments(
    output_dir=final_output_dir,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    evaluation_strategy=script_args.evaluation_strategy,
    eval_steps=script_args.eval_steps,
    save_strategy=script_args.save_strategy,
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=script_args.logging_steps,
    warmup_ratio=0.03,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    run_name=script_args.wandb_name,
    max_grad_norm=5.0,
    report_to=script_args.report_to,
    remove_unused_columns=False,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,
    load_best_model_at_end=script_args.load_best_model_at_end,
    metric_for_best_model=script_args.metric_for_best_model,
    greater_is_better=script_args.greater_is_better,
    save_safetensors=script_args.save_safetensors,
    push_to_hub=script_args.push_to_hub,
    hub_model_id=script_args.hub_model_id,
    hub_private_repo=script_args.hub_private_repo,
    hub_strategy=script_args.hub_strategy,
    seed=script_args.seed,
)

# Callback to save the value head
# (By default only LoRA adapters are saved)
class SaveVHeadCallback(TrainerCallback):
    """
    Save the reward value head alongside each checkpoint.
    Creates:
      - <output_dir>/checkpoint-<step>/v_head.pt  on every save
      - <output_dir>/v_head.pt                    at train end
      - <best_model_checkpoint>/v_head.pt         if available
    """
    @staticmethod
    def _unwrap_to_vhead(model):
        # PeftModel -> base_model (LoraModel) -> model (AutoModelForCausalLMWithValueHead) -> v_head
        inner = getattr(model, "base_model", model)
        inner = getattr(inner, "model", inner)
        return getattr(inner, "v_head", None)

    def on_save(self, args, state, control, **kwargs):
        v_head = self._unwrap_to_vhead(kwargs["model"])
        if v_head is None:
            return
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(v_head.state_dict(), os.path.join(ckpt_dir, "v_head.pt"))
        print(f"[SaveVHeadCallback] wrote {ckpt_dir}/v_head.pt")

    def on_train_end(self, args, state, control, **kwargs):
        v_head = self._unwrap_to_vhead(kwargs["model"])
        if v_head is None:
            return
        # save final head at top level
        torch.save(v_head.state_dict(), os.path.join(args.output_dir, "v_head.pt"))
        print(f"[SaveVHeadCallback] wrote {args.output_dir}/v_head.pt")
        # also save into best checkpoint if Trainer selected one
        best = getattr(state, "best_model_checkpoint", None)
        if best:
            try:
                torch.save(v_head.state_dict(), os.path.join(best, "v_head.pt"))
                print(f"[SaveVHeadCallback] wrote {best}/v_head.pt (best model)")
            except Exception as e:
                print(f"[SaveVHeadCallback] warn: {e}")

# Load the tokenizer.
tokenizer = AutoTokenizer.from_pretrained(script_args.base_model, use_fast = False)
tokenizer.max_length = script_args.max_length
if tokenizer.pad_token == None:
    if 'Llama' in script_args.base_model:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        tokenizer.pad_token = tokenizer.eos_token

# Load datasets
train_dataset, eval_dataset = load_train_eval_dataset(script_args.dataset, tokenizer, mode=script_args.dataset_mode, model_name='GRM', dataset_step_size=script_args.dataset_step_size, size=100 if script_args.debug else None)
print('Training dataset size: {}, validation dataset size: {}'.format(len(train_dataset), len(eval_dataset)))


model_params = {
    'vhead_layer_type': script_args.layer_type,
    'vhead_num_neurons': 1024,
    'vhead_num_layers': script_args.num_layers,
}
if len(script_args.attn_implementation):
    model_params["attn_implementation"] = script_args.attn_implementation


### load model
if not script_args.reference_free:
    reference_model = AutoModelForCausalLM.from_pretrained(script_args.base_model, device_map=device, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    reference_model.resize_token_embeddings(len(tokenizer))
    reference_model.config.pad_token_id = tokenizer.pad_token_id


model = AutoModelForCausalLMWithValueHead.from_pretrained(
    script_args.base_model, device_map=device, 
    torch_dtype=torch.bfloat16,
    **model_params,
)

model.pretrained_model.resize_token_embeddings(len(tokenizer))
print_trainable_parameters(model)
model.config.pad_token_id = tokenizer.pad_token_id

if script_args.use_lora:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules=script_args.lora_target_modules,
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
    )
    model = get_peft_model(model, peft_config)

## let value head trainable
if hasattr(model, 'v_head'):
    for parameter in model.v_head.parameters():
        parameter.requires_grad = True
print_trainable_parameters(model)


# Define the trainer parameters
trainer_params = {
    "model": model,
    "reference_model": reference_model if not script_args.reference_free else None,
    "args": training_args,
    "tokenizer": tokenizer,
    "train_dataset": train_dataset,
    "eval_dataset": eval_dataset,
    "compute_metrics": grm_compute_metrics,
    "data_collator": GRMDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length),
    'weight_ratio': script_args.weight_ratio,
    'reference_free': script_args.reference_free,
    'sft_only': script_args.sft_only,
    'no_logsigmoid_sft': script_args.no_logsigmoid_sft,
    'beta': script_args.beta,
    'use_lora': script_args.use_lora,
    'info_to_save' : {
        'base_model': script_args.base_model,
        'layer_type': script_args.layer_type,
        'num_neurons': script_args.num_neurons,
        'num_layers': script_args.num_layers,
    }
}


# Train the model, woohoo.
trainer = GRMRewardTrainer(**trainer_params)
trainer.add_callback(SaveVHeadCallback())
if script_args.push_to_hub and script_args.hub_model_id:
    trainer.add_callback(
        PromoteAndTagCallback(
            repo_id=script_args.hub_model_id,
            hf_token=os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
        )
    )

# Before we begin, let's take a checkpoint at "step 0"
if training_args.save_strategy == "steps" and training_args.save_steps > 0:
    print("Saving initial checkpoint at step 0")
    trainer.save_model(f"{training_args.output_dir}/checkpoint-0")

print('training start')
trainer.train()
