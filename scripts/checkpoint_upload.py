#!/usr/bin/env python3
"""
Script to upload a trained reward model checkpoint to HuggingFace Hub
"""
import argparse
import os
import sys
sys.path.append('./reward_models')

from transformers import AutoTokenizer
from grm_utils import AutoModelForCausalLMWithValueHead

def main():
    parser = argparse.ArgumentParser(description="Upload GRM checkpoint to HuggingFace Hub")
    parser.add_argument("--checkpoint_path", required=True, help="Path to checkpoint directory")
    parser.add_argument("--hub_model_id", required=True, help="Hub model ID (e.g., username/model-name)")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    parser.add_argument("--dry_run", action="store_true", help="Test loading without uploading")
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint path {args.checkpoint_path} does not exist")
        return 1
    
    print(f"Loading model from {args.checkpoint_path}")
    
    try:
        # Load the model and tokenizer
        model = AutoModelForCausalLMWithValueHead.from_pretrained(args.checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)

        # Ensure num_labels is set correctly in config for saving/loading
        if hasattr(model.pretrained_model, 'config'):
            model.pretrained_model.config.num_labels = 1

        print("✓ Model and tokenizer loaded successfully")
        
        # Print model info
        print(f"Model type: {type(model).__name__}")
        print(f"Base model: {model.pretrained_model.config._name_or_path if hasattr(model.pretrained_model.config, '_name_or_path') else 'Unknown'}")
        if hasattr(model, 'v_head'):
            print("✓ Value head detected")
        
        if args.dry_run:
            print("Dry run completed - model loads correctly!")
            return 0
            
        print(f"\nUploading to {args.hub_model_id}")
        
        # Upload model
        model.push_to_hub(
            args.hub_model_id, 
            private=args.private,
            commit_message="Upload GRM reward model checkpoint"
        )
        
        # Upload tokenizer  
        tokenizer.push_to_hub(
            args.hub_model_id,
            private=args.private,
            commit_message="Upload tokenizer"
        )
        
        print("✓ Upload completed successfully!")
        print(f"Model available at: https://huggingface.co/{args.hub_model_id}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())