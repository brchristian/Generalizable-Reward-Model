#!/usr/bin/env python3
"""
Simple test to check checkpoint contents without complex imports
"""
import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Check checkpoint contents")
    parser.add_argument("--checkpoint_path", required=True, help="Path to checkpoint directory")
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint path {args.checkpoint_path} does not exist")
        return 1
    
    print(f"Examining checkpoint: {args.checkpoint_path}")
    
    # List files
    files = os.listdir(args.checkpoint_path)
    print(f"Files found: {files}")
    
    # Check config
    config_path = os.path.join(args.checkpoint_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Model type: {config.get('model_type', 'unknown')}")
        print(f"Base model: {config.get('_name_or_path', 'unknown')}")
        print(f"Has value head config: {any(k.startswith('vhead_') for k in config.keys())}")
    
    # Check for required files for upload
    required_files = ['model.safetensors', 'config.json', 'tokenizer.model', 'tokenizer_config.json']
    missing_files = [f for f in required_files if f not in files]
    
    if missing_files:
        print(f"Warning: Missing files for upload: {missing_files}")
    else:
        print("✓ All required files present for upload")
    
    return 0

if __name__ == "__main__":
    exit(main())