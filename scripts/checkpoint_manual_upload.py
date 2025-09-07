#!/usr/bin/env python3
"""
Manual upload script - run this in your terminal with:
conda activate value-from-language && python manual_upload.py

This avoids conda activation issues in the subprocess
"""
import sys
import os

# Add reward_models to path for imports
sys.path.append('./reward_models')

def test_imports():
    """Test that we can import everything we need"""
    try:
        from transformers import AutoTokenizer
        print("✓ transformers imported")
        
        from grm_utils import AutoModelForCausalLMWithValueHead  
        print("✓ grm_utils imported")
        
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✓ HF Hub authenticated as: {user_info['name']}")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def upload_checkpoint(checkpoint_path, hub_model_id, private=True):
    """Upload a checkpoint to HF Hub"""
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint path {checkpoint_path} does not exist")
        return False
    
    try:
        from transformers import AutoTokenizer
        from grm_utils import AutoModelForCausalLMWithValueHead
        
        print(f"Loading model from {checkpoint_path}...")
        model = AutoModelForCausalLMWithValueHead.from_pretrained(checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        
        print("✓ Model and tokenizer loaded successfully")
        print(f"Model type: {type(model).__name__}")
        
        print(f"\nUploading to {hub_model_id}...")
        
        # Upload model
        model.push_to_hub(
            hub_model_id,
            private=private,
            commit_message="Upload GRM reward model checkpoint"
        )
        print("✓ Model uploaded")
        
        # Upload tokenizer
        tokenizer.push_to_hub(
            hub_model_id,
            private=private, 
            commit_message="Upload tokenizer"
        )
        print("✓ Tokenizer uploaded")
        
        print(f"\n🎉 Upload completed!")
        print(f"Model available at: https://huggingface.co/{hub_model_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False

def main():
    print("=== GRM Checkpoint Upload Tool ===\n")
    
    # Test imports first
    print("Testing imports...")
    if not test_imports():
        print("\n❌ Please make sure you're in the right conda environment:")
        print("conda activate value-from-language")
        return 1
    
    print("\n" + "="*50)
    
    # Configuration - EDIT THESE VALUES
    CHECKPOINT_PATH = "save_reward_models/GRM_Gemma2-2B_ckpts/checkpoint-2000"
    HUB_MODEL_ID = "brianchristian/test-grm-checkpoint"  # Change this!
    PRIVATE = True  # Set to False for public repo
    
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Hub Model ID: {HUB_MODEL_ID}")
    print(f"Private: {PRIVATE}")
    
    # Confirm before upload
    response = input("\nProceed with upload? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Upload cancelled")
        return 0
    
    # Do the upload
    success = upload_checkpoint(CHECKPOINT_PATH, HUB_MODEL_ID, PRIVATE)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())