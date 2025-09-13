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
        from transformers import AutoTokenizer, AutoConfig
        from grm_utils import AutoModelForCausalLMWithValueHead
        from grm_utils import load_model_withhead
        
        print(f"Loading model from {checkpoint_path}...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        config = AutoConfig.from_pretrained("Ray2333/GRM-Gemma2-2B-sftreg")
        config.attn_implementation = "eager"
        model = load_model_withhead(
            model_name="Ray2333/GRM-Gemma2-2B-sftreg",
            peft_name=checkpoint_path,
            tokenizer=tokenizer,
            device="cpu"
        )

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    except Exception as e:
        print(f"❌ Model load error: {e}")
        return False

    try:
        print("✓ Model and tokenizer loaded successfully")
        print(f"Model type: {type(model).__name__}")
        
        # Ensure num_labels is set correctly in config for saving/loading
        if hasattr(model, 'config'):
            model.config.num_labels = 1

        print(f"DEBUG: About to upload to hub_model_id: {hub_model_id}")
        print(f"\nUploading to {hub_model_id}...")
        
        # Upload model
        model.push_to_hub(
            repo_id=hub_model_id,
            private=private,
            commit_message="Upload GRM reward model checkpoint"
        )
        print("✓ Model uploaded")
        
        # Upload tokenizer
        tokenizer.push_to_hub(
            repo_id=hub_model_id,
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
    # print("Testing imports...")
    # if not test_imports():
    #     print("\n❌ Please make sure you're in the right conda environment:")
    #     print("conda activate value-from-language")
    #     return 1
    
    print("\n" + "="*50)
    
    # Configuration - EDIT THESE VALUES
    MODEL_FOLDER = "GRM_Gemma2-2B_ckpts"  # Folder where model is saved
    CHECKPOINT = "12772"  # Checkpoint number to upload
    CHECKPOINT_PATH = f"save_reward_models/{MODEL_FOLDER}/checkpoint-{CHECKPOINT}"
    HUB_MODEL_ID = f"brianchristian/{MODEL_FOLDER}_checkpoint-{CHECKPOINT}"
    PRIVATE = True  # Set to False for public repo
    
    print(f"DEBUG: CHECKPOINT_PATH = {CHECKPOINT_PATH}")
    print(f"DEBUG: HUB_MODEL_ID = {HUB_MODEL_ID}")
    print(f"DEBUG: Are they different? {CHECKPOINT_PATH != HUB_MODEL_ID}")

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Hub Model ID: {HUB_MODEL_ID}")
    print(f"Private: {PRIVATE}")
    
    # # Confirm before upload
    # response = input("\nProceed with upload? (yes/no): ")
    # if response.lower() not in ['yes', 'y']:
    #     print("Upload cancelled")
    #     return 0
    
    # Do the upload
    success = upload_checkpoint(CHECKPOINT_PATH, HUB_MODEL_ID, PRIVATE)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())