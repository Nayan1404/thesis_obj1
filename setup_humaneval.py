# +
# #!/usr/bin/env python3
"""
Script to download and prepare HumanEval dataset
"""
import json
import os

def download_humaneval():
    """Download HumanEval dataset from Hugging Face"""
    print("Downloading HumanEval dataset...")
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not found")
        print("Please install it with: pip install datasets")
        return False
    
    try:
        # Load the dataset
        dataset = load_dataset("openai_humaneval", split="test")
        
        # Save to JSONL
        output_file = "humaneval.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")
        
        print(f"Successfully downloaded {len(dataset)} problems")
        print(f"Saved to: {output_file}")
        
        # Show sample
        print("\n" + "="*70)
        print("Sample problem:")
        print("="*70)
        sample = dataset[0]
        print(f"Task ID: {sample['task_id']}")
        print(f"Entry point: {sample['entry_point']}")
        print(f"\nPrompt:\n{sample['prompt'][:200]}...")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def verify_humaneval(filepath="humaneval.jsonl"):
    """Verify the HumanEval file is properly formatted"""
    print(f"\nVerifying {filepath}...")
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        problems = []
        for line in lines:
            problems.append(json.loads(line.strip()))
        
        print(f"Found {len(problems)} problems")
        
        # Check required fields
        required_fields = ["task_id", "prompt", "entry_point", "test", "canonical_solution"]
        sample = problems[0]
        
        print("\nChecking required fields:")
        for field in required_fields:
            has_field = field in sample
            status = "OK" if has_field else "MISSING"
            print(f"  [{status}] {field}")
        
        return True
        
    except Exception as e:
        print(f"Error verifying file: {e}")
        return False

if __name__ == "__main__":
    print("="*70)
    print("HumanEval Setup Script")
    print("="*70)
    
    # Download
    if download_humaneval():
        # Verify
        verify_humaneval()
        
        print("\n" + "="*70)
        print("Setup complete!")
        print("="*70)
        print("\nNext steps:")
        print("1. Run generation:")
        print("   python generation.py --model deepseekcoder --data_path humaneval.jsonl --save_path results/humaneval_deepseekcoder.jsonl")
        print("\n2. Run evaluation:")
        print("   python eval.py --generation_file results/humaneval_deepseekcoder.jsonl")
        print("="*70)
    else:
        print("\nSetup failed. Please check the errors above.")
