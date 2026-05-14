import argparse
import json
import os
from tqdm import tqdm
from models import DeepSeekCoder, CodeLLaMA_7b, WizardCoder13B, CodeLLaMA_13b
from utils import read_json, generate_prompt

# Batch size can be tuned based on GPU memory
BATCH_SIZE = 1

MODEL_MAP = {
    "deepseekcoder": DeepSeekCoder,
    "codellama_7b": CodeLLaMA_7b,
    "wizardcoder_13b": WizardCoder13B,
    "codellama_13b": CodeLLaMA_13b,
}

def get_model(name, path=None):
    """Return the model class instance with optional path."""
    return MODEL_MAP[name](model_path=path)

def main(args):
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    
    # Initialize model with the optional local path
    model = get_model(args.model, args.model_path)

    # Load problems
    with open(args.data_path, encoding="utf-8") as f:
        problems = [json.loads(line) for line in f]
    
    if len(problems) > 5000:
        problems = problems[:5000]
        print(f"⚠️  Limiting to first 5000 problems")
    
    print(f"✅ Loaded {len(problems)} problems from {args.data_path}")

    # Set pad_token_id if missing
    try:
        if hasattr(model, "pipe") and hasattr(model.pipe, "tokenizer"):
            tokenizer = model.pipe.tokenizer
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = model.model.config.eos_token_id
    except Exception as e:
        print(f"[WARN] Could not set pad_token_id: {e}")

    successful = 0
    failed = 0

    with open(args.save_path, "w", encoding="utf-8") as fout:
        for i, p in enumerate(tqdm(problems, desc=f"Generating with {args.model}")):
            task_id = p.get("task_id", p.get("problem_id", i))

            # --- FIX 1: Handle MBPP "text" field ---
            if p.get("prompt"):
                raw_prompt_text = p["prompt"]
            elif p.get("text"):
                # MBPP uses 'text' as the problem description
                raw_prompt_text = p["text"]
            else:
                raw_prompt_text = ""

            # Inject the extracted text back into p so generate_prompt can find it
            p["prompt"] = raw_prompt_text
            
            # Generate the formatted prompt (Instructions + Problem)
            prompt = generate_prompt(p)
            
            # --- FIX 2: Handle MBPP "test_list" field ---
            # Eval.py needs a single string 'test', but MBPP has a list
            if p.get("test"):
                test_code = p["test"]
            elif p.get("test_list"):
                # Combine setup code + assertions
                setup = p.get("test_setup_code", "")
                assertions = "\n".join(p["test_list"])
                test_code = f"{setup}\n{assertions}"
            else:
                test_code = ""

            try:
                # Generate
                out = model.pipe(
                    [prompt],
                    do_sample=False,
                    temperature=args.temperature,
                    max_new_tokens=512,
                    batch_size=1,
                    return_full_text=False
                )
                
                # Extract text safely
                if isinstance(out, list) and len(out) > 0:
                    o = out[0]
                else:
                    o = out
                
                if isinstance(o, dict):
                    raw = o.get("generated_text", o.get("text", str(o)))
                elif isinstance(o, list) and o:
                    raw = (o[0].get("generated_text", str(o[0])) if isinstance(o[0], dict) else str(o[0]))
                else:
                    raw = str(o)

                # Remove prompt echo
                if raw.startswith(prompt):
                    raw = raw[len(prompt):].strip()

                # Extract code block
                try:
                    code = model.extract_code(raw)
                except Exception as ex:
                    code = raw
                
                successful += 1

            except Exception as e:
                print(f"[ERROR] Problem {i} failed: {e}")
                raw = f"# ERROR: {str(e)}"
                code = f"# ERROR: {str(e)}"
                failed += 1

            # Prepare result for eval.py
            result = {
                "task_id": task_id,
                "prompt": prompt,
                "deal_response": code,
                "full_response": raw,
                "input_output": json.dumps(p.get("input_output", "{}")),
                "test": test_code,  # IMPORTANT: Saving the fixed test string
                "entry_point": p.get("entry_point")
            }
            
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"\n✅ Generation complete! Saved to: {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model generation on dataset")
    parser.add_argument("--model", required=True, choices=MODEL_MAP.keys(), help="Model name")
    parser.add_argument("--model_path", default=None, help="Path to local pruned model weights")
    parser.add_argument("--data_path", required=True, help="Path to dataset JSONL")
    parser.add_argument("--save_path", required=True, help="Path to save results JSONL")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    args = parser.parse_args()
    main(args)
