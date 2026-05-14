import json

def read_json(path):
    """Reads a JSONL file from the given path."""
    problems = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                problems.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed line in {path}")
    return problems

def generate_prompt(problem, model_type=None):
    """
    Generate a prompt for HumanEval code generation.
    HumanEval problems require completing a function.
    """
    # HumanEval format
    prompt_text = problem.get("prompt", "").strip()
    entry_point = problem.get("entry_point", "").strip()
    
    # Create instruction-based prompt
    prompt = (
        "You are a Python code generator. Complete the following function.\n\n"
        "CRITICAL RULES:\n"
        "1. Start your response IMMEDIATELY with <START_CODE>\n"
        "2. Write ONLY the complete Python function between the markers\n"
        "3. End with <END_CODE>\n"
        "4. NO explanations, NO comments outside the code block, NO markdown\n"
        "5. The function must be syntactically correct and complete\n\n"
        f"Complete this function:\n\n{prompt_text}\n\n"
        "Response format (STRICT):\n"
        "<START_CODE>\n"
        "# Your complete function implementation here\n"
        "<END_CODE>\n\n"
        "Generate your solution now (start with <START_CODE>):"
    )
    
    return prompt
