# +
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---------------------------
# Helper function with aggressive cleaning
# ---------------------------
def extract_between_markers(text, start="<START_CODE>", end="<END_CODE>"):
    """
    Extract ONLY valid Python code, removing ALL non-code text.
    """
    if not text:
        return ""

    # Handle list outputs (pipeline responses)
    if isinstance(text, list):
        if len(text) > 0:
            if isinstance(text[0], dict) and 'generated_text' in text[0]:
                text = text[0]['generated_text']
            else:
                text = str(text[0])
        else:
            return ""

    # Handle dict responses
    if isinstance(text, dict):
        text = text.get('generated_text', text.get('text', str(text)))

    text = str(text)
    
    # Fix escaped newlines
    if '\\n' in text and '\n' not in text:
        text = text.replace('\\n', '\n')

    # Priority 1: Extract from markers
    pattern = re.escape(start) + r"([\s\S]*?)" + re.escape(end)
    match = re.search(pattern, text)
    if match:
        code = match.group(1).strip()
        return clean_extracted_code(code)

    # Priority 2: Extract from triple backticks
    blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", text)
    if blocks:
        code = blocks[0].strip()
        return clean_extracted_code(code)

    # Priority 3: Extract function/class definitions
    code_match = re.search(r"((?:def|class)\s+\w+[\s\S]*)", text)
    if code_match:
        return clean_extracted_code(code_match.group(1))

    # Priority 4: Remove common non-code prefixes
    lines = text.split('\n')
    code_started = False
    code_lines = []
    
    for line in lines:
        stripped = line.strip()
        if not code_started:
            if (stripped.startswith('def ') or
                stripped.startswith('class ') or
                stripped.startswith('import ') or
                stripped.startswith('from ') or
                (stripped and not any(x in stripped.lower() for x in
                    ['here', 'sure', 'solution', 'explanation', 'note:', '**', 'output:', 'example:']))):
                code_started = True
                code_lines.append(line)
        else:
            if stripped.startswith('This ') or stripped.startswith('The ') or stripped.startswith('Note:'):
                break
            code_lines.append(line)
    
    if code_lines:
        return clean_extracted_code('\n'.join(code_lines))
    
    return clean_extracted_code(text)

def clean_extracted_code(code):
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not cleaned_lines and not stripped: continue
        if stripped.startswith('```'): continue
        if stripped.lower().startswith(('note:', 'example:', 'output:', 'explanation:')): continue
        if stripped.startswith('**') and stripped.endswith('**'): continue
        cleaned_lines.append(line)
    
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()
    
    result = '\n'.join(cleaned_lines).strip()
    return result if result else code.strip()


# ---------------------------
# Robust Model Loader
# ---------------------------
# In models.py

def load_model_robust(model_path, default_id):
    path_to_use = model_path if model_path else default_id
    
    # Try loading tokenizer
    try:
        print(f"Loading tokenizer from: {path_to_use}")
        tokenizer = AutoTokenizer.from_pretrained(path_to_use, trust_remote_code=True)
    except Exception:
        print(f"Falling back to Hub tokenizer: {default_id}")
        tokenizer = AutoTokenizer.from_pretrained(default_id, trust_remote_code=True)

    print(f"Loading model weights from: {path_to_use}")
    
    # --- UPDATE HERE: Added load_in_8bit=True ---
    model = AutoModelForCausalLM.from_pretrained(
        path_to_use, 
        device_map="auto", 
        trust_remote_code=True,
        load_in_8bit=True  # This reduces VRAM usage from ~26GB to ~14GB
    )
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    if pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token_id = model.config.eos_token_id
        
    return tokenizer, model, pipe


# ---------------------------
# Model Classes
# ---------------------------

class DeepSeekCoder:
    def __init__(self, model_path=None):
        self.default_id = "deepseek-ai/deepseek-coder-6.7b-instruct"
        self.tokenizer, self.model, self.pipe = load_model_robust(model_path, self.default_id)

    def generate(self, prompt, **kwargs):
        return self.pipe(prompt, **kwargs)[0]["generated_text"]

    def extract_code(self, text):
        return extract_between_markers(text)

class CodeLLaMA_7b:
    def __init__(self, model_path=None):
        self.default_id = "codellama/CodeLlama-7b-Instruct-hf"
        self.tokenizer, self.model, self.pipe = load_model_robust(model_path, self.default_id)

    def generate(self, prompt, **kwargs):
        return self.pipe(prompt, **kwargs)[0]["generated_text"]

    def extract_code(self, text):
        return extract_between_markers(text)

class WizardCoder13B:
    def __init__(self, model_path=None):
        self.default_id = "WizardLMTeam/WizardCoder-Python-13B-V1.0"
        self.tokenizer, self.model, self.pipe = load_model_robust(model_path, self.default_id)

    def generate(self, prompt, **kwargs):
        return self.pipe(prompt, **kwargs)[0]["generated_text"]

    def extract_code(self, text):
        return extract_between_markers(text)

class CodeLLaMA_13b:
    def __init__(self, model_path=None):
        self.default_id = "codellama/CodeLlama-13b-Instruct-hf"
        self.tokenizer, self.model, self.pipe = load_model_robust(model_path, self.default_id)

    def generate(self, prompt, **kwargs):
        return self.pipe(prompt, **kwargs)[0]["generated_text"]

    def extract_code(self, text):
        return extract_between_markers(text)
