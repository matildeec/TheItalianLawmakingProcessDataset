import os
from datetime import datetime

BASEDIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(BASEDIR)

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================
# Ollama inference hyperparameters.
# Note: The context window (num_ctx) is set to 65536 to accommodate the largest inputs,
# prompt templates (~9.8k tokens), and projected long JSON outputs (~42k-48k tokens) without truncation.
INFERENCE_PARAMS = {
    "temperature": 0.0,
    "top_k": 10,
    "top_p": 0.9,
    "num_predict": 49152,       # Accommodates the longest output in the corpus (projected ~42k tokens)
    "num_ctx": 65536,        # Total context window (Prompt ~9.8k + Input ~2.8k + Output ~48k tokens)
    "seed": 42,
    "keep_alive": "1h",
    "think": False
}

PROMPT_SETTING = 'few-shot'

# ==============================================================================
# MODELS
# ==============================================================================
# Reference Ollama model: https://ollama.com/library/gemma4
MODELS_CONFIG = {
    'gemma4:31b':       {'active': True,  'alias': 'gemma4:31b'}
}

# Dynamically build the list of active models
ACTIVE_MODELS = [model for model, params in MODELS_CONFIG.items() if params['active']]

# ==============================================================================
# PROMPT FILE
# ==============================================================================
PROMPT_DIR = "prompts"
PROMPT_FILE = "event_extraction_instruct.txt"
PROMPT_PATH = os.path.join(BASEDIR, PROMPT_DIR, PROMPT_FILE)

# ==============================================================================
# INPUT/OUTPUT PATHS
# ==============================================================================

# Input CSV
CSV_CONFIG = {
    "path": os.path.join(PIPELINE_ROOT, "data", "dataset_final_extraction.csv"),  # Contains all corpus bills except the validation sample ("val_sample_100.csv")
    "text": "lavori_preparatori", 
    "id": "case_id",
    "delimiter": ",",
}

# Output
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
OUTPUT_DIR = os.path.join(PIPELINE_ROOT, "output")
OUTPUT_FILENAME = f"extraction_{PROMPT_SETTING}_{timestamp}.jsonl"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

os.makedirs(OUTPUT_DIR, exist_ok=True)
