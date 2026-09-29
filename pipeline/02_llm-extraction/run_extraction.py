# ================================================
# --- Usage Instructions
#
# $ python run_extraction.py --start 0 --end 5 --mode mock   # TEST
# $ python run_extraction.py --start 0 --end 100 --mode real
# ================================================

import pandas as pd
import json
import os
import argparse
import time
import subprocess
from tqdm import tqdm
import ollama

# --- IMPORT CONFIG ---
from config import (
    BASEDIR,
    INFERENCE_PARAMS,
    PROMPT_SETTING,
    ACTIVE_MODELS,
    CSV_CONFIG,
    PROMPT_PATH,
    PROMPT_FILE,
    OUTPUT_PATH,
)

# ==============================================================================
# FUNCTIONS
# ==============================================================================

def verify_nvidia_gpu():
    """
    Check if NVIDIA GPU drivers are detected.
    Issues an informational message if not present, but allows execution to proceed
    for Apple Silicon (Metal/MPS), AMD (ROCm), or remote Ollama instances.
    """
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print("NVIDIA GPU detected (Ready for inference).")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[INFO] nvidia-smi not detected. Proceeding with Ollama default accelerator (e.g., Apple Silicon Metal, ROCm, or remote server).")


def load_prompt(path: str) -> str:
    """
    Load the system prompt from a .txt file.
    Raises FileNotFoundError if the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def llm_generate(model: str, user_prompt: str, system_prompt: str, config_params: dict) -> dict:
    """
    Execute the inference call to Ollama.
    The expected output is a JSON array, enforcing format='json'.
    """
    options = {
        "temperature":        config_params["temperature"],
        "top_k":              config_params["top_k"],
        "top_p":              config_params["top_p"],
        "num_predict":        config_params["num_predict"],
        "num_ctx":            config_params["num_ctx"],
        "seed":               config_params["seed"],
    }

    try:
        response = ollama.generate(
            model=model,
            system=system_prompt,
            prompt="Input: " + user_prompt,
            think=config_params["think"],
            stream=False,
            keep_alive=config_params["keep_alive"],
            format="json",
            options=options,
        )
        return {
            "response":            response.get("response", "").strip(),
            "total_duration":      response.get("total_duration"),
            "load_duration":       response.get("load_duration"),
            "prompt_eval_count":   response.get("prompt_eval_count"),
            "prompt_eval_duration":response.get("prompt_eval_duration"),
            "eval_count":          response.get("eval_count"),
            "eval_duration":       response.get("eval_duration"),
            "thinking":            response.get("thinking"),
            "done":  True,
            "error": None,
        }
    except Exception as e:
        return {
            "response":            None,
            "total_duration":      0,
            "load_duration":       0,
            "prompt_eval_count":   0,
            "prompt_eval_duration":0,
            "eval_count":          0,
            "eval_duration":       0,
            "thinking":            None,
            "done":  False,
            "error": str(e),
        }


def mock_generate() -> dict:
    """
    Simulate the model response (returning a mock JSON array of events).
    """
    time.sleep(1)
    mock_events = [
        {
            "activity":   "presentazione",
            "time":       "2024-01-15-09",
            "commission": None,
            "person":     "Mario Rossi",
            "chamber":    "camera",
            "chunk":      "Il disegno di legge è stato presentato da Mario Rossi il 15 gennaio 2024."
        },
        {
            "activity":   "assegnazione",
            "time":       "2024-01-20-10",
            "commission": "Commissione 1",
            "person":     None,
            "chamber":    "camera",
            "chunk":      "Il testo è stato assegnato alla Commissione 1 il 20 gennaio 2024."
        }
    ]
    return {
        "response":            json.dumps(mock_events),
        "total_duration":      100,
        "load_duration":       50,
        "prompt_eval_count":   200,
        "prompt_eval_duration":30,
        "eval_count":          80,
        "eval_duration":       70,
        "thinking":            None,
        "done":  True,
        "error": None,
    }


def parse_events(raw_response: str) -> tuple[list, str | None]:
    """
    Attempt to parse the JSON response returned by the model.
    Returns (event_list, error).
    - On success: (list, None)
    - On failure: ([], error_message)
    """
    if not raw_response:
        return [], "Empty response"
    try:
        parsed = json.loads(raw_response)
        # The model is expected to return an array; if it returns an object
        # with a root key (e.g., {"events": [...]}), normalize it.
        if isinstance(parsed, list):
            return parsed, None
        if isinstance(parsed, dict):
            # Find the first key whose value is a list
            for v in parsed.values():
                if isinstance(v, list):
                    return v, None
            return [], f"Unexpected JSON object without list value: {raw_response[:200]}"
        return [], f"Unexpected JSON type ({type(parsed).__name__}): {raw_response[:200]}"
    except json.JSONDecodeError as e:
        return [], f"JSON Decode Error: {e} | Raw: {raw_response[:200]}"


def get_fewshot_injection(fewshot_dir: str) -> str:
    """
    Read all JSON files in fewshot_dir and return the formatted section
    with Few-Shot examples to inject into the system prompt.
    """
    if not os.path.exists(fewshot_dir):
        print(f"[WARNING] Few-shot directory not found at: {fewshot_dir}")
        return ""

    import glob
    files = sorted(glob.glob(os.path.join(fewshot_dir, "*.json")))
    if not files:
        print(f"[WARNING] No few-shot JSON example files found in: {fewshot_dir}")
        return ""

    examples = []
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            text = data.get("text", "")
            eventi = data.get("eventi", [])
            ex_str = f"Input:\n\"{text}\"\n\nOutput:\n{{\n  \"eventi\": {json.dumps(eventi, indent=2, ensure_ascii=False)}\n}}"
            examples.append(ex_str)
        except Exception as e:
            print(f"[WARNING] Error reading few-shot example {fpath}: {e}")

    if not examples:
        return ""

    return "### ESEMPI DI ESTRAZIONE (FEW-SHOT):\n\n" + "\n\n---\n\n".join(examples)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="The Italian Lawmaking Process Dataset – Event Extraction Runner")
    parser.add_argument("--start", type=int, default=0,    help="Start index (CSV row)")
    parser.add_argument("--end",   type=int, default=None, help="End index (CSV row, exclusive)")
    parser.add_argument("--mode",  choices=["mock", "real"], default="mock",
                        help="Execution mode: mock (test) or real (Ollama)")
    args = parser.parse_args()

    print(f"\n=== THE ITALIAN LAWMAKING PROCESS DATASET – EVENT EXTRACTION ===")
    print(f"Mode:        {args.mode.upper()}")
    print(f"Setting:     {PROMPT_SETTING}")
    print(f"Prompt file: {PROMPT_FILE}")
    print(f"Output:      {OUTPUT_PATH}")

    if args.mode == "real":
        verify_nvidia_gpu()

    # --- Load system prompt ---
    system_prompt = load_prompt(PROMPT_PATH)
    print(f"\n[Prompt loaded from: {PROMPT_PATH}]")

    # --- Inject Few-Shot examples if PROMPT_SETTING == 'few-shot' ---
    if str(PROMPT_SETTING).lower() == "few-shot":
        fewshot_dir = os.path.join(os.path.dirname(BASEDIR), "data", "fewshot_examples")
        if not os.path.exists(fewshot_dir):
            fewshot_dir = os.path.join(BASEDIR, "data", "fewshot_examples")

        fewshot_str = get_fewshot_injection(fewshot_dir)
        if fewshot_str:
            if "### ESEMPIO DI ESTRAZIONE" in system_prompt:
                system_prompt = system_prompt.split("### ESEMPIO DI ESTRAZIONE")[0].strip() + "\n\n" + fewshot_str
            else:
                system_prompt = system_prompt + "\n\n" + fewshot_str
            print(f"[INFO] Injected Few-Shot examples from: {fewshot_dir}")

    # --- Load dataset ---
    try:
        df = pd.read_csv(CSV_CONFIG["path"], sep=CSV_CONFIG["delimiter"])
        print(f"[CSV loaded: {len(df)} total rows]")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    end_idx = args.end if args.end is not None else len(df)
    df_subset = df.iloc[args.start:end_idx].copy()

    if df_subset.empty:
        print("Empty subset. Check --start and --end parameters.")
        return

    print(f"[Selected rows: {args.start} → {end_idx} ({len(df_subset)} records)]\n")

    # --- Iterate over models ---
    for model_name in ACTIVE_MODELS:
        print(f"\n>>> ACTIVE MODEL: {model_name}")

        progress_bar = tqdm(df_subset.iterrows(), total=len(df_subset), desc=model_name)

        for idx, row in progress_bar:
            text_input = " ".join(str(row[CSV_CONFIG["text"]]).split())
            row_id     = str(row[CSV_CONFIG["id"]])

            # --- Model inference call ---
            if args.mode == "real":
                res = llm_generate(model_name, text_input, system_prompt, INFERENCE_PARAMS)
            else:
                res = mock_generate()

            # --- Parse extracted events ---
            events, parse_error = parse_events(res["response"])

            # --- Construct JSONL record ---
            record = {
                "id":         row_id,
                "input_text": text_input,
                "model":      model_name,
                "setting":    PROMPT_SETTING,
                "prompt_file":PROMPT_FILE,
                "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
                "output": {
                    "events":      events,
                    "parse_error": parse_error,
                    "raw_response":res["response"] if parse_error else None,
                },
                "metadata": {
                    "done":               res["done"],
                    "llm_error":          res["error"],
                    "total_duration":     res.get("total_duration"),
                    "load_duration":      res.get("load_duration"),
                    "prompt_eval_count":  res.get("prompt_eval_count"),
                    "prompt_eval_duration":res.get("prompt_eval_duration"),
                    "eval_count":         res.get("eval_count"),
                    "eval_duration":      res.get("eval_duration"),
                    "thinking":           res.get("thinking"),
                    "hardware":           "NVIDIA GPU" if args.mode == "real" else "N/A (Mock)"
                },
            }

            # --- Append record to JSONL output ---
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")

    print(f"\n[Extraction completed. Output saved to: {OUTPUT_PATH}]")


if __name__ == "__main__":
    main()