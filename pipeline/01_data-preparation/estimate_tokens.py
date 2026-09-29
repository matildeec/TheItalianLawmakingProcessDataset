import pandas as pd
import json
import glob
import os
import argparse
from datetime import datetime
import tiktoken
from tokenizers import Tokenizer

# --- PATH CONFIGURATION ---
BASEDIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASEDIR)

DEFAULT_INPUT_CSV = os.path.join(PROJECT_ROOT, "data", "dataset_with_legislatures.csv")
DEFAULT_GOLD_JSON = os.path.join(PROJECT_ROOT, "data", "gold_standard.json")
DEFAULT_VAL_CSV = os.path.join(PROJECT_ROOT, "data", "val_sample_100.csv")
DEFAULT_PROMPT_FILE = os.path.join(PROJECT_ROOT, "02_llm-extraction", "prompts", "event_extraction_instruct.txt")
DEFAULT_FEWSHOT_DIR = os.path.join(PROJECT_ROOT, "data", "fewshot_examples")
DEFAULT_OUTPUT_JSON = os.path.join(BASEDIR, "token_statistics.json")

TEXT_COL = "lavori_preparatori"
ID_COL = "case_id"


def load_tokenizers():
    """Load standard tokenizers used in open-weight and reference LLMs."""
    tokenizers = {}

    # 1. Gemma (Google) - Tokenizer proxy for Gemma family models (including Ollama gemma4:31b)
    try:
        tok_gemma = Tokenizer.from_pretrained("unsloth/gemma-2-27b-it")
        tokenizers["gemma"] = {
            "name": "Gemma (Google)",
            "vocab_size": tok_gemma.get_vocab_size(),
            "type": "hf",
            "obj": tok_gemma,
        }
    except Exception as e:
        print(f"[WARN] Gemma tokenizer could not be loaded: {e}")

    # 2. Qwen 2.5 (Alibaba)
    try:
        tok_qwen = Tokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")
        tokenizers["qwen"] = {
            "name": "Qwen 2.5 (Alibaba)",
            "vocab_size": tok_qwen.get_vocab_size(),
            "type": "hf",
            "obj": tok_qwen,
        }
    except Exception as e:
        print(f"[WARN] Qwen tokenizer could not be loaded: {e}")

    # 3. Mistral (Mistral AI)
    try:
        tok_mistral = Tokenizer.from_pretrained("mistralai/Mistral-Small-24B-Instruct-2501")
        tokenizers["mistral"] = {
            "name": "Mistral Small (Mistral AI)",
            "vocab_size": tok_mistral.get_vocab_size(),
            "type": "hf",
            "obj": tok_mistral,
        }
    except Exception as e:
        print(f"[WARN] Mistral tokenizer could not be loaded: {e}")

    # 4. OpenAI BPE (cl100k_base)
    try:
        tok_openai = tiktoken.get_encoding("cl100k_base")
        tokenizers["openai_bpe"] = {
            "name": "OpenAI BPE (cl100k_base)",
            "vocab_size": 100256,
            "type": "tiktoken",
            "obj": tok_openai,
        }
    except Exception as e:
        print(f"[WARN] OpenAI tiktoken could not be loaded: {e}")

    return tokenizers


def count_tokens(text: str, tok_info: dict) -> int:
    """Calculate the exact token count for a text string."""
    if not text or pd.isna(text):
        return 0
    if tok_info["type"] == "hf":
        return len(tok_info["obj"].encode(str(text)).ids)
    elif tok_info["type"] == "tiktoken":
        return len(tok_info["obj"].encode(str(text)))
    return len(str(text).split())


def load_prompts(prompt_path: str, fewshot_dir: str):
    """Load Zero-Shot and Few-Shot system prompts."""
    if not os.path.exists(prompt_path):
        return "", ""

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_base = f.read().strip()

    files = sorted(glob.glob(os.path.join(fewshot_dir, "*.json")))
    examples = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        text = data.get("text", "")
        eventi = data.get("eventi", [])
        ex_str = f"Input:\n\"{text}\"\n\nOutput:\n{{\n  \"eventi\": {json.dumps(eventi, indent=2, ensure_ascii=False)}\n}}"
        examples.append(ex_str)

    fewshot_str = "### ESEMPI DI ESTRAZIONE (FEW-SHOT):\n\n" + "\n\n---\n\n".join(examples)
    if "### ESEMPIO DI ESTRAZIONE" in prompt_base:
        prompt_fewshot = prompt_base.split("### ESEMPIO DI ESTRAZIONE")[0].strip() + "\n\n" + fewshot_str
    else:
        prompt_fewshot = prompt_base + "\n\n" + fewshot_str

    return prompt_base, prompt_fewshot


def get_series_percentiles(series: pd.Series) -> dict:
    """Return standard statistical summary metrics in a structured dictionary."""
    return {
        "count": int(len(series)),
        "mean": round(float(series.mean()), 2),
        "std": round(float(series.std()), 2),
        "min": int(series.min()),
        "p25": round(float(series.quantile(0.25)), 1),
        "median": round(float(series.quantile(0.50)), 1),
        "p75": round(float(series.quantile(0.75)), 1),
        "p90": round(float(series.quantile(0.90)), 1),
        "p95": round(float(series.quantile(0.95)), 1),
        "p99": round(float(series.quantile(0.99)), 1),
        "p99_9": round(float(series.quantile(0.999)), 1),
        "max": int(series.max())
    }


def main():
    parser = argparse.ArgumentParser(description="The Italian Lawmaking Process Dataset – Tokenizer & Context Estimator")
    parser.add_argument("--csv", default=DEFAULT_INPUT_CSV, help="Path to primary dataset CSV")
    parser.add_argument("--gold", default=DEFAULT_GOLD_JSON, help="Path to annotated Gold Standard JSON")
    parser.add_argument("--val-csv", default=DEFAULT_VAL_CSV, help="Path to validation sample CSV")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT_FILE, help="Path to prompt file")
    parser.add_argument("--fewshot-dir", default=DEFAULT_FEWSHOT_DIR, help="Directory containing Few-Shot example JSONs")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON, help="Path to output statistics JSON file")
    args = parser.parse_args()

    print("=" * 80)
    print(" The Italian Lawmaking Process Dataset – Tokenizer & Context Estimator")
    print("=" * 80)

    # 1. Load Tokenizers
    print("\n1. Initializing native tokenizers (Gemma, Qwen, Mistral, OpenAI)...")
    tokenizers = load_tokenizers()
    if not tokenizers:
        print("[ERROR] No tokenizer available.")
        return

    # 2. Load Dataset
    print(f"2. Loading primary dataset from: {args.csv}")
    if not os.path.exists(args.csv):
        print(f"[ERROR] File {args.csv} not found.")
        return

    df_main = pd.read_csv(args.csv)
    df_main[TEXT_COL] = df_main[TEXT_COL].fillna("")
    print(f"   -> {len(df_main)} texts loaded.")

    # Reference dataset for Gold Standard texts
    df_ref = df_main
    if os.path.exists(args.val_csv):
        df_val = pd.read_csv(args.val_csv)
        df_ref = pd.concat([df_main, df_val]).drop_duplicates(subset=[ID_COL])

    # 3. General dataset overview (Words and Legislatures)
    df_main["word_count"] = df_main[TEXT_COL].apply(lambda x: len(str(x).split()))
    word_stats = get_series_percentiles(df_main["word_count"])
    leg_dist = df_main["legislature"].value_counts().to_dict() if "legislature" in df_main.columns else {}

    dataset_overview = {
        "source_file": os.path.relpath(args.csv, PROJECT_ROOT),
        "total_laws": len(df_main),
        "legislature_distribution": leg_dist,
        "input_word_distribution": word_stats
    }

    # 4. Gold Standard overview
    gold_overview = {}
    gold_cases = []
    if os.path.exists(args.gold):
        with open(args.gold, "r", encoding="utf-8") as f:
            gold_data = json.load(f)
        gold_cases = gold_data.get("casi", gold_data)
        event_counts = pd.Series([len(c.get("eventi", [])) for c in gold_cases])
        gold_overview = {
            "source_file": os.path.relpath(args.gold, PROJECT_ROOT),
            "total_annotated_laws": len(gold_cases),
            "total_events": int(event_counts.sum()),
            "events_per_law_distribution": {
                "mean": round(float(event_counts.mean()), 2),
                "std": round(float(event_counts.std()), 2),
                "min": int(event_counts.min()),
                "median": round(float(event_counts.median()), 1),
                "p90": round(float(event_counts.quantile(0.90)), 1),
                "p95": round(float(event_counts.quantile(0.95)), 1),
                "max": int(event_counts.max())
            }
        }

    # 5. Analysis per tokenizer
    prompt_zero, prompt_few = load_prompts(args.prompt, args.fewshot_dir)
    text_map = dict(zip(df_ref[ID_COL], df_ref[TEXT_COL])) if (ID_COL in df_ref.columns and TEXT_COL in df_ref.columns) else {}

    token_metrics = {}
    table_rows = []

    for key, tok_info in tokenizers.items():
        name = tok_info["name"]

        # Prompt tokens
        p_zero_tokens = count_tokens(prompt_zero, tok_info)
        p_few_tokens = count_tokens(prompt_few, tok_info)

        # Input dataset tokens
        s_input = df_main[TEXT_COL].apply(lambda x: count_tokens(x, tok_info))
        input_percentiles = get_series_percentiles(s_input)
        max_in_idx = s_input.idxmax()
        max_case_id = str(df_main.loc[max_in_idx, ID_COL]) if ID_COL in df_main.columns else "N/A"
        max_title = str(df_main.loc[max_in_idx, "titolo"]) if "titolo" in df_main.columns else "N/A"

        # Gold Standard per tokenizer
        gold_metrics = {}
        if gold_cases:
            gold_rows = []
            for c in gold_cases:
                cid = c.get(ID_COL, c.get("id", ""))
                txt = text_map.get(cid, c.get("text", c.get(TEXT_COL, "")))
                ev = c.get("eventi", c.get("events", []))
                json_str = json.dumps(ev, indent=2, ensure_ascii=False)

                t_in = count_tokens(txt, tok_info)
                t_out = count_tokens(json_str, tok_info)
                gold_rows.append({
                    "case_id": cid,
                    "input_tokens": t_in,
                    "output_tokens": t_out,
                    "events": len(ev),
                    "ratio": t_out / t_in if t_in > 0 else 0.0
                })

            df_g = pd.DataFrame(gold_rows)
            max_g_in = df_g.loc[df_g["input_tokens"].idxmax()]
            max_g_out = df_g.loc[df_g["output_tokens"].idxmax()]

            gold_metrics = {
                "max_input": {
                    "case_id": max_g_in["case_id"],
                    "tokens": int(max_g_in["input_tokens"])
                },
                "max_output": {
                    "case_id": max_g_out["case_id"],
                    "tokens": int(max_g_out["output_tokens"]),
                    "events_count": int(max_g_out["events"])
                },
                "empirical_expansion_ratio": {
                    "mean": round(float(df_g["ratio"].mean()), 2),
                    "median": round(float(df_g["ratio"].median()), 2),
                    "p90": round(float(df_g["ratio"].quantile(0.90)), 2),
                    "p95": round(float(df_g["ratio"].quantile(0.95)), 2),
                    "max": round(float(df_g["ratio"].max()), 2)
                },
                "avg_tokens_per_event": round(float((df_g["output_tokens"] / df_g["events"]).mean()), 1)
            }

        # Worst-case context budget estimation
        worst_input = input_percentiles["max"]
        max_ratio = gold_metrics.get("empirical_expansion_ratio", {}).get("max", 15.0)
        projected_max_output = int(round(worst_input * max_ratio))
        worst_case_total_context = p_few_tokens + worst_input + projected_max_output

        context_budget = {
            "few_shot_prompt_tokens": p_few_tokens,
            "max_input_tokens": worst_input,
            "projected_worst_case_output_tokens": projected_max_output,
            "worst_case_total_context_tokens": worst_case_total_context,
            "recommended_num_predict": 49152,
            "recommended_num_ctx": 65536,
            "context_headroom_tokens": 65536 - (p_few_tokens + worst_input + 49152) if (p_few_tokens + worst_input + 49152) <= 65536 else 0
        }

        token_metrics[key] = {
            "model_name": name,
            "vocab_size": tok_info["vocab_size"],
            "prompts": {
                "zero_shot_tokens": p_zero_tokens,
                "few_shot_tokens": p_few_tokens
            },
            "input_token_distribution": input_percentiles,
            "longest_input_case": {
                "case_id": max_case_id,
                "title": max_title,
                "input_tokens": worst_input
            },
            "gold_standard_empirical": gold_metrics,
            "context_budget_analysis": context_budget
        }

        # Rows for summary table display
        table_rows.append({
            "Tokenizer": name,
            "Vocab": tok_info["vocab_size"],
            "Prompt (Few-Shot)": p_few_tokens,
            "Input (Median)": input_percentiles["median"],
            "Input (p95)": input_percentiles["p95"],
            "Input (Max)": worst_input,
            "Gold Out (Max)": gold_metrics.get("max_output", {}).get("tokens", "N/A"),
            "Ratio Max": gold_metrics.get("empirical_expansion_ratio", {}).get("max", "N/A"),
            "Projected Out": projected_max_output
        })

    # 6. Final JSON structure
    final_output = {
        "metadata": {
            "description": "Italian Lawmaking Process Dataset Token Distribution, Empirical Expansion Ratios and Context Budgeting",
            "generated_at": datetime.now().isoformat(),
            "target_paper_section": "Dataset Construction & Event Extraction Hyperparameters"
        },
        "dataset_overview": dataset_overview,
        "gold_standard_overview": gold_overview,
        "token_metrics": token_metrics
    }

    # Save JSON
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] Unified JSON file saved to:")
    print(f"     -> {os.path.relpath(args.output_json, PROJECT_ROOT)}")

    # 7. Print summary table
    print("\n" + "=" * 105)
    print(f"{'THE ITALIAN LAWMAKING PROCESS DATASET – TOKEN DISTRIBUTION & CONTEXT BUDGETING SUMMARY':^105}")
    print("=" * 105)
    df_table = pd.DataFrame(table_rows)
    print(df_table.to_string(index=False))
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()