# ================================================
# --- Usage Instructions
#
# Evaluation from configuration (reads FILES_TO_EVALUATE in config_eval.py or latest file in output/):
# $ python run_evaluation.py
#
# Evaluation of a specific file via command line:
# $ python run_evaluation.py --input output/extraction_zero-shot_2026-08-13_18-19.jsonl
#
# Force re-evaluation of existing entries:
# $ python run_evaluation.py --force
# ================================================

import os
import sys
import glob
import json
import argparse
import time
import pandas as pd

# Import paths and parameters from config_eval.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_eval import (
    GOLD_STANDARD_PATH,
    EXTRACTION_OUTPUT_DIR,
    FILES_TO_EVALUATE,
    RESULTS_PATH,
    HUNGARIAN_SIMILARITY_THRESHOLD,
    SENTENCE_TRANSFORMER_MODEL,
    BASE_DIR,
)

from utils import (
    verify_prompt_schema_compliance,
    verify_exact_match_jaccard,
    verify_hungarian_matching,
)

try:
    from sentence_transformers import SentenceTransformer
    import torch
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


def load_gold_standard(gold_path: str) -> dict:
    """
    Load Gold Standard annotations from a JSON or JSONL file.
    
    Returns:
        dict: Mapping {case_id: list_of_gold_events}
    """
    gold_data = {}
    if not os.path.exists(gold_path):
        raise FileNotFoundError(f"Gold Standard file not found: {gold_path}")

    with open(gold_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    try:
        parsed_json = json.loads(content)
        if isinstance(parsed_json, dict) and "casi" in parsed_json:
            for case in parsed_json["casi"]:
                c_id = str(case.get("case_id") or case.get("id"))
                gold_data[c_id] = case.get("eventi") or case.get("events", [])
            return gold_data
    except Exception:
        pass

    for line in content.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line.strip())
        c_id = str(rec.get("id") or rec.get("case_id"))
        gold_data[c_id] = rec.get("events") or rec.get("eventi", [])

    return gold_data


def get_evaluated_models(output_json_path: str, target_setting: str = None, target_eval_mode: str = None) -> set:
    """
    Read the evaluation results JSON file (if it exists) and return
    a set of model names already evaluated for the specific setting and eval_mode.
    """
    evaluated_models = set()
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for rec in data:
                        model = rec.get("model")
                        rec_setting = rec.get("setting")
                        rec_mode = rec.get("eval_mode")
                        if model:
                            setting_match = target_setting is None or rec_setting is None or rec_setting == target_setting
                            mode_match = target_eval_mode is None or rec_mode is None or rec_mode == target_eval_mode
                            if setting_match and mode_match:
                                evaluated_models.add(model)
        except (json.JSONDecodeError, Exception):
            pass
    return evaluated_models


def save_evaluation_results(
    results_by_model: dict, output_json_path: str, input_file: str, gold_file: str, setting: str, prompt_file: str, mode_label: str
):
    """
    Save or update model evaluations in JSON format with metadata.
    If the combination (model, setting, eval_mode) already exists in the file, it is updated.
    """
    if not results_by_model:
        return

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    eval_mode_str = "raw"
    new_keys = {(m, setting, eval_mode_str) for m in results_by_model.keys()}

    existing_records = []
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for rec in data:
                        rec_key = (rec.get("model"), rec.get("setting"), rec.get("eval_mode"))
                        if rec_key not in new_keys:
                            existing_records.append(rec)
        except (json.JSONDecodeError, Exception):
            existing_records = []

    for model_name, metrics_dict in results_by_model.items():
        record = {
            "model": model_name,
            "setting": setting,
            "prompt_file": prompt_file,
            "eval_mode": eval_mode_str,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_file": os.path.basename(input_file),
            "gold_file": os.path.basename(gold_file),
            "evaluations": metrics_dict,
        }
        existing_records.append(record)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(existing_records, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Results saved/updated in: {output_json_path}")


def evaluate_single_setting(
    input_file: str,
    gold_file: str,
    output_json_path: str,
    mode_label: str,
    threshold: float,
    force: bool,
    embedder,
    gold_data: dict,
):
    """
    Execute the evaluation cycle for a single setting (RAW) and save/append results.
    """
    records_by_model = {}
    records_setting = {}
    records_prompt = {}

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line.strip())
            m = rec.get("model", "Unknown-Model")
            records_by_model.setdefault(m, []).append(rec)
            if m not in records_setting:
                # Infer setting (few-shot vs zero-shot) from record or filename
                st = rec.get("setting")
                if not st:
                    st = "few-shot" if "few-shot" in input_file.lower() or "few_shot" in input_file.lower() else "zero-shot"
                records_setting[m] = st
                records_prompt[m] = rec.get("prompt_file", "event_extraction_instruct.txt")

    # Determine primary setting for execution
    sample_model = list(records_by_model.keys())[0] if records_by_model else "Unknown"
    current_setting = records_setting.get(sample_model, "zero-shot")
    current_prompt = records_prompt.get(sample_model, "event_extraction_instruct.txt")

    eval_mode_str = "raw"
    already_evaluated = get_evaluated_models(output_json_path, target_setting=current_setting, target_eval_mode=eval_mode_str)

    models_to_evaluate = {}
    for m, recs in records_by_model.items():
        if m in already_evaluated and not force:
            print(f"[SKIP] Model '{m}' ({current_setting}) already evaluated for setting {mode_label} in '{output_json_path}'. Skipping.")
        else:
            models_to_evaluate[m] = recs

    if not models_to_evaluate:
        print(f"[INFO] No new evaluations to run for setting {mode_label} ({current_setting}).")
        return

    print("\n" + "=" * 80)
    print(f"MODEL EVALUATION - SETTING: {mode_label.upper()} | INFERENCE: {current_setting.upper()}")
    print("=" * 80)
    print(f"Input File:          {input_file}")
    print(f"Gold Standard:       {gold_file}")
    print(f"Setting:             {current_setting}")
    print(f"Prompt File:         {current_prompt}")
    print(f"Output File:         {output_json_path}")
    print(f"Match Threshold:     {threshold}\n")

    results_by_model = {}

    for model_name, recs in models_to_evaluate.items():
        print(f"\n>>> RUNNING EVALUATION [{mode_label.upper()} | {current_setting.upper()}] FOR: {model_name} ({len(recs)} documents) <<<")

        schema_res = verify_prompt_schema_compliance(recs)
        print("  [OK] Prompt Schema Compliance: completed.")

        jaccard_res = verify_exact_match_jaccard(recs, gold_data)
        print("  [OK] Exact Match Event Jaccard per Case: completed.")

        hungarian_res = verify_hungarian_matching(recs, gold_data, threshold=threshold, embedder=embedder)
        print("  [OK] Hungarian Algorithm Event Matching: completed.")

        results_by_model[model_name] = {
            "prompt_schema_compliance": schema_res,
            "exact_match_event_jaccard": jaccard_res,
            "hungarian_event_matching": hungarian_res,
        }

    # --- GLOBAL EVALUATION SUMMARY ---
    print("\n" + "=" * 80)
    print(f"GLOBAL EVALUATION SUMMARY [{mode_label.upper()} | {current_setting.upper()}]")
    print("=" * 80)

    summary_table = []
    for m_name, m_res in results_by_model.items():
        s_info = m_res.get("prompt_schema_compliance", {})
        e_info = m_res["exact_match_event_jaccard"]
        h_info = m_res["hungarian_event_matching"]

        summary_table.append({
            "Model": m_name,
            "Setting": current_setting,
            "Total Docs": len(records_by_model[m_name]),
            "Extracted Events": s_info.get("total_events_extracted", 0),
            "Schema Valid (%)": f"{s_info.get('schema_adherence_rate_pct', 0)}% ({s_info.get('schema_valid_events', 0)}/{s_info.get('total_events_extracted', 0)})",
            "Exact Jaccard (Mean)": e_info["mean_event_jaccard_per_case"],
            "Perfect Cases (%)": f"{e_info['perfect_cases_pct']}% ({e_info['perfect_cases_count']}/{e_info['total_cases_evaluated']})",
            "Critical Cases (J<0.5)": f"{e_info['critical_cases_pct']}% ({e_info['critical_cases_count']}/{e_info['total_cases_evaluated']})",
            "Hungarian Micro F1": h_info["micro_f1"],
            "Hungarian Macro F1": h_info["macro_f1"],
        })

    df_sum = pd.DataFrame(summary_table)
    print(df_sum.to_markdown(index=False))
    print("=" * 80)

    # --- ATTRIBUTE-LEVEL ACCURACY BREAKDOWN MATRIX ---
    print("\n>>> ATTRIBUTE-LEVEL ACCURACY MATRIX (EXACT ATTRIBUTE MATCH) <<<")
    for m_name, m_res in results_by_model.items():
        attr_acc = m_res["exact_match_event_jaccard"].get("attribute_accuracy", {})
        print(f"\nModel: {m_name} ({current_setting})")
        if attr_acc:
            attr_table = []
            for attr, info in sorted(attr_acc.items(), key=lambda x: x[1]["accuracy_pct"]):
                attr_table.append({
                    "Attribute": attr,
                    "Exact Accuracy (%)": f"{info['accuracy_pct']}%",
                    "Exact Matches": f"{info['matched']} / {info['total']}"
                })
            df_attr = pd.DataFrame(attr_table)
            print(df_attr.to_markdown(index=False))
        else:
            print("  [INFO] Attribute details not available.")
    print("=" * 80)

    # --- HUNGARIAN SEMANTIC DETAILS ---
    print("\n>>> SEMANTIC FIDELITY ANALYSIS (HUNGARIAN ALGORITHM) <<<")
    for m_name, m_res in results_by_model.items():
        h_info = m_res["hungarian_event_matching"]
        print(f"\nModel: {m_name} ({current_setting}) | Similarity Threshold: {h_info['similarity_threshold']}")
        print(f"  • Matched Events:     {h_info['total_matched_events']} / {h_info['total_gold_events']} gold (Extracted: {h_info['total_predicted_events']})")
        print(f"  • Micro-Averaged:     Precision: {h_info['micro_precision']} | Recall: {h_info['micro_recall']} | Micro-F1: {h_info['micro_f1']}")
        print(f"  • Macro-Averaged:     Precision: {h_info['macro_precision']} | Recall: {h_info['macro_recall']} | Macro-F1: {h_info['macro_f1']}")
        print(f"  • 100% Semantic Cases: {h_info['perfect_semantic_cases_count']}/{len(records_by_model[m_name])} ({h_info['perfect_semantic_cases_pct']}%)")
    print("=" * 80)

    # --- PROMPT SCHEMA ERROR BREAKDOWN ---
    print("\n>>> PROMPT SCHEMA VIOLATION BREAKDOWN BY ATTRIBUTE <<<")
    for m_name, m_res in results_by_model.items():
        print(f"\nModel: {m_name} ({current_setting})")
        s_res = m_res.get("prompt_schema_compliance", {})
        err_breakdown = s_res.get("attribute_error_breakdown", {})
        if not err_breakdown:
            print("  [OK] No schema violations found! All events comply with the prompt schema.")
        else:
            for attr, count in sorted(err_breakdown.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {attr}: {count} violations")

    # --- CASES WITH SCHEMA VIOLATIONS ---
    print("\n>>> CASES WITH SCHEMA VIOLATIONS (FOR MANUAL AUDIT) <<<")
    for m_name, m_res in results_by_model.items():
        s_res = m_res.get("prompt_schema_compliance", {})
        invalid_cases = s_res.get("invalid_cases", {})
        print(f"\nModel: {m_name} ({current_setting}) | ({len(invalid_cases)} cases with errors out of {len(records_by_model[m_name])})")
        if not invalid_cases:
            print("  [OK] No case_ids contain schema errors!")
        else:
            for c_id, info in list(invalid_cases.items())[:10]:
                print(f"  - case_id: {c_id} ({info['error_count']} errors)")
                for e_msg in info["errors"][:3]:
                    print(f"      • {e_msg}")
            if len(invalid_cases) > 10:
                print(f"  ... and {len(invalid_cases) - 10} additional cases with schema errors.")

    # --- CRITICAL CASES LIST ---
    print("\n>>> CRITICAL CASES LIST (EXACT MATCH JACCARD < 0.5) FOR MANUAL AUDIT <<<")
    for m_name, m_res in results_by_model.items():
        e_info = m_res["exact_match_event_jaccard"]
        crit_details = e_info.get("critical_case_details", [])
        perf_count = e_info.get("perfect_cases_count", 0)
        part_count = e_info.get("partial_cases_count", 0)
        tot_c = e_info.get("total_cases_evaluated", 0)

        print(f"\nModel: {m_name} ({current_setting})")
        
        t1 = e_info.get("tier1_perfect_count", perf_count)
        t2 = e_info.get("tier2_high_conf_count", 0)
        t3 = e_info.get("tier3_medium_conf_count", part_count - t2)
        t4 = e_info.get("tier4_drop_count", len(crit_details))
        
        t1_pct = round((t1 / tot_c * 100), 1) if tot_c > 0 else 0.0
        t2_pct = round((t2 / tot_c * 100), 1) if tot_c > 0 else 0.0
        t3_pct = round((t3 / tot_c * 100), 1) if tot_c > 0 else 0.0
        t4_pct = round((t4 / tot_c * 100), 1) if tot_c > 0 else 0.0

        print(f"  • 100% Perfect Cases (J=1.0): {t1}/{tot_c} ({t1_pct}%)")
        print(f"  • High-Confidence Cases (0.8<=J<1.0): {t2}/{tot_c} ({t2_pct}%)")
        print(f"  • Medium-Confidence Cases (0.5<=J<0.8): {t3}/{tot_c} ({t3_pct}%)")
        print(f"  • Critical Cases (J<0.5): {t4}/{tot_c} ({t4_pct}%)")
        
        safe_tot = t1 + t2
        safe_pct = round((safe_tot / tot_c * 100), 1) if tot_c > 0 else 0.0
        print(f"  => Total Safe Cases (J>=0.8): {safe_tot}/{tot_c} ({safe_pct}%)")

        if not crit_details:
            print("  [OK] No critical cases encountered!")
        else:
            print(f"  Listing first {min(15, len(crit_details))} critical cases:")
            for item in crit_details[:15]:
                print(f"    - case_id: {item['case_id']} | Jaccard: {item['jaccard']} | Matched Events: {item['matched_events']}/{item['gold_events']} (Pred: {item['pred_events']})")
            if len(crit_details) > 15:
                print(f"    ... and {len(crit_details) - 15} additional critical cases.")

    print("=" * 80)

    # Save to unified output file (evaluation_results.json)
    save_evaluation_results(results_by_model, output_json_path, input_file, gold_file, current_setting, current_prompt, mode_label)


def resolve_file_path(file_entry: str) -> str:
    """
    Resolve a relative file path by checking EXTRACTION_OUTPUT_DIR or BASE_DIR.
    """
    if os.path.exists(file_entry):
        return file_entry

    candidate1 = os.path.join(EXTRACTION_OUTPUT_DIR, file_entry)
    if os.path.exists(candidate1):
        return candidate1

    candidate2 = os.path.join(BASE_DIR, file_entry)
    if os.path.exists(candidate2):
        return candidate2

    return file_entry


def main():
    parser = argparse.ArgumentParser(
        description="The Italian Lawmaking Process Dataset – Evaluation Orchestrator (LLM Output Validation against Gold Standard)"
    )
    parser.add_argument("--input", type=str, nargs="*", default=None, help="Inference JSONL output file(s) to evaluate.")
    parser.add_argument("--gold", type=str, default=None, help=f"Gold Standard file path. Default: {GOLD_STANDARD_PATH}")
    parser.add_argument(
        "--mode",
        choices=["raw"],
        default="raw",
        help="Evaluation mode: validate raw model extraction output. Default: raw",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=HUNGARIAN_SIMILARITY_THRESHOLD,
        help=f"Similarity threshold for the Hungarian Algorithm. Default: {HUNGARIAN_SIMILARITY_THRESHOLD}",
    )
    parser.add_argument(
        "--force",
        "--overwrite",
        action="store_true",
        help="Force re-evaluation even if the model already exists in output results.",
    )
    args = parser.parse_args()

    # 1. Determine files to evaluate
    target_files = []
    if args.input:
        target_files = [resolve_file_path(f) for f in args.input]
    elif FILES_TO_EVALUATE:
        print(f"[INFO] Using {len(FILES_TO_EVALUATE)} files specified in 'FILES_TO_EVALUATE' of config_eval.py.")
        target_files = [resolve_file_path(f) for f in FILES_TO_EVALUATE]
    else:
        search_dirs = [
            EXTRACTION_OUTPUT_DIR,
            os.path.join(BASE_DIR, "output"),
            os.path.join(BASE_DIR, "02_llm-extraction", "output"),
        ]
        jsonl_files = []
        for d in search_dirs:
            if os.path.exists(d):
                jsonl_files.extend(glob.glob(os.path.join(d, "*.jsonl")))
        if not jsonl_files:
            print("[ERROR] No .jsonl extraction files found.")
            sys.exit(1)
        jsonl_files.sort(key=os.path.getmtime, reverse=True)
        target_files = [jsonl_files[0]]
        print(f"[INFO] No file specified. Automatically evaluating most recent file: {target_files[0]}")

    # 2. Locate Gold Standard file
    gold_file = args.gold if args.gold else GOLD_STANDARD_PATH
    if not os.path.exists(gold_file):
        gold_file_alt = os.path.join(BASE_DIR, "data", "gold_standard.jsonl")
        if os.path.exists(gold_file_alt):
            gold_file = gold_file_alt
        else:
            print(f"[ERROR] Gold Standard file not found at '{gold_file}' or '{gold_file_alt}'.")
            sys.exit(1)

    gold_data = load_gold_standard(gold_file)

    # 3. Load Embedder (if available)
    embedder = None
    if HAS_SENTENCE_TRANSFORMERS:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"[INFO] Loading Sentence Transformer ({SENTENCE_TRANSFORMER_MODEL}) on device '{device}'...")
        try:
            embedder = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, device=device)
            print(f"[INFO] Embedder successfully loaded on {device.upper()}.")
        except Exception as e:
            print(f"[WARNING] SentenceTransformer failed: {e}. Falling back to token Jaccard similarity.")
            embedder = None

    # 4. Run evaluation for each file in target_files
    for input_file in target_files:
        if not os.path.exists(input_file):
            print(f"[ERROR] Input file not found: {input_file}. Skipping.")
            continue

        evaluate_single_setting(
            input_file=input_file,
            gold_file=gold_file,
            output_json_path=RESULTS_PATH,
            mode_label="RAW (LLM Output)",
            threshold=args.threshold,
            force=args.force,
            embedder=embedder,
            gold_data=gold_data,
        )


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

if __name__ == "__main__":
    base_name, _ = os.path.splitext(RESULTS_PATH)
    log_file = f"{base_name}.txt"
    sys.stdout = Logger(log_file)
    print(f"[INFO] Terminal output will also be saved to: {log_file}")
    try:
        main()
    finally:
        sys.stdout.log.close()
        sys.stdout = sys.stdout.terminal

