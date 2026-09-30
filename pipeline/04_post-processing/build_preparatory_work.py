"""
build_preparatory_work.py

Constructs the standardized dataset/preparatory-work.jsonl:
1. Merges 5,246 LLM extraction cases (pipeline/output/full_raw_extraction_gemma4.jsonl)
   with the 100 Gold Standard cases (pipeline/data/gold_standard.json + pipeline/data/val_sample_100.csv)
   for a total of exactly 5,346 laws.
2. Strips experimental metadata fields:
   {model, setting, prompt_file, timestamp, metadata, parse_error, raw_response, output}
3. Unnests events so that 'events' is a first-class top-level property.
4. Performs deterministic post-processing on each event using functions from
   pipeline/04_post-processing/postprocess_results.py:
   - English schema & controlled vocabularies (activity, procedural_mode, chamber).
   - Canonicalized commission names and clean sponsor names.
5. Injects the boolean quality flag 'suspicious':
   - True if case_id appears in pipeline/04_post-processing/error_detection/results/flagged_cases.jsonl
   - False otherwise.
"""

import json
import os
import sys
import pandas as pd

# Import post-processing logic
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
sys.path.insert(0, _SCRIPT_DIR)
from postprocess_results import postprocess_event

def load_flagged_case_ids(flagged_path):
    flagged_ids = set()
    if os.path.exists(flagged_path):
        with open(flagged_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    cid = item.get('case_id')
                    if cid:
                        flagged_ids.add(str(cid).strip())
    print(f"Loaded {len(flagged_ids)} flagged case IDs.")
    return flagged_ids

def build_dataset():
    flagged_path = os.path.join(_REPO_ROOT, 'pipeline', '04_post-processing', 'error_detection', 'results', 'flagged_cases.jsonl')
    flagged_ids = load_flagged_case_ids(flagged_path)
    
    # 1. Load Gold Standard texts from val_sample_100.csv
    val_csv_path = os.path.join(_REPO_ROOT, 'pipeline', 'data', 'val_sample_100.csv')
    val_df = pd.read_csv(val_csv_path, low_memory=False)
    gold_texts = dict(zip(val_df['case_id'], val_df['lavori_preparatori']))

    # Load Gold Standard annotations
    gs_path = os.path.join(_REPO_ROOT, 'pipeline', 'data', 'gold_standard.json')
    with open(gs_path, 'r', encoding='utf-8') as f:
        gs_data = json.load(f)
    
    gold_cases = {}
    for item in gs_data.get('casi', []):
        cid = item.get('case_id')
        events = item.get('eventi') or item.get('events') or []
        gold_cases[cid] = events

    print(f"Loaded {len(gold_cases)} gold standard cases.")

    output_path = os.path.join(_REPO_ROOT, 'dataset', 'preparatory-work.jsonl')
    print(f"Writing to {output_path}...")
    
    written_cases = set()
    total_events_processed = 0

    raw_extraction_path = os.path.join(_REPO_ROOT, 'pipeline', 'output', 'full_raw_extraction_gemma4.jsonl')
    with open(output_path, 'w', encoding='utf-8') as f_out:
        # Step A: Process 5,246 LLM extractions
        with open(raw_extraction_path, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                if not line.strip():
                    continue
                rec = json.loads(line)
                case_id = rec.get('id') or rec.get('case_id')
                input_text = rec.get('input_text', '')
                
                out = rec.get('output', {})
                raw_events = out.get('events') or out.get('eventi') or []
                
                # Apply post-processing
                clean_events = [postprocess_event(ev) for ev in raw_events if isinstance(ev, dict)]
                total_events_processed += len(clean_events)

                record_out = {
                    "case_id": case_id,
                    "input_text": input_text,
                    "suspicious": bool(case_id in flagged_ids),
                    "events": clean_events
                }
                f_out.write(json.dumps(record_out, ensure_ascii=False) + '\n')
                written_cases.add(case_id)

        print(f"Written {len(written_cases)} LLM extraction records.")

        # Step B: Process 100 Gold Standard cases
        gold_count = 0
        for cid, raw_events in gold_cases.items():
            if cid in written_cases:
                continue
            input_text = gold_texts.get(cid, '')
            clean_events = [postprocess_event(ev) for ev in raw_events if isinstance(ev, dict)]
            total_events_processed += len(clean_events)

            record_out = {
                "case_id": cid,
                "input_text": input_text,
                "suspicious": bool(cid in flagged_ids),
                "events": clean_events
            }
            f_out.write(json.dumps(record_out, ensure_ascii=False) + '\n')
            written_cases.add(cid)
            gold_count += 1

        print(f"Written {gold_count} Gold Standard records.")

    print(f"\nSUCCESS! Total records written: {len(written_cases)} (Total events: {total_events_processed})")
    print(f"Output saved to: {output_path}")

if __name__ == '__main__':
    build_dataset()
