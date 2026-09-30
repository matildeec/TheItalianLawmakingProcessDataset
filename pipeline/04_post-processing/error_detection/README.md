# flagging_pipeline

Standalone pipeline to flag **CRITICAL errors** in PM4Lawmaking 2.0 legislative event logs.

## Contents

| File | Description |
|------|-------------|
| `flag.py` | All CRITICAL error detection logic (zero external dependencies — stdlib only) |
| `run_flagging.py` | Pipeline script: loads dataset → runs checks → saves results |
| `results/` | Output directory (created automatically on first run) |

## Dependencies

**None** — only Python ≥ 3.8 stdlib (`json`, `os`, `sys`, `re`, `datetime`, `argparse`, `dataclasses`, `enum`).

## Usage

### Run with default paths (from inside `flagging_pipeline/`)
```bash
cd flagging_pipeline/
python run_flagging.py
```

The default input is `../../output_final_extraction/extraction_few-shot_merged_full.jsonl`.

### Custom input/output
```bash
python run_flagging.py \
    --input  /path/to/extraction_merged_full.jsonl \
    --output ./results
```

### With Gold Standard (adds hallucination & omission checks)
```bash
python run_flagging.py \
    --input  /path/to/extraction.jsonl \
    --gold   /path/to/gold_standard.json \
    --output ./results
```

### Self-test flag.py
```bash
python flag.py --test
```

---

## Output files

| File | Description |
|------|-------------|
| `results/flagging_results.json` | Full structured output: summary stats + every anomaly |
| `results/flagged_cases.jsonl` | One line per flagged case `{case_id, anomaly_count, anomalies}` — for downstream use |
| `results/flagging_summary.txt` | Human-readable report with category breakdown |

---

## CRITICAL checks performed

| # | Check | Category |
|---|-------|----------|
| 1 | Invalid / out-of-range date (1948–2030) | `IMPLAUSIBLE_DATE` |
| 2 | `sequence_id` order doesn't match `date` order | `SEQUENCE_DATE_MISMATCH` |
| 3 | Temporal inversion involving key procedural steps | `CHRONOLOGY_INVERSION` |
| 4 | `richiesta_parere` with `sede ≠ consultiva` | `SEDE_INCOMPATIBILITY` |
| 5 | `approvazione` without prior `esame_commissione` / `esame_aula` | `UNAUTHORIZED_APPROVAL` |
| 6 | Hallucinated / omitted events vs Gold Standard *(optional)* | `LLM_EVENT_HALLUCINATION` / `LLM_EVENT_OMISSION` |
