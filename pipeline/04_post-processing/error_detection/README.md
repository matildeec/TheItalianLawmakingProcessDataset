# Procedural Error Detection & Quality Assurance

This standalone module performs automated quality audits on extracted legislative event logs by detecting **critical procedural and chronological anomalies**.

The anomalies detected by this engine are directly propagated as a quality flag (`suspicious: true/false`) in the final distributed dataset (`dataset/preparatory-work.jsonl`).

---

## Architecture & Contents

| File | Description |
| :--- | :--- |
| `flag.py` | Core domain-rule detection engine (pure Python standard library, zero external dependencies). |
| `run_flagging.py` | Orchestration script: loads event logs, executes checks, and exports structured results. |
| `results/` | Output directory containing generated audit artifacts: |
| `results/flagged_cases.jsonl` | List of anomalous cases with detailed diagnostics (used by downstream build scripts). |
| `results/flagging_results.json` | Comprehensive machine-readable summary and taxonomy breakdown. |
| `results/flagging_summary.txt` | Human-readable audit report and statistics. |

---

## Critical Anomaly Taxonomy

The module evaluates event logs against core principles of parliamentary procedure:

1. **`SEQUENCE_DATE_MISMATCH`**:
   - Verifies that sequential ordering (`sequence_id`) aligns with event dates (`date`).
   - Flagged when an event declared with a higher sequence order possesses a date strictly earlier than a preceding event.
2. **`CHRONOLOGY_INVERSION`**:
   - Detects severe procedural inversions (e.g., initial bill presentation or committee assignment occurring *after* floor debate or final approval).
3. **`IMPLAUSIBLE_DATE`**:
   - Identifies malformed date strings or dates falling outside the plausible historical range of the Italian Republic (1948–2026).
4. **`UNAUTHORIZED_APPROVAL`**:
   - Flags final passage (`approval`) events that occur without any recorded prior examination (either in committee or on the floor) within that parliamentary chamber.

---

## Dependencies

**None** — uses exclusively the Python standard library (`json`, `os`, `sys`, `re`, `datetime`, `argparse`, `dataclasses`, `enum`).

---

## Usage

### Run from repository root:

```bash
# Audit raw extractions
python pipeline/04_post-processing/error_detection/run_flagging.py \
    --input pipeline/output/full_raw_extraction_gemma4.jsonl \
    --output pipeline/04_post-processing/error_detection/results
```

### Run from inside the module directory:

```bash
cd pipeline/04_post-processing/error_detection/

python run_flagging.py \
    --input ../../output/full_raw_extraction_gemma4.jsonl \
    --output ./results
```

### Custom input / output:

```bash
python flag.py --input /path/to/any_extraction.jsonl
```

---

## Downstream Dataset Integration

The output file `results/flagged_cases.jsonl` serves as the quality index for `pipeline/04_post-processing/build_preparatory_work.py`. Any bill whose `case_id` appears in `flagged_cases.jsonl` is assigned `"suspicious": true` in `dataset/preparatory-work.jsonl`, providing end-users with full transparency regarding potential chronological or procedural parsing irregularities.
