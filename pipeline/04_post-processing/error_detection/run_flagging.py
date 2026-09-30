#!/usr/bin/env python3
"""
run_flagging.py - Pipeline to flag CRITICAL errors in the full extracted dataset.

Loads extraction_few-shot_merged_full.jsonl, runs all CRITICAL error detection
checks defined in flag.py, and saves:
  - flagging_results.json    : full structured results (summary + all anomalies)
  - flagging_summary.txt     : human-readable report printed to stdout and saved
  - flagged_cases.jsonl      : one line per flagged case with its anomalies (for downstream use)

Usage:
  # Default paths (run from inside flagging_pipeline/ or from project root):
  $ python run_flagging.py

  # Custom input/output:
  $ python run_flagging.py \\
      --input  ../../output_final_extraction/extraction_few-shot_merged_full.jsonl \\
      --output ./results
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from typing import Dict, List, Any

# ---------------------------------------------------------------------------
# Locate flag.py — works whether this script is run from:
#   • flagging_pipeline/       (same dir)
#   • 04_error_detection/      (parent dir)
#   • project root             (any relative path given via --input)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from flag import flag_dataset  # noqa: E402  (after path fix)


# ---------------------------------------------------------------------------
# Default paths (relative to project root, assuming standard repo layout)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

DEFAULT_INPUT = os.path.join(
    _PROJECT_ROOT,
    "output_final_extraction",
    "extraction_few-shot_merged_full.jsonl",
)
DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "results")


# ---------------------------------------------------------------------------
# Loader: reads the merged full JSONL into {case_id: [events]}
# ---------------------------------------------------------------------------
def load_extraction_jsonl(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Reads a JSONL extraction file where each line is a record with:
      - 'id' or 'case_id'  : case identifier
      - 'output'           : dict with 'events' or 'eventi' key

    Returns a dict  { case_id -> list_of_events }.
    Records where output.events is empty or missing are included as empty lists
    (so they count toward total_cases but yield no anomalies).
    """
    cases: Dict[str, List[Dict[str, Any]]] = {}
    skipped = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  [WARN] Line {line_no}: JSON parse error — {exc}", file=sys.stderr)
                skipped += 1
                continue

            # Resolve case_id
            case_id = str(rec.get("id") or rec.get("case_id") or f"case_{line_no}")

            # Resolve events list
            evs: List[Dict[str, Any]] = []
            output = rec.get("output")
            if isinstance(output, dict):
                evs = output.get("events") or output.get("eventi") or []
            elif isinstance(output, list):
                evs = output
            elif "events" in rec:
                evs = rec["events"]
            elif "eventi" in rec:
                evs = rec["eventi"]

            if not isinstance(evs, list):
                evs = []

            cases[case_id] = evs

    if skipped:
        print(f"  [WARN] Skipped {skipped} malformed lines.", file=sys.stderr)

    return cases


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------
def build_summary_text(results: Dict[str, Any], input_path: str, elapsed: float) -> str:
    total   = results["total_cases"]
    clean   = results["clean_cases"]
    pct     = results["clean_cases_pct"]
    crit_n  = results["total_critical_anomalies"]
    flagged = results["cases_with_critical"]
    cats    = results["category_counts"]

    lines = [
        "=" * 72,
        "CRITICAL ERROR FLAGGING REPORT  —  PM4Lawmaking 2.0",
        "=" * 72,
        f"Run timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Elapsed         : {elapsed:.1f}s",
        f"Input file      : {os.path.basename(input_path)}",
        "-" * 72,
        f"Total cases     : {total:>6}",
        f"Clean cases     : {clean:>6}  ({pct:.2f}%)",
        f"Flagged cases   : {flagged:>6}  ({100 - pct:.2f}%)",
        f"Total CRITICAL  : {crit_n:>6}",
        "-" * 72,
        "CRITICAL anomalies by category:",
    ]
    if not cats:
        lines.append("  None detected — dataset is fully compliant with all CRITICAL rules.")
    else:
        max_cnt = max(cats.values())
        bar_scale = 40 / max_cnt if max_cnt > 0 else 1
        for cat, cnt in sorted(cats.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(cnt * bar_scale)
            lines.append(f"  {cat:<36} {cnt:>5}  {bar}")

    lines.append("=" * 72)
    return "\n".join(lines)


def save_flagged_cases_jsonl(results: Dict[str, Any], path: str) -> None:
    """
    Writes one JSON line per flagged case:
      { case_id, anomaly_count, anomalies: [...] }
    Useful for downstream processing in the target repo.
    """
    cases_summary = results.get("cases_summary", {})
    with open(path, "w", encoding="utf-8") as f:
        for case_id, anomalies in cases_summary.items():
            record = {
                "case_id": case_id,
                "anomaly_count": len(anomalies),
                "anomalies": anomalies,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flag CRITICAL errors in the full PM4Lawmaking extraction dataset."
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
        help=f"Path to the merged JSONL extraction file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where output files are saved. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--no-jsonl",
        action="store_true",
        help="Skip writing flagged_cases.jsonl (only write full JSON + summary).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------------
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"  PM4Lawmaking 2.0  —  CRITICAL Error Flagging Pipeline")
    print(f"{'='*72}")
    print(f"  Input  : {args.input}")
    print(f"  Output : {args.output}")
    print()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("  [1/3] Loading extraction dataset...", end=" ", flush=True)
    t0 = time.time()
    cases_data = load_extraction_jsonl(args.input)
    print(f"done — {len(cases_data):,} cases loaded  ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # Run CRITICAL checks
    # ------------------------------------------------------------------
    print("  [2/3] Running CRITICAL error checks...", end=" ", flush=True)
    t1 = time.time()
    results = flag_dataset(cases_data)
    elapsed = time.time() - t1
    print(f"done  ({elapsed:.1f}s)")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    print("  [3/3] Saving results...")

    # Full JSON
    json_out = os.path.join(args.output, "flagging_results.json")
    results["run_metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "input_file": os.path.abspath(args.input),
        "elapsed_seconds": round(elapsed, 2),
    }
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"        flagging_results.json  →  {json_out}")

    # Flagged cases JSONL
    if not args.no_jsonl:
        jsonl_out = os.path.join(args.output, "flagged_cases.jsonl")
        save_flagged_cases_jsonl(results, jsonl_out)
        print(f"        flagged_cases.jsonl    →  {jsonl_out}")

    # Human-readable summary
    total_elapsed = time.time() - t0
    summary_text = build_summary_text(results, args.input, total_elapsed)
    summary_out = os.path.join(args.output, "flagging_summary.txt")
    with open(summary_out, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"        flagging_summary.txt   →  {summary_out}")

    # Print summary to stdout
    print()
    print(summary_text)


if __name__ == "__main__":
    main()
