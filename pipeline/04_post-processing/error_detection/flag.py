#!/usr/bin/env python3
"""
flag.py - Detection of CRITICAL errors in legislative event logs (PM4Lawmaking 2.0).

Standalone module — zero external dependencies (only Python stdlib).
Collects ONLY checks that identify CRITICAL anomalies (strictly excluding
WARNING and INFO levels).

Critical Checks included:
1. check_implausible_date:
   - Malformed date format (expected YYYY-MM-DD).
   - Date outside historical republican plausible range (1948 - 2027).
2. check_sequence_date_order:
   - The ordering of 'sequence_id' must correspond to that of 'date'.
   - Flagged when an event with higher sequence_id has an earlier date.
   - Duplicate sequence_ids are silently ignored.
3. check_critical_chronology_inversion:
   - Temporal inversion involving critical procedural activities
     (e.g., presentazione/assegnazione occurring after other events, or approvazione/esame_aula
     preceding subsequent events).
4. check_unauthorized_approval:
   - 'approvazione' must be preceded by prior examination ('esame_commissione' or 'esame_aula')
     in that chamber for that act.

Usage:
  # Run on an extraction file:
  $ python flag.py --input ../output/extraction_few-shot_merged_full.jsonl
"""

import os
import sys
import re
import json
import argparse
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Optional

# ------------------------------------------------------------------------------
# Taxonomy and Data Structures (inline — no external dependencies)
# ------------------------------------------------------------------------------
class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class ErrorCategory(str, Enum):
    CHRONOLOGY_INVERSION = "CHRONOLOGY_INVERSION"
    SEQUENCE_DATE_MISMATCH = "SEQUENCE_DATE_MISMATCH"
    IMPLAUSIBLE_DATE = "IMPLAUSIBLE_DATE"
    SEDE_INCOMPATIBILITY = "SEDE_INCOMPATIBILITY"
    UNAUTHORIZED_APPROVAL = "UNAUTHORIZED_APPROVAL"
    LLM_EVENT_OMISSION = "LLM_EVENT_OMISSION"
    LLM_EVENT_HALLUCINATION = "LLM_EVENT_HALLUCINATION"


@dataclass
class DetectedAnomaly:
    case_id: str
    category: ErrorCategory
    severity: Severity
    message: str
    sequence_id: Optional[int] = None
    event_activity: Optional[str] = None
    event_date: Optional[str] = None
    chamber: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["category"] = self.category.value if hasattr(self.category, "value") else str(self.category)
        data["severity"] = self.severity.value if hasattr(self.severity, "value") else str(self.severity)
        return data


# Historical republican plausibility bounds (Article 70-74 Cost.)
MIN_PLAUSIBLE_YEAR = 1948
MAX_PLAUSIBLE_YEAR = 2027

DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ------------------------------------------------------------------------------
# Helper Parsing Functions
# ------------------------------------------------------------------------------
def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """
    Parses a date string in YYYY-MM-DD format.
    Returns datetime object or None if invalid.
    """
    if not date_str or not isinstance(date_str, str):
        return None
    clean = date_str.strip()
    if not DATE_REGEX.match(clean):
        return None
    try:
        return datetime.strptime(clean, "%Y-%m-%d")
    except ValueError:
        return None


def parse_sequence_id(val: Any, fallback: int) -> int:
    """
    Safely converts a sequence_id value to an integer, falling back to a default value.
    """
    if val is None:
        return fallback
    try:
        return int(val)
    except (ValueError, TypeError):
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return fallback


# ------------------------------------------------------------------------------
# CRITICAL Check 1: Date Format & Plausibility
# ------------------------------------------------------------------------------
def check_implausible_date(
    events: List[Dict[str, Any]],
    case_id: str = "",
    min_year: int = MIN_PLAUSIBLE_YEAR,
    max_year: int = MAX_PLAUSIBLE_YEAR,
) -> List[DetectedAnomaly]:
    """
    Identifies CRITICAL date anomalies:
    1. Invalid date format (not YYYY-MM-DD or invalid calendar date).
    2. Date outside plausible historical range (1948 - 2027).
    Note: events with date=None are considered WARNINGs in 04_error_detection and are excluded.
    """
    anomalies: List[DetectedAnomaly] = []
    for idx, ev in enumerate(events, start=1):
        seq_id = parse_sequence_id(ev.get("sequence_id"), idx)
        act = str(ev.get("activity", "")).strip()
        raw_date = ev.get("date")
        chamber = ev.get("chamber")

        if raw_date is None:
            continue

        raw_date_str = str(raw_date).strip()
        dt = parse_date(raw_date_str)
        if dt is None:
            anomalies.append(
                DetectedAnomaly(
                    case_id=case_id,
                    category=ErrorCategory.IMPLAUSIBLE_DATE,
                    severity=Severity.CRITICAL,
                    message=f"Event #{seq_id} ('{act}'): invalid date format '{raw_date}' (expected YYYY-MM-DD).",
                    sequence_id=seq_id,
                    event_activity=act,
                    event_date=raw_date_str,
                    chamber=chamber,
                    details={"raw_date": raw_date},
                )
            )
            continue

        if dt.year < min_year or dt.year > max_year:
            anomalies.append(
                DetectedAnomaly(
                    case_id=case_id,
                    category=ErrorCategory.IMPLAUSIBLE_DATE,
                    severity=Severity.CRITICAL,
                    message=(
                        f"Event #{seq_id} ('{act}'): date outside plausible range ({dt.strftime('%Y-%m-%d')}). "
                        f"Expected year between {min_year} and {max_year}."
                    ),
                    sequence_id=seq_id,
                    event_activity=act,
                    event_date=raw_date_str,
                    chamber=chamber,
                    details={"parsed_year": dt.year, "min_year": min_year, "max_year": max_year},
                )
            )

    return anomalies


# Alias
check_date_validity = check_implausible_date


# ------------------------------------------------------------------------------
# CRITICAL Check 2: Sequence ID vs Date Monotonicity
# ------------------------------------------------------------------------------
def check_sequence_date_order(
    events: List[Dict[str, Any]],
    case_id: str = "",
) -> List[DetectedAnomaly]:
    """
    CRITICAL CHECK:
    Verifies that the ordering defined by 'sequence_id' corresponds to the chronological
    order of 'date'.

    Formal Rule:
    For any two events with valid dates:
    If sequence_id_A < sequence_id_B, then date_A <= date_B.
    A CRITICAL anomaly is flagged when an event with a higher sequence_id has a date
    strictly earlier than an event with a lower sequence_id.
    Duplicate sequence_ids are ignored (not flagged).
    """
    anomalies: List[DetectedAnomaly] = []
    if not events:
        return anomalies

    # Extract all events having a parseable date
    dated_events: List[Dict[str, Any]] = []
    for idx, ev in enumerate(events, start=1):
        seq_id = parse_sequence_id(ev.get("sequence_id"), idx)
        raw_date = ev.get("date")
        dt = parse_date(str(raw_date)) if raw_date is not None else None
        if dt is not None:
            dated_events.append({
                "original_idx": idx,
                "sequence_id": seq_id,
                "dt": dt,
                "date_str": str(raw_date).strip(),
                "activity": str(ev.get("activity", "")).strip(),
                "chamber": ev.get("chamber"),
                "raw_event": ev,
            })

    if len(dated_events) < 2:
        return anomalies

    # Sort events by declared sequence_id (using original position as deterministic tie-breaker)
    dated_events.sort(key=lambda x: (x["sequence_id"], x["original_idx"]))

    # Scan through sequence-ordered events maintaining the latest date seen so far
    max_seen_item = dated_events[0]

    for i in range(1, len(dated_events)):
        curr_item = dated_events[i]
        curr_seq = curr_item["sequence_id"]
        curr_dt = curr_item["dt"]
        curr_date_str = curr_item["date_str"]
        curr_act = curr_item["activity"]
        curr_chamber = curr_item["chamber"]

        prev_item = dated_events[i - 1]
        prev_seq = prev_item["sequence_id"]
        prev_dt = prev_item["dt"]
        prev_date_str = prev_item["date_str"]
        prev_act = prev_item["activity"]

        # Skip duplicate sequence_ids — not flagged
        if curr_seq == prev_seq:
            continue

        # sequence_id increased but date decreased → CRITICAL
        if curr_seq > prev_seq:
            ref_item = None
            if curr_dt < prev_dt:
                ref_item = prev_item
            elif curr_dt < max_seen_item["dt"] and curr_seq > max_seen_item["sequence_id"]:
                ref_item = max_seen_item

            if ref_item is not None:
                ref_seq = ref_item["sequence_id"]
                ref_dt = ref_item["dt"]
                ref_date_str = ref_item["date_str"]
                ref_act = ref_item["activity"]
                days_diff = (ref_dt - curr_dt).days

                anomalies.append(
                    DetectedAnomaly(
                        case_id=case_id,
                        category=ErrorCategory.SEQUENCE_DATE_MISMATCH,
                        severity=Severity.CRITICAL,
                        message=(
                            f"Sequence ID / date mismatch: Event #{curr_seq} ('{curr_act}', date: {curr_date_str}) "
                            f"has sequence_id {curr_seq} > {ref_seq}, but its date is {days_diff} days earlier "
                            f"than Event #{ref_seq} ('{ref_act}', date: {ref_date_str})."
                        ),
                        sequence_id=curr_seq,
                        event_activity=curr_act,
                        event_date=curr_date_str,
                        chamber=curr_chamber,
                        details={
                            "type": "sequence_precedes_date",
                            "curr_sequence_id": curr_seq,
                            "curr_date": curr_date_str,
                            "curr_activity": curr_act,
                            "prev_sequence_id": ref_seq,
                            "prev_date": ref_date_str,
                            "prev_activity": ref_act,
                            "days_diff": days_diff,
                        },
                    )
                )

        if curr_dt > max_seen_item["dt"]:
            max_seen_item = curr_item

    return anomalies


# Alias
check_sequence_date_mismatch = check_sequence_date_order


# ------------------------------------------------------------------------------
# CRITICAL Check 3: Chronology Inversion involving Key Procedural Milestones
# ------------------------------------------------------------------------------
def check_critical_chronology_inversion(
    events: List[Dict[str, Any]],
    case_id: str = "",
) -> List[DetectedAnomaly]:
    """
    CRITICAL CHECK from chronological_rules.py:
    Identifies temporal inversions that involve critical activities:
    - curr_act in ('presentazione', 'assegnazione')
    - prev_act in ('approvazione', 'esame_aula')
    (General non-critical inversions are WARNINGs in 04_error_detection and are excluded).
    """
    anomalies: List[DetectedAnomaly] = []
    if not events:
        return anomalies

    last_valid_dt: Optional[datetime] = None
    last_valid_ev: Optional[Dict[str, Any]] = None

    for idx, ev in enumerate(events, start=1):
        raw_date = ev.get("date")
        if raw_date is None:
            continue
        curr_dt = parse_date(str(raw_date))
        if curr_dt is None:
            continue

        curr_seq = parse_sequence_id(ev.get("sequence_id"), idx)
        curr_act = str(ev.get("activity", "")).strip()
        curr_chamber = ev.get("chamber")
        curr_date_str = str(raw_date).strip()

        if last_valid_dt is not None and last_valid_ev is not None:
            if curr_dt < last_valid_dt:
                prev_seq = parse_sequence_id(last_valid_ev.get("sequence_id"), idx - 1)
                prev_act = str(last_valid_ev.get("activity", "")).strip()
                prev_date_str = str(last_valid_ev.get("date", "")).strip()

                is_critical = (
                    curr_act in ("presentazione", "assegnazione")
                    or prev_act in ("approvazione", "esame_aula")
                )
                if is_critical:
                    days_diff = (last_valid_dt - curr_dt).days
                    anomalies.append(
                        DetectedAnomaly(
                            case_id=case_id,
                            category=ErrorCategory.CHRONOLOGY_INVERSION,
                            severity=Severity.CRITICAL,
                            message=(
                                f"Temporal inversion of {days_diff} days: Event #{curr_seq} ('{curr_act}', {curr_date_str}) "
                                f"precedes previous Event #{prev_seq} ('{prev_act}', {prev_date_str})."
                            ),
                            sequence_id=curr_seq,
                            event_activity=curr_act,
                            event_date=curr_date_str,
                            chamber=curr_chamber,
                            details={
                                "prev_sequence_id": prev_seq,
                                "prev_activity": prev_act,
                                "prev_date": prev_date_str,
                                "curr_date": curr_date_str,
                                "days_diff": days_diff,
                            },
                        )
                    )

        last_valid_dt = curr_dt
        last_valid_ev = ev

    return anomalies


# ------------------------------------------------------------------------------
# CRITICAL Check 4: Unauthorized Approval without Prior Examination
# ------------------------------------------------------------------------------
def check_unauthorized_approval(
    events: List[Dict[str, Any]],
    case_id: str = "",
) -> List[DetectedAnomaly]:
    """
    CRITICAL CHECK from lifecycle_rules.py:
    Approval ('approvazione') must be preceded by at least one examination
    (either committee examination 'esame_commissione' or floor debate 'esame_aula')
    recorded for that bill in that parliamentary chamber.
    """
    anomalies: List[DetectedAnomaly] = []
    if not events:
        return anomalies

    # Track state per (numero_atto, chamber)
    act_chamber_states: Dict[tuple, Dict[str, bool]] = {}

    def get_state(key: tuple) -> Dict[str, bool]:
        if key not in act_chamber_states:
            act_chamber_states[key] = {
                "examined_comm": False,
                "examined_aula": False,
            }
        return act_chamber_states[key]

    for idx, ev in enumerate(events, start=1):
        seq_id = parse_sequence_id(ev.get("sequence_id"), idx)
        act = str(ev.get("activity", "")).strip()
        raw_chamber = ev.get("chamber")
        chamber = str(raw_chamber).strip().lower() if raw_chamber is not None else "unknown"
        num_atto = str(ev.get("numero_atto", "")).strip()

        state = get_state((num_atto, chamber))

        if act == "esame_commissione":
            state["examined_comm"] = True

        elif act == "esame_aula":
            state["examined_aula"] = True

        elif act == "approvazione":
            if not state["examined_comm"] and not state["examined_aula"]:
                chamber_display = chamber.capitalize() if chamber != "unknown" else "unknown chamber"
                anomalies.append(
                    DetectedAnomaly(
                        case_id=case_id,
                        category=ErrorCategory.UNAUTHORIZED_APPROVAL,
                        severity=Severity.CRITICAL,
                        message=(
                            f"Event #{seq_id} ('approvazione' at {chamber_display}): "
                            f"occurred without any prior examination (neither in committee nor on the floor)."
                        ),
                        sequence_id=seq_id,
                        event_activity=act,
                        event_date=ev.get("date"),
                        chamber=chamber,
                        details={"numero_atto": num_atto, "chamber": chamber},
                    )
                )

    return anomalies


# ------------------------------------------------------------------------------
# Master Orchestration Functions
# ------------------------------------------------------------------------------
def flag_case(
    events: List[Dict[str, Any]],
    case_id: str = "",
) -> List[DetectedAnomaly]:
    """
    Executes all CRITICAL error detection checks on a single case (legislative process):
    1. check_implausible_date
    2. check_sequence_date_order
    3. check_critical_chronology_inversion
    4. check_unauthorized_approval

    Returns:
        List of DetectedAnomaly containing exclusively CRITICAL errors.
    """
    anomalies: List[DetectedAnomaly] = []

    anomalies.extend(check_implausible_date(events, case_id))
    anomalies.extend(check_sequence_date_order(events, case_id))
    anomalies.extend(check_critical_chronology_inversion(events, case_id))
    anomalies.extend(check_unauthorized_approval(events, case_id))

    return anomalies


def flag_dataset(
    cases_data: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Executes all CRITICAL error detection checks on a dictionary of cases {case_id: events}.

    Returns:
        A dictionary with summary statistics and the list of all detected CRITICAL anomalies.
    """
    all_anomalies: List[DetectedAnomaly] = []
    cases_summary: Dict[str, List[Dict[str, Any]]] = {}
    cases_with_critical = 0

    for case_id, events in cases_data.items():
        anomalies = flag_case(events, case_id=case_id)

        if anomalies:
            cases_with_critical += 1
            cases_summary[case_id] = [a.to_dict() for a in anomalies]
            all_anomalies.extend(anomalies)

    category_counts: Dict[str, int] = {}
    for a in all_anomalies:
        cat = a.category.value if hasattr(a.category, "value") else str(a.category)
        category_counts[cat] = category_counts.get(cat, 0) + 1

    total_cases = len(cases_data)
    clean_cases = total_cases - cases_with_critical
    clean_pct = round((clean_cases / total_cases * 100), 2) if total_cases > 0 else 100.0

    return {
        "total_cases": total_cases,
        "clean_cases": clean_cases,
        "clean_cases_pct": clean_pct,
        "cases_with_critical": cases_with_critical,
        "total_critical_anomalies": len(all_anomalies),
        "category_counts": category_counts,
        "anomalies": [a.to_dict() for a in all_anomalies],
        "cases_summary": cases_summary,
    }


def flag_file(
    input_path: str
) -> Dict[str, Any]:
    """
    Loads an extraction file (.jsonl or .json) and executes all CRITICAL error checks.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    cases_data: Dict[str, List[Dict[str, Any]]] = {}
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Try JSON array / dict first
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            for idx, item in enumerate(parsed):
                c_id = str(item.get("id") or item.get("case_id") or f"case_{idx+1}")
                evs = item.get("events") or item.get("eventi") or []
                cases_data[c_id] = evs
        elif isinstance(parsed, dict) and "casi" in parsed:
            for item in parsed["casi"]:
                c_id = str(item.get("case_id") or item.get("id"))
                evs = item.get("eventi") or item.get("events") or []
                cases_data[c_id] = evs
    except Exception:
        pass

    # JSONL fallback
    if not cases_data:
        for line_idx, line in enumerate(content.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line.strip())
                c_id = str(rec.get("id") or rec.get("case_id", f"case_{line_idx}"))
                evs = []
                out = rec.get("output")
                if isinstance(out, dict):
                    evs = out.get("events") or out.get("eventi") or []
                elif isinstance(out, list):
                    evs = out
                elif "events" in rec:
                    evs = rec["events"]
                elif "eventi" in rec:
                    evs = rec["eventi"]
                cases_data[c_id] = evs
            except Exception:
                continue


    results = flag_dataset(cases_data)
    results["input_file"] = os.path.basename(input_path)
    return results

# ------------------------------------------------------------------------------
# CLI Interface
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="flag.py - Detection of CRITICAL errors in legislative event logs (PM4Lawmaking 2.0)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to JSONL or JSON extraction file to evaluate."
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Optional path to save detected CRITICAL anomalies (JSON format)."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run unit test suite verifying all critical error detection rules."
    )

    args = parser.parse_args()


    if not args.input:
        parser.print_help()
        sys.exit(1)

    print(f"\nEvaluating CRITICAL errors for: {args.input}")

    results = flag_file(args.input)

    total_cases = results["total_cases"]
    clean_cases = results["clean_cases"]
    clean_pct = results["clean_cases_pct"]
    crit_count = results["total_critical_anomalies"]
    cat_counts = results["category_counts"]

    print("\n" + "=" * 70)
    print("CRITICAL ERROR DETECTION REPORT (Severity: CRITICAL only)")
    print("=" * 70)
    print(f"Analyzed File:               {results.get('input_file', args.input)}")
    print(f"Total Cases:                 {total_cases}")
    print(f"Clean Cases (0 critical):    {clean_cases} ({clean_pct}%)")
    print(f"Cases with Critical Errors:  {results['cases_with_critical']}")
    print(f"Total Critical Anomalies:    {crit_count}")
    print("-" * 70)
    print("CRITICAL Anomalies by Category:")
    if not cat_counts:
        print("  None detected (100% compliant with CRITICAL rules).")
    else:
        for cat, cnt in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {cat:<32}: {cnt}")
    print("=" * 70)

    # Print first 10 critical anomalies as sample
    if results["anomalies"]:
        sample_count = min(10, len(results["anomalies"]))
        print(f"\nSample of first {sample_count} detected CRITICAL anomalies:")
        for idx, a in enumerate(results["anomalies"][:sample_count], start=1):
            print(f"  [{idx}] Case {a['case_id']} | Category: {a['category']} | Seq: #{a['sequence_id']}")
            print(f"      Message: {a['message']}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved structured results to: {args.output}")


if __name__ == "__main__":
    main()
