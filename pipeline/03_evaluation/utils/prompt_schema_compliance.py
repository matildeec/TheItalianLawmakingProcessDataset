"""
Verify structural and domain adherence of extracted events to the prompt schema.
"""

import re
import os
import sys

# Centralized schema constraints from config_eval.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_eval import (
    REQUIRED_SCHEMA,
    DATE_REGEX_PATTERN,
    ALLOWED_ACTIVITIES,
    ALLOWED_SEDI,
    ALLOWED_CHAMBERS,
)

DATE_REGEX = re.compile(DATE_REGEX_PATTERN)


def verify_event_schema(event: dict) -> tuple[bool, list[str]]:
    """
    Verify whether a single event strictly complies with required schema properties and prompt constraints.
    Returns (is_valid, list_of_errors).
    """
    errors = []

    if not isinstance(event, dict):
        return False, ["Event is not a valid JSON object (dict)"]

    # 1. Verify presence and type for each required property
    for key, expected_type in REQUIRED_SCHEMA.items():
        if key not in event:
            errors.append(f"Missing required key: '{key}'")
            continue

        val = event[key]

        # In Python bool is a subclass of int (isinstance(True, int) -> True), handle explicitly
        if expected_type is int and (isinstance(val, bool) or not isinstance(val, int)):
            errors.append(f"Incorrect type for '{key}': expected int, got {type(val).__name__}")
        elif expected_type is bool and not isinstance(val, bool):
            errors.append(f"Incorrect type for '{key}': expected bool, got {type(val).__name__}")
        elif isinstance(expected_type, tuple) and not isinstance(val, expected_type):
            exp_name = " | ".join([t.__name__ for t in expected_type])
            errors.append(f"Incorrect type for '{key}': expected {exp_name}, got {type(val).__name__}")
        elif isinstance(expected_type, type) and expected_type not in (int, bool) and not isinstance(val, expected_type):
            errors.append(f"Incorrect type for '{key}': expected {expected_type.__name__}, got {type(val).__name__}")

    # 2. Check for unexpected extra keys
    allowed_keys = set(REQUIRED_SCHEMA.keys())
    extra_keys = set(event.keys()) - allowed_keys
    if extra_keys:
        errors.append(f"Unrecognized extra keys not defined in prompt schema: {sorted(list(extra_keys))}")

    if errors:
        return False, errors

    # 3. Domain value constraints (Prompt Rules)
    if event["sequence_id"] <= 0:
        errors.append(f"Invalid sequence_id ({event['sequence_id']}): must be an integer > 0")

    if not str(event["numero_atto"]).strip():
        errors.append("numero_atto is empty")

    act = event["activity"]
    if act not in ALLOWED_ACTIVITIES:
        errors.append(f"Activity '{act}' not allowed (permitted: {sorted(list(ALLOWED_ACTIVITIES))})")

    if event["date"] is not None:
        if not isinstance(event["date"], str) or not DATE_REGEX.match(event["date"]):
            errors.append(f"Invalid date format '{event['date']}': expected YYYY-MM-DD or null")

    # Commissions: list of strings, each strictly in uppercase
    if not isinstance(event["commissions"], list) or not all(isinstance(c, str) for c in event["commissions"]):
        errors.append("commissions must be a list of strings")
    else:
        for comm in event["commissions"]:
            if comm and comm != comm.upper():
                errors.append(f"Commission '{comm}' must be strictly in uppercase")

    # Procedural mode (sede): permitted value and rule constraints
    sede = event["sede"]
    if sede not in ALLOWED_SEDI:
        errors.append(f"Procedural mode (sede) '{sede}' not allowed")
    else:
        # Prompt constraints:
        # - richiesta_parere -> sede MUST be "consultiva"
        # - presentazione, esame_aula, relazione -> sede MUST be null
        if act == "richiesta_parere" and sede != "consultiva":
            errors.append(f"Prompt constraint: for activity 'richiesta_parere' procedural mode must be 'consultiva', got '{sede}'")
        elif act in ("presentazione", "esame_aula", "relazione") and sede is not None:
            errors.append(f"Prompt constraint: for activity '{act}' procedural mode must be null, got '{sede}'")

    # Persons: list of strings in uppercase, without "ed altri"
    if not isinstance(event["persons"], list) or not all(isinstance(p, str) for p in event["persons"]):
        errors.append("persons must be a list of strings")
    else:
        for person in event["persons"]:
            if person:
                if person != person.upper():
                    errors.append(f"Person '{person}' must be strictly in uppercase")
                if "ed altri" in person.lower():
                    errors.append(f"Person '{person}' contains disallowed 'ed altri' suffix")

    if not isinstance(event["atti_unificati"], list) or not all(isinstance(a, str) for a in event["atti_unificati"]):
        errors.append("atti_unificati must be a list of strings")

    if event["chamber"] not in ALLOWED_CHAMBERS:
        errors.append(f"Chamber '{event['chamber']}' not allowed")

    is_valid = len(errors) == 0
    return is_valid, errors


def verify_prompt_schema_compliance(records: list[dict]) -> dict:
    """
    Verify prompt schema compliance across all extracted events produced by a model.
    Returns aggregate compliance metrics and a list of invalid cases with specific errors.
    """
    tot_events_extracted = 0
    valid_structural_events = 0
    attribute_error_counts = {}
    invalid_cases = {}

    for rec in records:
        c_id = str(rec.get("id") or rec.get("case_id") or "UNKNOWN")
        out = rec.get("output", {})
        events = out.get("events")
        if events is None:
            events = out.get("eventi", [])

        case_has_errors = False
        case_errors = []

        if isinstance(events, list):
            tot_events_extracted += len(events)
            for idx, ev in enumerate(events, 1):
                is_valid, errs = verify_event_schema(ev)
                if is_valid:
                    valid_structural_events += 1
                else:
                    case_has_errors = True
                    for err in errs:
                        formatted_err = f"Event #{idx} (seq_id: {ev.get('sequence_id', 'N/A')}): {err}"
                        case_errors.append(formatted_err)

                        attr_key = "other_errors"
                        for key in REQUIRED_SCHEMA.keys():
                            if f"'{key}'" in err or f" {key} " in err:
                                attr_key = key
                                break
                        if "Missing required key" in err:
                            attr_key = "missing_keys"
                        elif "Unrecognized extra keys" in err:
                            attr_key = "extra_keys"

                        attribute_error_counts[attr_key] = attribute_error_counts.get(attr_key, 0) + 1

        if case_has_errors:
            invalid_cases[c_id] = {
                "error_count": len(case_errors),
                "errors": case_errors
            }

    schema_adherence_rate = (
        (valid_structural_events / tot_events_extracted * 100)
        if tot_events_extracted > 0
        else 100.0
    )

    return {
        "total_events_extracted": tot_events_extracted,
        "schema_valid_events": valid_structural_events,
        "schema_adherence_rate_pct": round(schema_adherence_rate, 2),
        "attribute_error_breakdown": attribute_error_counts,
        "invalid_cases": invalid_cases,
    }

