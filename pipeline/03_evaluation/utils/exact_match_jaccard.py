"""
Compute Jaccard similarity across extracted event sets against the Gold Standard for each case_id (Event-Set Exact Jaccard).
An event is considered an Exact Match if all its properties match identically.
Provides attribute-level exact accuracy breakdown and counts of 100% Perfect, Partial, and Critical cases.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

DOMAIN_ATTRIBUTES = [
    "activity",
    "date",
    "commissions",
    "sede",
    "persons",
    "co_firmatari",
    "con_modificazioni",
    "atti_unificati",
    "chamber",
]


def canonicalize_event(event: dict, include_sequence_id: bool = False) -> tuple:
    """
    Convert an event dictionary into an immutable canonical tuple for exact match comparison.
    If include_sequence_id is False (default), the ordinal sequence is excluded from exact match comparison.
    """
    if not isinstance(event, dict):
        return ()

    activity = str(event.get("activity", "")).strip().lower()
    date = str(event.get("date")).strip() if event.get("date") is not None else None
    
    commissions = tuple(sorted([str(c).strip() for c in (event.get("commissions") or [])]))
    sede_val = event.get("procedural_mode") if "procedural_mode" in event else event.get("sede")
    sede = str(sede_val).strip().lower() if sede_val is not None else None
    persons_val = event.get("sponsors") or event.get("persons") or []
    persons = tuple(sorted([str(p).strip().upper() for p in persons_val]))
    
    co_firmatari = bool(event.get("has_co_sponsors") if "has_co_sponsors" in event else event.get("co_firmatari", False))
    con_modificazioni = bool(event.get("with_amendments") if "with_amendments" in event else event.get("con_modificazioni", False))
    unificati_val = event.get("unified_bills") or event.get("atti_unificati") or []
    atti_unificati = tuple(sorted([str(a).strip() for a in unificati_val]))
    chamber = str(event.get("chamber", "")).strip().lower()

    if include_sequence_id:
        seq_id = int(event.get("sequence_id", 0))
        return (
            seq_id,
            activity,
            date,
            commissions,
            sede,
            persons,
            co_firmatari,
            con_modificazioni,
            atti_unificati,
            chamber,
        )
    else:
        return (
            activity,
            date,
            commissions,
            sede,
            persons,
            co_firmatari,
            con_modificazioni,
            atti_unificati,
            chamber,
        )


def verify_exact_match_jaccard(records: list[dict], gold_data: dict, include_sequence_id: bool = False) -> dict:
    """
    Compute Event-Set Exact Match Jaccard for each case_id and the aggregate mean.
    Provides disaggregated accuracy for each domain attribute (Attribute-Level Breakdown).
    Categorizes cases into 100% Perfect (J=1.0), Partial (0.5 <= J < 1.0), and Critical (J < 0.5).
    """
    case_jaccard_scores = []
    tot_exact_matches = 0
    tot_pred_events = 0
    tot_gold_events = 0

    perfect_case_ids = []
    partial_case_details = []
    critical_case_details = []

    # Attribute accuracy matrix counters
    attr_matched_counts = {attr: 0 for attr in DOMAIN_ATTRIBUTES}
    attr_total_counts = {attr: 0 for attr in DOMAIN_ATTRIBUTES}

    for rec in records:
        case_id = str(rec.get("id") or rec.get("case_id"))
        
        extracted_raw = rec.get("output", {}).get("events")
        if extracted_raw is None:
            extracted_raw = rec.get("output", {}).get("eventi", [])

        gold_raw = gold_data.get(case_id, [])

        pred_tuples = [canonicalize_event(ev, include_sequence_id=include_sequence_id) for ev in extracted_raw if isinstance(ev, dict)]
        gold_tuples = [canonicalize_event(ev, include_sequence_id=include_sequence_id) for ev in gold_raw if isinstance(ev, dict)]

        pred_set = set(pred_tuples)
        gold_set = set(gold_tuples)

        tot_pred_events += len(pred_set)
        tot_gold_events += len(gold_set)

        intersection = pred_set.intersection(gold_set)
        union = pred_set.union(gold_set)

        matched_count = len(intersection)
        tot_exact_matches += matched_count

        if not pred_set and not gold_set:
            jaccard = 1.0
        elif not union:
            jaccard = 0.0
        else:
            jaccard = len(intersection) / len(union)

        case_jaccard_scores.append(jaccard)

        detail_info = {
            "case_id": case_id,
            "jaccard": round(float(jaccard), 4),
            "matched_events": matched_count,
            "pred_events": len(pred_set),
            "gold_events": len(gold_set)
        }

        if jaccard == 1.0:
            perfect_case_ids.append(case_id)
        elif jaccard >= 0.5:
            partial_case_details.append(detail_info)
        else:
            critical_case_details.append(detail_info)

        # --- Compute Attribute-Level Accuracy Breakdown ---
        if pred_tuples and gold_tuples:
            # Build attribute overlap matrix (0-10) between all predicted and gold events
            offset = 1 if include_sequence_id else 0
            num_attrs = len(DOMAIN_ATTRIBUTES)
            overlap_matrix = np.zeros((len(pred_tuples), len(gold_tuples)))

            for i, p_tup in enumerate(pred_tuples):
                for j, g_tup in enumerate(gold_tuples):
                    if len(p_tup) > offset and len(g_tup) > offset:
                        matches = sum(1 for k in range(num_attrs) if p_tup[offset + k] == g_tup[offset + k])
                        overlap_matrix[i, j] = matches

            # Bipartite matching to align events maximizing matching attributes
            row_ind, col_ind = linear_sum_assignment(-overlap_matrix)

            for r, c in zip(row_ind, col_ind):
                p_tup = pred_tuples[r]
                g_tup = gold_tuples[c]
                if len(p_tup) > offset and len(g_tup) > offset:
                    for k, attr_name in enumerate(DOMAIN_ATTRIBUTES):
                        attr_total_counts[attr_name] += 1
                        if p_tup[offset + k] == g_tup[offset + k]:
                            attr_matched_counts[attr_name] += 1

    tot_cases = len(case_jaccard_scores)
    mean_jaccard = float(np.mean(case_jaccard_scores)) if tot_cases > 0 else 0.0

    perfect_count = len(perfect_case_ids)
    partial_count = len(partial_case_details)
    critical_count = len(critical_case_details)

    # --- Compute Case Tiers for benchmark reporting ---
    tier1 = sum(1 for j in case_jaccard_scores if j == 1.0)
    tier2 = sum(1 for j in case_jaccard_scores if 0.80 <= j < 1.0)
    tier3 = sum(1 for j in case_jaccard_scores if 0.50 <= j < 0.80)
    tier4 = sum(1 for j in case_jaccard_scores if j < 0.50)

    # Compute accuracy percentage per attribute
    attribute_accuracy_pct = {}
    for attr in DOMAIN_ATTRIBUTES:
        tot = attr_total_counts[attr]
        matched = attr_matched_counts[attr]
        acc = (matched / tot * 100.0) if tot > 0 else 100.0
        attribute_accuracy_pct[attr] = {
            "accuracy_pct": round(acc, 2),
            "matched": matched,
            "total": tot,
        }

    return {
        "mean_event_jaccard_per_case": round(mean_jaccard, 4),
        "total_cases_evaluated": tot_cases,
        "perfect_cases_count": perfect_count,
        "perfect_cases_pct": round((perfect_count / tot_cases * 100), 2) if tot_cases > 0 else 0.0,
        "partial_cases_count": partial_count,
        "partial_cases_pct": round((partial_count / tot_cases * 100), 2) if tot_cases > 0 else 0.0,
        "critical_cases_count": critical_count,
        "critical_cases_pct": round((critical_count / tot_cases * 100), 2) if tot_cases > 0 else 0.0,
        "total_exact_matched_events": tot_exact_matches,
        "total_predicted_events": tot_pred_events,
        "total_gold_events": tot_gold_events,
        "attribute_accuracy": attribute_accuracy_pct,
        "perfect_case_ids": perfect_case_ids,
        "partial_case_details": partial_case_details,
        "critical_case_details": critical_case_details,
        "tier1_perfect_count": tier1,
        "tier2_high_conf_count": tier2,
        "tier3_medium_conf_count": tier3,
        "tier4_drop_count": tier4,
        "total_safe_cases_pct": round(((tier1 + tier2) / tot_cases * 100), 2) if tot_cases > 0 else 0.0
    }
