"""
Perform optimal bipartite matching (Hungarian Algorithm) between model-extracted events
and Gold Standard events, computing Micro and Macro Precision, Recall, and F1-Scores.
"""

import re
import os
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_eval import HUNGARIAN_SIMILARITY_THRESHOLD

try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


def event_to_text(ev: dict) -> str:
    """
    Convert a structured event into a textual description for semantic similarity comparison.
    """
    if not isinstance(ev, dict):
        return ""

    activity = ev.get("activity") or "sconosciuta"
    date = ev.get("date") or "non specificata"
    chamber = ev.get("chamber") or "non specificata"
    commissions = ", ".join(ev.get("commissions") or []) or "nessuna"
    persons = ", ".join(ev.get("persons") or []) or "nessuno"
    sede = ev.get("sede") or "non specificata"

    parts = [
        f"Attività {activity}",
        f"Data {date}",
        f"Camera {chamber}",
        f"Commissioni {commissions}",
        f"Persone {persons}",
        f"Sede {sede}",
    ]
    if ev.get("co_firmatari"):
        parts.append("con co-firmatari")
    if ev.get("con_modificazioni"):
        parts.append("approvato con modificazioni")
    if ev.get("atti_unificati"):
        parts.append(f"atti unificati: {', '.join(ev.get('atti_unificati'))}")

    return ". ".join(parts)


def get_token_jaccard_similarity(str1: str, str2: str) -> float:
    """
    Fallback token Jaccard similarity between two strings.
    """
    words1 = set(re.findall(r"\w+", str1.lower()))
    words2 = set(re.findall(r"\w+", str2.lower()))
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    return len(words1.intersection(words2)) / len(words1.union(words2))


def verify_hungarian_matching(
    records: list[dict], gold_data: dict, threshold: float = HUNGARIAN_SIMILARITY_THRESHOLD, embedder=None
) -> dict:
    """
    Perform bipartite matching using the Hungarian Algorithm between predicted events and the Gold Standard.
    Computes both Micro-averaged (global) and Macro-averaged (per-case mean) metrics.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_predicted = 0
    total_gold = 0

    case_precisions = []
    case_recalls = []
    case_f1s = []

    perfect_semantic_cases_count = 0

    for rec in records:
        case_id = str(rec.get("id") or rec.get("case_id"))
        extracted_raw = rec.get("output", {}).get("events")
        if extracted_raw is None:
            extracted_raw = rec.get("output", {}).get("eventi", [])

        gold_raw = gold_data.get(case_id, [])

        pred_events = [ev for ev in extracted_raw if isinstance(ev, dict)]
        gold_events = [ev for ev in gold_raw if isinstance(ev, dict)]

        total_predicted += len(pred_events)
        total_gold += len(gold_events)

        if not pred_events and not gold_events:
            case_precisions.append(1.0)
            case_recalls.append(1.0)
            case_f1s.append(1.0)
            perfect_semantic_cases_count += 1
            continue

        if not pred_events and gold_events:
            total_fn += len(gold_events)
            case_precisions.append(0.0)
            case_recalls.append(0.0)
            case_f1s.append(0.0)
            continue
            
        if pred_events and not gold_events:
            total_fp += len(pred_events)
            case_precisions.append(0.0)
            case_recalls.append(0.0)
            case_f1s.append(0.0)
            continue

        pred_texts = [event_to_text(ev) for ev in pred_events]
        gold_texts = [event_to_text(ev) for ev in gold_events]

        sim_matrix = np.zeros((len(pred_events), len(gold_events)))

        if embedder is not None and HAS_SENTENCE_TRANSFORMERS:
            p_embs = embedder.encode(pred_texts, convert_to_tensor=True)
            g_embs = embedder.encode(gold_texts, convert_to_tensor=True)
            sim_matrix = util.cos_sim(p_embs, g_embs).cpu().numpy()
        else:
            for i, p_t in enumerate(pred_texts):
                for j, g_t in enumerate(gold_texts):
                    sim_matrix[i, j] = get_token_jaccard_similarity(p_t, g_t)

        cost_matrix = -sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        case_tp = 0
        for r, c in zip(row_ind, col_ind):
            if sim_matrix[r, c] >= threshold:
                case_tp += 1

        case_fp = len(pred_events) - case_tp
        case_fn = len(gold_events) - case_tp

        total_tp += case_tp
        total_fp += case_fp
        total_fn += case_fn

        c_prec = case_tp / (case_tp + case_fp) if (case_tp + case_fp) > 0 else 0.0
        c_rec = case_tp / (case_tp + case_fn) if (case_tp + case_fn) > 0 else 0.0
        c_f1 = (2 * c_prec * c_rec) / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0.0

        case_precisions.append(c_prec)
        case_recalls.append(c_rec)
        case_f1s.append(c_f1)

        if c_f1 == 1.0:
            perfect_semantic_cases_count += 1

    tot_cases = len(case_f1s)
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    macro_prec = float(np.mean(case_precisions)) if tot_cases > 0 else 0.0
    macro_rec = float(np.mean(case_recalls)) if tot_cases > 0 else 0.0
    macro_f1 = float(np.mean(case_f1s)) if tot_cases > 0 else 0.0

    return {
        "micro_precision": round(micro_prec, 4),
        "micro_recall": round(micro_rec, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "perfect_semantic_cases_count": perfect_semantic_cases_count,
        "perfect_semantic_cases_pct": round((perfect_semantic_cases_count / tot_cases * 100), 2) if tot_cases > 0 else 0.0,
        "similarity_threshold": threshold,
        "total_matched_events": total_tp,
        "total_predicted_events": total_predicted,
        "total_gold_events": total_gold,
        # Standard metric aliases
        "precision": round(micro_prec, 4),
        "recall": round(micro_rec, 4),
        "f1_score": round(micro_f1, 4),
    }

