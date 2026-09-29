"""
Deterministic post-processing (single script).

Reads a JSONL file of LLM extractions, syntactically sanitizes every event and 
writes a new JSONL file using the canonical English schema.

Usage:
  python postprocess_results.py <input.jsonl> [output.jsonl]
If output is omitted: <input>_postprocessed.jsonl
"""

import argparse
import copy
import json
import os
import re

# ==============================================================================
# ITALIAN -> ENGLISH MAPPINGS
# ==============================================================================

FIELD_MAPPING_TO_ENGLISH = {
    "sequence_id": "sequence_id",
    "numero_atto": "bill_id",
    "activity": "activity",
    "date": "date",
    "commissions": "commissions",
    "sede": "procedural_mode",
    "persons": "sponsors",
    "co_firmatari": "has_co_sponsors",
    "con_modificazioni": "with_amendments",
    "atti_unificati": "unified_bills",
    "chamber": "chamber",
}
FIELD_MAPPING_TO_ITALIAN = {en: it for it, en in FIELD_MAPPING_TO_ENGLISH.items()}

ACTIVITY_TO_ENGLISH = {
    "presentazione": "presentation",
    "assegnazione": "assignment",
    "richiesta_parere": "opinion_request",
    "esame_commissione": "committee_examination",
    "esame_aula": "floor_examination",
    "approvazione": "approval",
    "relazione": "report",
    "stralcio": "severance",
}
ACTIVITY_TO_ITALIAN = {en: it for it, en in ACTIVITY_TO_ENGLISH.items()}

PROCEDURAL_MODE_TO_ENGLISH = {
    "referente": "referral",
    "legislativa": "legislative",
    "deliberante": "deliberative",
    "redigente": "drafting",
    "consultiva": "advisory",
}
PROCEDURAL_MODE_TO_ITALIAN = {en: it for it, en in PROCEDURAL_MODE_TO_ENGLISH.items()}


def _norm_str(v) -> str:
    return str(v).strip().lower() if v is not None else ""


# ==============================================================================
# PER-FIELD SANITIZERS
# ==============================================================================

ROMAN_TO_ARABIC = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
    "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15
}

TITLES_REGEX = re.compile(
    r'^(?:(?:sen|on)\.|\b(?:senatore|onorevole|ministro\s+della\s+\w+|ministro\s+degli\s+\w+|ministro\s+per\s+\w+|ministro|presidente\s+del\s+consiglio(?:\s+dei\s+ministri)?|presidente)\b)\s*',
    re.IGNORECASE
)


def clean_bill_id(ev: dict) -> str:
    """
    Strip superfluous prefixes from the bill number (e.g. "A.S.", "A.C.", "DDL", "d.d.l.")
    and normalize spaces around hyphens and suffixes (e.g. "2519- bis" -> "2519-bis", "1504/b" -> "1504/B").
    """
    num = str(ev.get("numero_atto") or "").strip()
    if not num:
        return num

    num_clean = re.sub(r'^(A\.S\.|A\.C\.|DDL|d\.d\.l\.|Atto|n\.)\s*', '', num, flags=re.IGNORECASE).strip()
    num_clean = re.sub(r'\s*-\s*', '-', num_clean)
    num_clean = re.sub(r'-\s*(bis|ter|quater)\b', r'-\1', num_clean, flags=re.IGNORECASE)
    num_clean = re.sub(r'([/-])([a-z])\b', lambda m: m.group(1) + m.group(2).upper(), num_clean)
    return num_clean


def canonicalize_commission_name(c_text: str) -> str:
    """
    - Convert roman, cardinal or ordinal numbers to the canonical UPPERCASE form "COMMISSIONE X".
    - Keep unnumbered bodies, strictly in UPPERCASE (e.g. "GIUNTA PER IL REGOLAMENTO").
    """
    if not isinstance(c_text, str) or not c_text.strip():
        return c_text

    text = c_text.strip()
    text_no_parentheses = re.sub(r'\s*\([^)]*\)', '', text).strip()

    roman_match = re.search(r'\b(XV|XIV|XIII|XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\b', text_no_parentheses, re.IGNORECASE)
    if roman_match and re.search(r'commissione', text_no_parentheses, re.IGNORECASE):
        rom = roman_match.group(1).upper()
        if rom in ROMAN_TO_ARABIC:
            return f"COMMISSIONE {ROMAN_TO_ARABIC[rom]}"

    arabic_match = re.search(r'\b(\d{1,2})\b', text_no_parentheses)
    if arabic_match and (re.search(r'commissione', text_no_parentheses, re.IGNORECASE) or text_no_parentheses.upper().startswith("COMMISSIONE")):
        return f"COMMISSIONE {arabic_match.group(1)}"

    text_clean = re.sub(r'^(alla|dalla|delle|della|la|il|del|dal)\s+', '', text_no_parentheses, flags=re.IGNORECASE).strip()
    return text_clean.upper()


def clean_commissions(ev: dict) -> list:
    cleaned = []
    for c in ev.get("commissions") or []:
        c_clean = canonicalize_commission_name(c)
        if c_clean and c_clean not in cleaned:
            cleaned.append(c_clean)
    return cleaned


def clean_persons(ev: dict) -> list:
    """
    Strip honorific titles and spurious phrases ("ED ALTRI"); keep NAME SURNAME in UPPERCASE.
    """
    cleaned = []
    for p in ev.get("persons") or []:
        if not isinstance(p, str) or not p.strip():
            continue
        p_clean = TITLES_REGEX.sub('', p).strip()
        p_clean = re.sub(r'[\(\)]', '', p_clean).strip()
        p_clean = re.sub(r'\b(ed\s+altri|ed\.\s*altri)\b.*$', '', p_clean, flags=re.IGNORECASE).strip()
        p_clean = re.sub(r'\s+', ' ', p_clean).upper()
        if p_clean and p_clean not in cleaned:
            cleaned.append(p_clean)
    return cleaned


def clean_sede_and_chamber(ev: dict, activity_it: str) -> tuple:
    """
    Normalize sede (in Italian, before translation) and chamber according to Rule 6 of Prompt v5:
    - richiesta_parere -> "consultiva"
    - presentazione, esame_aula, relazione -> null
    - assegnazione, esame_commissione, approvazione, stralcio -> canonical value or null.
    """
    sede_raw = ev.get("sede")
    sede_clean = None

    if activity_it == "richiesta_parere":
        sede_clean = "consultiva"
    elif activity_it in ("assegnazione", "esame_commissione", "approvazione", "stralcio"):
        if isinstance(sede_raw, str) and sede_raw.strip():
            s_low = sede_raw.strip().lower()
            for canonical in PROCEDURAL_MODE_TO_ENGLISH:
                if canonical in s_low:
                    sede_clean = canonical
                    break

    chamber_raw = _norm_str(ev.get("chamber", ""))
    chamber_clean = "camera" if "camera" in chamber_raw else ("senato" if "senato" in chamber_raw else chamber_raw)
    return sede_clean, chamber_clean


# ==============================================================================
# EVENT / RECORD / FILE
# ==============================================================================

def postprocess_event(ev: dict) -> dict:
    """Sanitize an event and translate its keys and values (activity, procedural_mode) to English."""
    if not isinstance(ev, dict):
        return ev

    activity_it = _norm_str(ev.get("activity"))
    sede_it, chamber = clean_sede_and_chamber(ev, activity_it)

    d = ev.get("date")
    unified = ev.get("atti_unificati") or []

    clean_it = {
        "sequence_id": ev.get("sequence_id"),
        "numero_atto": clean_bill_id(ev),
        "activity": activity_it,
        "date": str(d).strip() if d is not None else None,
        "commissions": clean_commissions(ev),
        "sede": sede_it,
        "persons": clean_persons(ev),
        "co_firmatari": bool(ev.get("co_firmatari", False)),
        "con_modificazioni": bool(ev.get("con_modificazioni", False)),
        "atti_unificati": [str(u).strip() for u in unified if str(u).strip()],
        "chamber": chamber,
    }

    out = {FIELD_MAPPING_TO_ENGLISH[k]: v for k, v in clean_it.items()}
    out["activity"] = ACTIVITY_TO_ENGLISH.get(activity_it, activity_it)
    if sede_it is not None:
        out["procedural_mode"] = PROCEDURAL_MODE_TO_ENGLISH.get(sede_it, sede_it)
    return out


def to_italian_event(ev: dict) -> dict:
    """Inverse of postprocess_event (translation only): used by the metrics to compare against the Gold Standard."""
    if not isinstance(ev, dict):
        return ev
    out = {FIELD_MAPPING_TO_ITALIAN.get(k, k): v for k, v in ev.items()}
    if isinstance(out.get("activity"), str):
        a = out["activity"].strip().lower()
        out["activity"] = ACTIVITY_TO_ITALIAN.get(a, out["activity"])
    if isinstance(out.get("sede"), str):
        s = out["sede"].strip().lower()
        out["sede"] = PROCEDURAL_MODE_TO_ITALIAN.get(s, out["sede"])
    return out


def postprocess_record(record: dict) -> dict:
    """Sanitize an LLM extraction record. Events always end up under `output["events"]`."""
    rec = copy.deepcopy(record)
    out = rec.get("output", {})

    events = out.pop("events", None)
    legacy = out.pop("eventi", None)
    if events is None:
        events = legacy or []

    out["events"] = [postprocess_event(ev) for ev in events if isinstance(ev, dict)]
    rec["output"] = out
    return rec


def postprocess_jsonl_file(input_file: str, output_file: str) -> int:
    """Post-process a whole JSONL file. Returns the number of records written."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    dir_name = os.path.dirname(output_file)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    n = 0
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            rec = postprocess_record(json.loads(line))
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Deterministic post-processing of LLM extractions (JSONL).")
    ap.add_argument("input")
    ap.add_argument("output", nargs="?")
    args = ap.parse_args()

    base, ext = os.path.splitext(args.input)
    outp = args.output or f"{base}_postprocessed{ext or '.jsonl'}"
    n = postprocess_jsonl_file(args.input, outp)
    print(f"[SUCCESS] Processed {n} records. File saved to: {outp}")

