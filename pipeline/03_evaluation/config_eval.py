import os

# ==============================================================================
# BASE PATHS CONFIGURATION
# ==============================================================================
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(EVAL_DIR)

# Default Gold Standard path
GOLD_STANDARD_PATH = os.path.join(BASE_DIR, "data", "gold_standard.json")

# Directory containing benchmark LLM extraction files
EXTRACTION_OUTPUT_DIR = os.path.join(EVAL_DIR, "extractions")

# List of output files in extractions/ to evaluate directly.
# If empty ([]), run_evaluation.py will evaluate the most recently generated file in extractions/.
# If specified, run_evaluation.py will evaluate all listed files sequentially.
FILES_TO_EVALUATE = [
    "extraction_zero-shot_2026-08-13_18-19.jsonl",
    "extraction_few-shot_2026-08-13_19-59.jsonl",
    "extraction_zero-shot_2026-08-13_21-35.jsonl",
    "extraction_few-shot_2026-08-14_11-07.jsonl",
    "extraction_zero-shot_2026-08-14_14-46.jsonl",
    "extraction_few-shot_2026-08-14_15-19.jsonl",
    "extraction_zero-shot_2026-08-14_19-43.jsonl",
    "extraction_few-shot_2026-08-14_17-40.jsonl",
    "extraction_zero-shot_2026-08-14_23-24.jsonl",
    "extraction_few-shot_2026-08-14_22-05.jsonl"
]

# Destination path for evaluation results in JSON format
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "evaluation_results.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================================================================
# EVALUATION PARAMETERS
# ==============================================================================
# Minimum similarity threshold for event bipartite matching (Hungarian Algorithm).
# Serves as a standard threshold for model selection, filtering spurious alignments
# while accommodating minor paraphrasing in committee and procedural names.
HUNGARIAN_SIMILARITY_THRESHOLD = 0.80

# Multilingual Sentence Transformer model for semantic embeddings
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ==============================================================================
# PROMPT SCHEMA STRUCTURE, ENUMS & CONSTRAINTS
# ==============================================================================
REQUIRED_SCHEMA = {
    "sequence_id": int,
    "activity": str,
    "date": (str, type(None)),
    "commissions": list,
    "sede": (str, type(None)),
    "persons": list,
    "co_firmatari": bool,
    "con_modificazioni": bool,
    "atti_unificati": list,
    "chamber": str,
}

DATE_REGEX_PATTERN = r"^\d{4}-\d{2}-\d{2}$"

ALLOWED_ACTIVITIES = {
    "presentazione",
    "assegnazione",
    "esame_commissione",
    "esame_aula",
    "approvazione",
    "richiesta_parere",
    "relazione",
    "stralcio",
}

ALLOWED_SEDI = {
    "referente",
    "legislativa",
    "deliberante",
    "redigente",
    "consultiva",
    None,
}

ALLOWED_CHAMBERS = {
    "camera",
    "senato",
}
