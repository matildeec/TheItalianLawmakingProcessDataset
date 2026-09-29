"""
Evaluation Utils Package
========================
Evaluation metrics and verification modules:
1. prompt_schema_compliance: Verify structural and domain adherence to the prompt schema.
2. exact_match_jaccard: Compute Event-Set Exact Match Jaccard similarity and attribute-level accuracy.
3. hungarian_matching: Bipartite matching via the Hungarian Algorithm for precision, recall, and F1-scores.
"""

from .prompt_schema_compliance import (
    verify_prompt_schema_compliance,
    verify_event_schema,
)
from .exact_match_jaccard import verify_exact_match_jaccard
from .hungarian_matching import verify_hungarian_matching

__all__ = [
    "verify_prompt_schema_compliance",
    "verify_event_schema",
    "verify_exact_match_jaccard",
    "verify_hungarian_matching",
]
