# End-to-End Pipeline & Methodology

## Overview

This directory contains the computational pipeline used to extract, evaluate, and post-process event logs from Italian legislative preparatory texts (*lavori preparatori*). The workflow is organized into four sequential phases, progressing from dataset partitioning and token budgeting to LLM-based event extraction, rigorous benchmark evaluation, and deterministic post-processing.

All commands are intended to be executed from the `pipeline/` directory.

---

## Prerequisites & Environment Setup

The pipeline requires Python 3.10 or higher. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Hardware Considerations

- **Phases 1, 3, and 4**: Can be executed on standard multi-core CPU environments.
- **Phase 2 (LLM Extraction)**: Requires an active [Ollama](https://ollama.com/) instance serving [`gemma4:31b`](https://ollama.com/library/gemma4) (supported on NVIDIA GPUs, Apple Silicon Metal, or remote Ollama hosts). To bypass local inference, the full raw extraction artifact is pre-computed and tracked in `pipeline/output/full_raw_extraction_gemma4.jsonl`.

---

## Execution Protocol

### Phase 1: Data Preparation & Context Budgeting (`01_data-preparation`)

This phase handles dataset partitioning and empirical context window sizing.

1. **Stratified Partitioning (`prepare_splits.py`)**:
   Extracts a representative validation sample of 100 bills stratified proportionally across legislatures IX to XIX (`data/val_sample_100.csv`, random seed `42`). It also creates the extraction set (`data/dataset_final_extraction.csv`) by excluding the 100 validation bills to prevent data leakage.

   ```bash
   python 01_data-preparation/prepare_splits.py
   ```

2. **Empirical Token Distribution & Context Sizing (`estimate_tokens.py`)**:
   Measures token distributions across prompt templates, few-shot demonstrations, inputs, and annotated gold outputs using open-weight tokenizers (Gemma, Qwen, Mistral, tiktoken). It generates empirical statistics exported to `01_data-preparation/token_statistics.json`, justifying the context allocation parameters (`num_ctx: 65536`, `num_predict: 49152`) configured for Phase 2.

   ```bash
   python 01_data-preparation/estimate_tokens.py
   ```

### Phase 2: LLM Information Extraction (`02_llm-extraction`)

Extracts sequential parliamentary event objects from raw legislative text into structured JSON records.

- **Inference Runtime**: Ollama running [`gemma4:31b`](https://ollama.com/library/gemma4).
- **Prompt Strategy**: Few-shot in-context learning (`prompts/event_extraction_instruct.txt` augmented with curated demonstrations from `data/fewshot_examples/`).
- **Generation Settings**: Deterministic decoding (`temperature: 0.0`, `seed: 42`, `top_k: 10`, `top_p: 0.9`).

Execution commands:

```bash
# Test execution on a limited slice
python 02_llm-extraction/run_extraction.py --start 0 --end 5 --mode mock

# Full inference on the corpus
python 02_llm-extraction/run_extraction.py --start 0 --end 5346 --mode real
```

> **Note**: The full extraction across all corpus bills is available at `output/full_raw_extraction_gemma4.jsonl`, enabling immediate execution of Phase 3 and Phase 4 without re-running LLM inference.

### Phase 3: Validation & Metric Evaluation (`03_evaluation`)

Evaluates model extractions against the human-annotated Gold Standard (`data/gold_standard.json`, 100 bills, 2,032 events). To ensure rigorous ground-truth quality, candidate events were initially pre-annotated by a commercial LLM and subsequently audited, corrected, and verified across three independent human review passes.

The evaluation framework assesses:
- **Exact Match & Jaccard Overlap**: Token-level alignment across structured fields.
- **Hungarian Bipartite Matching**: Global assignment optimization across unordered predicted event sets and ground truth events (`threshold: 0.80` for model selection).
- **Schema Compliance**: Syntactic validity, date format consistency, and constraint satisfaction.

Execution commands:

```bash
# Evaluate files specified in config_eval.py or latest file in output/
python 03_evaluation/run_evaluation.py

# Evaluate a specific extraction artifact
python 03_evaluation/run_evaluation.py --input output/full_raw_extraction_gemma4.jsonl

# Force re-computation of cached evaluation results
python 03_evaluation/run_evaluation.py --force
```

### Phase 4: Post-Processing, Error Detection & Dataset Consolidation (`04_post-processing`)

Applies deterministic sanitization, normalizes procedural fields, runs rule-based error detection, and consolidates the final two-layer dataset published in `/dataset/`:

1. **Deterministic Sanitization & Ontology Normalization (`postprocess_results.py`)**:
   - Maps Italian activity types (`presentazione`, `assegnazione`, etc.) to canonical English labels (`presentation`, `assignment`, etc.).
   - Standardizes procedural modes (`referral`, `deliberative`, etc.) and parliamentary chambers.
   - Canonicalizes parliamentary committee names (`COMMISSIONE X`) and strips titles/spurious phrases from sponsor names.

2. **Error Detection & Quality Audit (`error_detection/flag.py`)**:
   - Audits chronological monotonicity, sequence order, and domain constraints across event logs.
   - Flags critical anomalies (temporal inversions, sequence-date mismatches, unauthorized approvals).
   - Generates `flagged_cases.jsonl` which is used to populate the `suspicious: true/false` quality audit flag.

   ```bash
   python 04_post-processing/error_detection/flag.py --input output/full_raw_extraction_gemma4.jsonl
   ```

3. **Master Dataset Export**:
   - Builds `dataset/bill-registry.csv` (consolidates 44,980 legislative bill initiatives across Legislatures 9–19 with 22 institutional metadata fields).
   - Builds `dataset/preparatory-work.jsonl` (merges the 5,246 model extractions with the 100 human-annotated Gold Standard bills, applies ontology translation, and appends the `suspicious` flag).

   ```bash
   # Consolidate bill initiatives and legislative metadata into the master registry
   python 04_post-processing/build_master_bill_registry.py

   # Build the finalized preparatory works event log
   python 04_post-processing/build_preparatory_work.py
   ```

---

## Directory Structure & Conventions

| Path | Description |
| :--- | :--- |
| `data/dataset_with_legislatures.csv` | Full corpus of bills with legislative texts and legislature assignments. |
| `data/val_sample_100.csv` | Stratified 100-bill validation split. |
| `data/dataset_final_extraction.csv` | Corpus partition strictly excluding the validation split. |
| `data/gold_standard.json` | Human-annotated ground-truth event logs for the validation set. |
| `data/fewshot_examples/` | Curated JSON demonstrations for few-shot prompt construction. |
| `output/full_raw_extraction_gemma4.jsonl` | Pre-computed LLM extraction output for the complete corpus. |
