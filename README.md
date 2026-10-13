# The Italian Lawmaking Process Dataset

## Dataset Overview

The Italian Lawmaking Process Dataset provides a comprehensive, multi-layer reconstruction of the Italian legislative process across four decades (Legislatures IX to XIX, 1985–2026). The dataset is published under the `/dataset/` directory and consists of three relationally coupled resources:

1. **The Laws Registry (`dataset/laws_registry.csv`)**: Macro-level tabular registry containing one record for each of the 5,346 enacted laws promulgated in the Italian Parliament across Legislatures IX to XIX, with 16 institutional, political, and procedural attributes.
2. **The Procedural Event Log (`dataset/preparatory_work.jsonl`)**: Micro-level, highly granular event log containing 101,413 chronologically ordered procedural events reconstructed from the preparatory works (*lavori preparatori*) of all 5,346 enacted laws promulgated between 1985 and 2026.
3. **The Full Statute Texts (`dataset/full_texts/`)**: Raw text corpus containing the complete legal texts of all 5,346 enacted laws stored as individual UTF-8 text files named after their unique statutory identifier (e.g., `26G00136.txt`).

---

## Repository Architecture

```text
TheItalianLawmakingProcessDataset/
├── README.md               # Master documentation, data dictionary, and quickstart guide
├── dataset/                # Published dataset distribution files
│   ├── laws_registry.csv   # Layer 1: Macro-level registry of enacted laws (5,346 laws / 5,346 records)
│   ├── preparatory_work.jsonl # Layer 2: Micro-level procedural event log (5,346 laws, 101,413 events)
│   └── full_texts/         # Layer 3: Full legal text corpus of enacted statutes (5,346 .txt files)
└── pipeline/               # End-to-end extraction, evaluation, and processing code
    ├── README.md           # Reproduction protocol and CLI execution guide
    ├── requirements.txt    # Python dependencies for the pipeline
    ├── data/               # Corpora, validation splits, and human-annotated gold standard
    ├── output/             # Full raw and post-processed extraction artifacts
    ├── 01_data-preparation/# Stratified sampling and context token budgeting
    ├── 02_llm-extraction/  # Ollama LLM extraction scripts and prompt templates
    ├── 03_evaluation/      # Benchmark evaluation framework against gold standard
    └── 04_post-processing/ # Post-processing, error detection, and master export scripts
```

---

## Relational Joining: Coupling Layer 1 and Layer 2

The two resources are linked via a common law identifier based on the official Gazette editorial code:

* **Join Key**: `law_id` (e.g., `26G00136`, `088G0238`)
* **Cardinality**: One-to-one. `law_id` is unique in both resources.
* **Coverage**: Exactly 5,346 out of 5,346 laws in `preparatory_work.jsonl` possess a corresponding record in `laws_registry.csv` (100% relational integrity).

---

## Data Dictionary

### 1. Laws Registry (`dataset/laws_registry.csv`)

| Column | Type | Description |
| :--- | :--- | :--- |
| `law_id` | String | Unique law identifier based on the official editorial code in the Official Gazette / Normattiva (e.g., `26G00136`). Primary key and join key to `preparatory_work.jsonl`. |
| `law_number` | String / Null | Formal statute number (e.g., `120/2026`, `129/2004`). |
| `legislature` | Integer | Legislature number in Arabic numerals (`9` to `19`). |
| `promulgation_date` | String / Null | Date of presidential enactment in ISO 8601 format (`YYYY-MM-DD`). |
| `num_articles` | Integer / Null | Total count of statutory articles composing the enacted statute. |
| `initiative` | String / Null | Institutional origin: `Government`, `Parliamentary`, `Regional Council`, `Popular`, `CNEL`. |
| `initiating_chamber` | String / Null | Chamber where the bill was initially introduced (`S` for Senate, `C` for Chamber of Deputies). |
| `chamber` | String / Null | Parliamentary chamber (`S` or `C`). |
| `sponsors` | String / Null | Primary bill sponsor(s), cabinet ministers, or submitting institutions. |
| `title` | String | Official title and subject matter of the enacted law. |
| `is_budget_law` | Boolean | `true` if the statute is an annual financial or budget law (*legge finanziaria / bilancio*). |
| `is_decree_conversion` | Boolean | `true` if the statute converts an urgent government decree-law into ordinary law. |
| `is_ratification` | Boolean | `true` if the statute ratifies an international treaty or bilateral agreement. |
| `confidence_vote_granted` | Boolean / Null | `true` if the government confidence vote passed during parliamentary consideration (tracked for Legislatures XIII–XIX; `null` for early historical archives IX–XII). |
| `is_merged` | Boolean / Null | `true` if the bill was formally consolidated into a unified legislative text (*testo unificato*; tracked for Legislatures XIV–XIX; `null` for Legislatures IX–XIII). |
| `normattiva_url` | String / Null | Canonical hyperlink to the full legal text on the official Normattiva portal. |

### 2. Procedural Event Log (`dataset/preparatory_work.jsonl`)

Each line in `preparatory_work.jsonl` represents an enacted law and contains its raw preparatory text, a quality audit flag, and the chronologically ordered procedural event sequence:

```json
{
  "law_id": "26G00136",
  "input_text": "Camera dei deputati (atto n. 1704): ...",
  "suspicious": false,
  "events": [ ... ]
}
```

#### Top-Level Law Properties:
| Field | Type | Description |
| :--- | :--- | :--- |
| `law_id` | String | Common law identifier based on the unique Gazette editorial code, matching `law_id` in `laws_registry.csv`. |
| `input_text` | String | Unstructured legal summary of preparatory proceedings (*lavori preparatori*). |
| `suspicious` | Boolean | Quality assurance flag: `true` if rule-based error detection identified critical temporal or procedural inversions, `false` otherwise. |
| `events` | List[Object] | Ordered array of granular parliamentary events. |

#### Event Object Schema (`events[]`):
| Field | Type | Description |
| :--- | :--- | :--- |
| `sequence_id` | Integer | 1-based chronological sequence order within the legislative proceeding. |
| `activity` | String | Canonical parliamentary action: `presentation`, `assignment`, `opinion_request`, `committee_examination`, `floor_examination`, `approval`, `report`, `severance`. |
| `date` | String / Null | Event date in ISO 8601 format (`YYYY-MM-DD`), or `null` if unstated. Exactly one date per event. |
| `commissions` | List[String] | Uppercase canonical designations of participating parliamentary bodies (e.g., `["COMMISSIONE 10"]`). |
| `procedural_mode` | String / Null | Parliamentary examination mode (*sede*): `referral` (*referente*), `legislative` (*legislativa*), `deliberative` (*deliberante*), `drafting` (*redigente*), `advisory` (*consultiva*), or `null`. |
| `sponsors` | List[String] | Uppercase names of explicitly designated rapporteurs or proposing sponsors. |
| `has_co_sponsors` | Boolean | `true` if additional co-sponsors are indicated (*"ed altri"*). |
| `with_amendments` | Boolean | `true` if the bill was approved with amendments (*con modificazioni*). |
| `unified_bills` | List[String] | Act numbers formally consolidated into the proceeding at this event. |
| `chamber` | String | Parliamentary house where the action occurred (`camera`, `senato`). |

### 3. Full Statute Texts (`dataset/full_texts/`)

The `dataset/full_texts/` directory stores the promulgated legal texts for all 5,346 unique enacted statutes in the dataset. Each file is formatted as raw UTF-8 text and named using the common statutory identifier: `<law_id>.txt` (e.g., `26G00136.txt`), allowing direct joining with both `laws_registry.csv` and `preparatory_work.jsonl`.

---

## Quickstart & Usage Example

```python
import json
import pandas as pd

# 1. Load Laws Registry
df_registry = pd.read_csv("dataset/laws_registry.csv", low_memory=False)
print(f"Total registry records: {len(df_registry)}")
print(f"Total unique enacted laws: {df_registry['law_id'].nunique()}")

# 2. Load Process Event Log
laws_events = []
with open("dataset/preparatory_work.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        laws_events.append(json.loads(line))

print(f"Total enacted process logs: {len(laws_events)}")

# 3. Relational Join: inspect events for a given enacted law
sample_case = laws_events[0]
sample_id = sample_case["law_id"]
registry_row = df_registry[df_registry["law_id"] == sample_id].iloc[0]

print(f"\nLaw {registry_row['law_number']} ({registry_row['title']})")
print(f"Promulgated on: {registry_row['promulgation_date']}, Articles: {registry_row['num_articles']}")
print(f"Reconstructed parliamentary events: {len(sample_case['events'])}")
for ev in sample_case["events"][:3]:
    print(f" - #{ev['sequence_id']} [{ev['date']}] {ev['activity']} ({ev['chamber']})")
```

For instructions on executing or reproducing the computational extraction and evaluation pipeline, refer to [`pipeline/README.md`](pipeline/README.md).

---

## License

This dataset is distributed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. Code is provided under the [MIT License](https://opensource.org/licenses/MIT).
