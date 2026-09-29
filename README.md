# The Italian Lawmaking Process Dataset

## Abstract

<!-- To be completed with the publication abstract. -->

## Dataset Overview

<!-- To be completed with publication metadata (collection period, corpus statistics, and official DOI). -->

## Repository Architecture

The repository is structured into two main components: published dataset artifacts and the end-to-end extraction and validation pipeline.

```text
TheItalianLawmakingProcessDataset/
├── README.md               # Project overview, data dictionary, and usage instructions
├── dataset/                # Published dataset files and registries
│   └── ddl_registry.csv    # Master registry of Italian legislative bills
└── pipeline/               # End-to-end extraction, evaluation, and processing code
    ├── README.md           # Detailed reproduction protocol and CLI instructions
    ├── requirements.txt    # Python dependencies for all pipeline phases
    ├── data/               # Input corpora, validation sample, and gold standard
    ├── output/             # Full raw and post-processed extraction artifacts
    ├── 01_data-preparation/# Stratified sampling and context token budgeting
    ├── 02_llm-extraction/  # Ollama LLM extraction scripts and prompt templates
    ├── 03_evaluation/      # Benchmark evaluation framework against gold standard
    └── 04_post-processing/ # Deterministic sanitization and ontology normalization
```

## Data Dictionary

### 1. Legislative Bill Registry (`dataset/ddl_registry.csv`)

The master registry contains legislative proposals introduced across Italian legislatures.

| Column | Type | Description |
| :--- | :--- | :--- |
| `ddl_id` | Integer | Unique internal bill identifier. |
| `ddl_number` | String | Formal parliamentary designation (e.g., `S.9`, `C.1704`). |
| `legislatura` | Integer / String | Legislature number (IX to XIX). |
| `ramo` | String | Parliamentary chamber of origin (`C` for Camera dei Deputati, `S` for Senato della Repubblica). |
| `iniziativa` | String | Institutional origin of the initiative (e.g., Parliamentary, Government, Regional Council). |
| `proponent` | String | Primary sponsor, proposing minister, or submitting institution. |
| `title` | String | Official title and subject matter of the legislative bill. |
| `events` | String | Chronological natural-language summary of legislative milestones (*lavori preparatori*). |
| `law_number` | String | Enacted law number, if promulgated. |
| `normattiva_titolo` | String | Formal title recorded in the official Normattiva legal database, if enacted. |
| `law_date` | String | Promulgation date of the resulting statute. |
| `merged` | Boolean | Flag indicating whether the bill was consolidated into a unified legislative text. |
| `merged_with` | String | Act identifier of the unified legislative bill, if applicable. |

### 2. Extracted Event Schema (`pipeline/output/`)

Extracted event logs follow a standardized sequential event schema. Each record contains an array of chronologically ordered parliamentary events.

| Field | Type | Description |
| :--- | :--- | :--- |
| `sequence_id` | Integer | 1-based index denoting the chronological sequence of the event within the bill. |
| `bill_id` | String | Normalized bill act identifier without textual prefixes. |
| `activity` | String | Canonical event class: `presentation`, `assignment`, `opinion_request`, `committee_examination`, `floor_examination`, `approval`, `report`, `severance`. |
| `date` | String / Null | ISO 8601 date string (`YYYY-MM-DD`), or `null` if unstated. Exactly one date per event object. |
| `commissions` | List[String] | Uppercase identifiers of involved parliamentary committees (e.g., `["COMMISSIONE 1"]`). |
| `procedural_mode` | String / Null | Canonical legislative procedure: `referral`, `legislative`, `deliberative`, `drafting`, `advisory`, or `null`. |
| `sponsors` | List[String] | Uppercase surnames of explicitly designated sponsors or rapporteurs. |
| `has_co_sponsors` | Boolean | Indicates whether additional co-sponsors are present (*"ed altri"*). |
| `with_amendments` | Boolean | Indicates whether the bill or amendment passed with textual modifications. |
| `unified_bills` | List[String] | List of bill identifiers consolidated into the current act. |
| `chamber` | String | Parliamentary chamber where the event occurred (`camera`, `senato`). |

## Getting Started

### Prerequisites

Ensure Python 3.10+ is installed. Clone the repository and install dependencies:

```bash
git clone https://github.com/matildeec/TheItalianLawmakingProcessDataset.git
cd TheItalianLawmakingProcessDataset
pip install -r pipeline/requirements.txt
```

### Loading the Dataset

To load and inspect the primary bill registry using pandas:

```python
import pandas as pd

# Load master registry
df_registry = pd.read_csv("dataset/ddl_registry.csv")
print(f"Total bills indexed: {len(df_registry)}")
print(df_registry[["ddl_number", "legislatura", "ramo", "title"]].head())
```

To inspect extracted process events:

```python
import json

# Inspect extracted event stream
with open("pipeline/output/full_raw_extraction_gemma4.jsonl", "r", encoding="utf-8") as f:
    sample_record = json.loads(f.readline())
    print(f"Case ID: {sample_record['id']}")
    print(json.dumps(sample_record.get('eventi', []), indent=2))
```

For instructions on executing or reproducing each phase of the extraction pipeline, refer to [`pipeline/README.md`](pipeline/README.md).

## Citation

<!-- To be completed upon publication. -->

```bibtex
@article{italian_lawmaking_dataset_2026,
  title={The Italian Lawmaking Process Dataset},
  author={},
  journal={},
  year={2026}
}
```

## License

<!-- Specify license details (e.g., Creative Commons Attribution 4.0 International for data, MIT for code). -->
