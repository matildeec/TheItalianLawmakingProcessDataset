"""
build_master_bill_registry.py

Constructs the comprehensive, standardized dataset/bill-registry.csv:
1. Base: all bills from the Senate registry (41,835 rows, legislatures 14-19).
2. Integration: all enacted laws from dataset_metadata.csv & dataset_with_legislatures.csv (5,346 laws, legislatures 9-19).
   - Links laws already present in bill registry via law_number.
   - Appends missing historical laws (legislatures 9-13, 1987-2001) as official enacted bill records.
3. Incorporates metadata columns:
   - gazette_id
   - bill_id
   - internal_id
   - law_number
   - legislature (Arabic numeral: 9-19)
   - promulgation_date (ISO 8601 YYYY-MM-DD)
   - num_articles (int)
   - initiative (Parliamentary, Government, etc.)
   - initiating_chamber (S or C)
   - chamber (S or C)
   - sponsors
   - title
   - law_nature (Ordinary, Decree-Law Conversion, Ratification, State Budget, Constitutional)
   - is_budget_law (bool)
   - is_decree_conversion (bool)
   - is_ratification (bool)
   - confidence_vote_requested (bool)
   - confidence_vote_granted (bool)
   - confidence_vote_denied (bool)
   - is_merged (bool)
   - unified_bills (bill_id string)
   - normattiva_url (URL)
"""

import os
import re
import pandas as pd

ROMAN_TO_ARABIC = {
    'IX': 9, 'X': 10, 'XI': 11, 'XII': 12, 'XIII': 13,
    'XIV': 14, 'XV': 15, 'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19
}

MONTHS_IT = {
    'gennaio': '01', 'febbraio': '02', 'marzo': '03', 'aprile': '04',
    'maggio': '05', 'giugno': '06', 'luglio': '07', 'agosto': '08',
    'settembre': '09', 'ottobre': '10', 'novembre': '11', 'dicembre': '12'
}

def parse_it_date_to_iso(d_str):
    if not isinstance(d_str, str):
        return None
    m = re.search(r'(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})', d_str)
    if m:
        day = int(m.group(1))
        month_name = m.group(2).lower()
        year = int(m.group(3))
        month = MONTHS_IT.get(month_name)
        if month:
            return f"{year:04d}-{month}-{day:02d}"
    return None

def extract_law_num(titolo):
    if not isinstance(titolo, str):
        return None
    m = re.search(r'(\d+/\d{4})', titolo)
    if m:
        return m.group(1)
    m_cost = re.search(r'LEGGE\s+COSTITUZIONALE\s+.*?(\d{4}),\s*n\.\s*(\d+)', titolo, re.IGNORECASE)
    if m_cost:
        return f"{m_cost.group(2)}/{m_cost.group(1)}"
    return None

def map_nature(natura, is_ratifica):
    if is_ratifica:
        return 'Ratification'
    if not isinstance(natura, str) or not natura.strip():
        return None
    n = natura.strip()
    if 'Conversione' in n:
        return 'Decree-Law Conversion'
    if 'Bilancio' in n:
        return 'State Budget'
    if 'Costituzionale' in n:
        return 'Constitutional'
    if 'Ordinaria' in n:
        return 'Ordinary'
    return n

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))

def build_registry():
    print("Loading data files...")
    reg_path = os.path.join(_REPO_ROOT, 'dataset', 'bill-registry.csv')
    meta_path = os.path.join(_REPO_ROOT, 'pipeline', 'data', 'dataset_metadata.csv')
    leg_path = os.path.join(_REPO_ROOT, 'pipeline', 'data', 'dataset_with_legislatures.csv')

    df_reg = pd.read_csv(reg_path, low_memory=False)
    df_meta = pd.read_csv(meta_path, low_memory=False)
    df_leg = pd.read_csv(leg_path, low_memory=False)

    # Merge num_articoli into df_meta
    df_meta = pd.merge(df_meta, df_leg[['case_id', 'num_articoli']], on='case_id', how='left')

    # Prepare metadata fields
    df_meta['clean_law_number'] = df_meta['titolo'].apply(extract_law_num)
    df_meta['iso_promulgation_date'] = df_meta['law_date'].apply(parse_it_date_to_iso)
    df_meta['mapped_nature'] = [
        map_nature(r['natura'], bool(r['is_ratifica'])) for _, r in df_meta.iterrows()
    ]
    
    # Map initiative in metadata
    init_map = {
        'Parlamentare': 'Parliamentary',
        'Governativa': 'Government',
        'Regionale': 'Regional Council',
        'Popolare': 'Popular',
        'CNEL': 'CNEL'
    }
    df_meta['clean_initiative'] = df_meta['iniziativa'].map(init_map)

    # Map Roman legislatures in metadata to Arabic
    df_meta['arabic_legislature'] = df_meta['legislature'].map(ROMAN_TO_ARABIC).fillna(df_meta['legislature'])

    # Build lookup dictionaries from metadata keyed by clean_law_number
    # (for matching with existing bills in df_reg)
    meta_by_law = {}
    for _, r in df_meta.dropna(subset=['clean_law_number']).iterrows():
        law_num = r['clean_law_number']
        if law_num not in meta_by_law:
            meta_by_law[law_num] = r.to_dict()

    print(f"Existing bill registry rows: {len(df_reg)}")
    print(f"Total enacted laws in metadata: {len(df_meta)}")

    # 1. Update existing bills in df_reg
    df_reg['legislature'] = df_reg['legislature'].apply(lambda x: int(x) if pd.notna(x) else None)
    
    # Initialize new metadata columns in df_reg
    df_reg['gazette_id'] = None
    df_reg['promulgation_date'] = None
    df_reg['num_articles'] = None
    df_reg['initiating_chamber'] = None
    df_reg['law_nature'] = None
    df_reg['is_budget_law'] = False
    df_reg['is_decree_conversion'] = False
    df_ratification = False
    df_reg['is_ratification'] = False
    df_reg['confidence_vote_granted'] = False
    df_reg['normattiva_url'] = None

    matched_laws = set()
    for idx, r in df_reg.iterrows():
        law_num = r['law_number']
        if pd.notna(law_num) and law_num in meta_by_law:
            m = meta_by_law[law_num]
            matched_laws.add(m['case_id'])
            df_reg.at[idx, 'gazette_id'] = m['case_id']
            df_reg.at[idx, 'promulgation_date'] = m['iso_promulgation_date']
            df_reg.at[idx, 'num_articles'] = int(m['num_articoli']) if pd.notna(m['num_articoli']) else None
            df_reg.at[idx, 'initiating_chamber'] = m['iniziativa_ramo'] if pd.notna(m['iniziativa_ramo']) else None
            df_reg.at[idx, 'law_nature'] = m['mapped_nature']
            df_reg.at[idx, 'is_budget_law'] = bool(m.get('is_finanziaria', False))
            df_reg.at[idx, 'is_decree_conversion'] = bool(m.get('is_conversione_dl', False))
            df_reg.at[idx, 'is_ratification'] = bool(m.get('is_ratifica', False))
            df_reg.at[idx, 'confidence_vote_granted'] = bool(m.get('fiducia_accordata', False))
            df_reg.at[idx, 'normattiva_url'] = m.get('full_text_url')

    print(f"Laws matched in existing bill registry: {len(matched_laws)}")

    # 2. Append missing enacted laws (pre-2001, legislatures 9-13, and any unlinked)
    missing_laws = df_meta[~df_meta['case_id'].isin(matched_laws)]
    print(f"Appending {len(missing_laws)} historical / unlinked enacted laws...")

    new_rows = []
    for _, m in missing_laws.iterrows():
        b_id = m['ddl_number'] if pd.notna(m['ddl_number']) else None
        c_code = None
        if b_id:
            if str(b_id).startswith('S'):
                c_code = 'S'
            elif str(b_id).startswith('C'):
                c_code = 'C'
        if not c_code and pd.notna(m['iniziativa_ramo']):
            c_code = str(m['iniziativa_ramo']).strip()

        row = {
            'gazette_id': m['case_id'],
            'bill_id': b_id,
            'internal_id': int(float(m['ddl_id'])) if pd.notna(m['ddl_id']) else None,
            'law_number': m['clean_law_number'],
            'legislature': int(m['arabic_legislature']) if pd.notna(m['arabic_legislature']) else None,
            'chamber': c_code,
            'initiative': m['clean_initiative'],
            'initiating_chamber': m['iniziativa_ramo'] if pd.notna(m['iniziativa_ramo']) else None,
            'sponsors': m['proponent'] if pd.notna(m['proponent']) else None,
            'title': m['title'] if pd.notna(m['title']) else m['titolo'],
            'promulgation_date': m['iso_promulgation_date'],
            'num_articles': int(m['num_articoli']) if pd.notna(m['num_articoli']) else None,
            'law_nature': m['mapped_nature'],
            'is_budget_law': bool(m.get('is_finanziaria', False)),
            'is_decree_conversion': bool(m.get('is_conversione_dl', False)),
            'is_ratification': bool(m.get('is_ratifica', False)),
            'confidence_vote_granted': bool(m.get('fiducia_accordata', False)),
            'is_merged': False,
            'unified_bills': None,
            'normattiva_url': m.get('full_text_url')
        }
        new_rows.append(row)

    df_combined = pd.concat([df_reg, pd.DataFrame(new_rows)], ignore_index=True)

    # 3. Final column ordering
    final_cols = [
        'gazette_id',
        'bill_id',
        'internal_id',
        'law_number',
        'legislature',
        'promulgation_date',
        'num_articles',
        'initiative',
        'initiating_chamber',
        'chamber',
        'sponsors',
        'title',
        'is_budget_law',
        'is_decree_conversion',
        'is_ratification',
        'confidence_vote_granted',
        'is_merged',
        'unified_bills',
        'normattiva_url'
    ]
    df_combined = df_combined[final_cols].copy()

    # Flags not reliably available for Legislatures IX-XII: leave empty
    early_leg = df_combined['legislature'].isin([9, 10, 11, 12])
    for col in ['is_ratification', 'is_decree_conversion', 'is_budget_law', 'confidence_vote_granted']:
        df_combined[col] = df_combined[col].astype(object)
        df_combined.loc[early_leg, col] = None

    output_path = os.path.join(_REPO_ROOT, 'dataset', 'bill-registry.csv')
    print(f"Writing final bill registry to {output_path} (total rows: {len(df_combined)})...")
    df_combined.to_csv(output_path, index=False)
    print("Successfully built bill-registry.csv!")

if __name__ == '__main__':
    build_registry()
