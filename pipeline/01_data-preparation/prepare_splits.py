"""
Dataset Split & Sampling Orchestrator
=====================================
Prepares:
1. Stratified Validation Sample (100 bills balanced across legislatures IX-XIX).
2. Final Extraction Split (the entire corpus excluding the 100 validation bills to prevent data leakage).
"""

import os
import argparse
import pandas as pd
import numpy as np

# Path definitions
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PIPELINE_ROOT, 'data')

DEFAULT_INPUT_CSV = os.path.join(DATA_DIR, 'dataset_with_legislatures.csv')
DEFAULT_VAL_CSV = os.path.join(DATA_DIR, 'val_sample_100.csv')
DEFAULT_FINAL_CSV = os.path.join(DATA_DIR, 'dataset_final_extraction.csv')

SAMPLE_SIZE = 100
RANDOM_SEED = 42


def run_stratified_sampling(
    input_csv: str = DEFAULT_INPUT_CSV,
    output_val_csv: str = DEFAULT_VAL_CSV,
    sample_size: int = SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Extracts a representative, stratified sample of bills proportionally distributed by legislature.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f'Input dataset not found at: {input_csv}')

    print(f'[STEP 1] Loading source dataset: {input_csv}...')
    df_main = pd.read_csv(input_csv)
    df_main = df_main.dropna(subset=['lavori_preparatori'])
    df_main = df_main[df_main['lavori_preparatori'].str.strip() != '']

    print(f'[STEP 1] Performing stratified sampling (N={sample_size}, seed={seed})...')
    leg_counts = df_main['legislature'].value_counts(normalize=True)
    sample_sizes = (leg_counts * sample_size).round().astype(int)

    # Rounding difference adjustment to ensure exact sample_size
    diff = sample_size - sample_sizes.sum()
    if diff != 0:
        largest_leg = sample_sizes.idxmax()
        sample_sizes[largest_leg] += diff

    sample_list = []
    for leg, size in sample_sizes.items():
        if size > 0:
            df_leg_group = df_main[df_main['legislature'] == leg]
            n_to_sample = min(size, len(df_leg_group))
            sampled = df_leg_group.sample(n=n_to_sample, random_state=seed)
            sample_list.append(sampled)

    df_sample = pd.concat(sample_list)
    os.makedirs(os.path.dirname(output_val_csv), exist_ok=True)
    df_sample.to_csv(output_val_csv, index=False)

    print(f'[SUCCESS] Stratified sample saved to: {output_val_csv} ({len(df_sample)} records)')
    return df_sample


def create_final_extraction_dataset(
    input_csv: str = DEFAULT_INPUT_CSV,
    val_csv: str = DEFAULT_VAL_CSV,
    output_final_csv: str = DEFAULT_FINAL_CSV,
) -> pd.DataFrame:
    """
    Creates the final extraction dataset by excluding the validation sample (zero data leakage).
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f'Input dataset not found at: {input_csv}')
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f'Validation sample not found at: {val_csv}. Please run sampling first.')

    print(f'[STEP 2] Creating final extraction dataset excluding validation set...')
    df_full = pd.read_csv(input_csv)
    df_val = pd.read_csv(val_csv)

    val_ids = set(df_val['case_id'])
    df_final = df_full[~df_full['case_id'].isin(val_ids)].copy()

    os.makedirs(os.path.dirname(output_final_csv), exist_ok=True)
    df_final.to_csv(output_final_csv, index=False)

    print(f'[SUCCESS] Final extraction dataset saved to: {output_final_csv} ({len(df_final)} records)')
    return df_final


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset splits (validation sample + final extraction set).')
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_CSV, help='Source CSV dataset with legislatures.')
    parser.add_argument('--val-output', type=str, default=DEFAULT_VAL_CSV, help='Destination for validation sample CSV.')
    parser.add_argument('--final-output', type=str, default=DEFAULT_FINAL_CSV, help='Destination for final extraction CSV.')
    parser.add_argument('--sample-size', type=int, default=SAMPLE_SIZE, help=f'Validation sample size (default: {SAMPLE_SIZE}).')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help=f'Random seed (default: {RANDOM_SEED}).')
    parser.add_argument('--step', choices=['all', 'sample', 'final'], default='all', help='Execution step (default: all).')

    args = parser.parse_args()

    if args.step in ('all', 'sample'):
        run_stratified_sampling(
            input_csv=args.input,
            output_val_csv=args.val_output,
            sample_size=args.sample_size,
            seed=args.seed,
        )

    if args.step in ('all', 'final'):
        create_final_extraction_dataset(
            input_csv=args.input,
            val_csv=args.val_output,
            output_final_csv=args.final_output,
        )


if __name__ == '__main__':
    main()
