"""
Create LC_train.csv and LC_test.csv from archive/loan.csv
- Temporal split: sorted by issue_d
- Train: 100,000 oldest loans
- Test: 10,000 next-oldest loans
- Select ~39 application-time variables (exclude post-origination leakage)
- Test: remove int_rate, add sequential ID column
"""

import pandas as pd
import numpy as np
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'archive', 'loan.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data')

# Application-time variables to keep (exclude post-origination leakage)
SELECTED_COLS = [
    # Core loan terms
    'loan_amnt', 'term', 'int_rate', 'installment',
    'grade', 'sub_grade',
    # Borrower info
    'emp_title', 'emp_length', 'home_ownership', 'annual_inc',
    'verification_status', 'issue_d',
    # Loan purpose & location
    'purpose', 'zip_code', 'addr_state',
    # Credit profile
    'dti', 'delinq_2yrs', 'earliest_cr_line',
    'total_bc_limit', 'percent_bc_gt_75',
    'inq_last_6mths', 'open_acc', 'pub_rec',
    'revol_bal', 'revol_util', 'total_acc',
    'initial_list_status', 'application_type',
    'mort_acc', 'pub_rec_bankruptcies',
    'collections_12_mths_ex_med', 'chargeoff_within_12_mths',
    'num_accts_ever_120_pd', 'num_tl_90g_dpd_24m', 'pct_tl_nvr_dlq',
    'acc_now_delinq', 'tax_liens',
    'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_tl',
]

def main():
    print(f"Loading selected columns from {DATA_PATH}...")
    
    # Load in chunks to manage memory
    chunks = []
    chunk_iter = pd.read_csv(DATA_PATH, usecols=SELECTED_COLS, chunksize=100_000, low_memory=False)
    for i, chunk in enumerate(chunk_iter):
        chunks.append(chunk)
        print(f"  Loaded chunk {i+1}: {len(chunk)} rows")
    
    df = pd.concat(chunks, ignore_index=True)
    print(f"Total rows loaded: {len(df):,}")
    
    # Convert issue_d to datetime for sorting
    print("Converting issue_d to datetime...")
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
    
    # Drop rows with missing issue_d (can't sort temporally)
    before_drop = len(df)
    df = df.dropna(subset=['issue_d'])
    after_drop = len(df)
    if before_drop != after_drop:
        print(f"Dropped {before_drop - after_drop} rows with missing issue_d")
    
    # Sort chronologically
    print("Sorting by issue_d...")
    df = df.sort_values('issue_d', ascending=True).reset_index(drop=True)
    
    # Temporal split: train = oldest 100k, test = next 10k
    print("Creating temporal train/test split...")
    train = df.iloc[:100_000].copy()
    test = df.iloc[100_000:110_000].copy()
    
    print(f"Train shape: {train.shape} | Date range: {train['issue_d'].min().date()} to {train['issue_d'].max().date()}")
    print(f"Test shape:  {test.shape} | Date range: {test['issue_d'].min().date()} to {test['issue_d'].max().date()}")
    
    # Save train
    train_path = os.path.join(OUTPUT_DIR, 'LC_train.csv')
    train.to_csv(train_path, index=False)
    print(f"Saved LC_train.csv ({len(train):,} rows, {len(train.columns)} cols)")
    
    # Prepare test: remove int_rate, add ID
    test = test.drop(columns=['int_rate'])
    test.insert(0, 'ID', range(1, len(test) + 1))
    
    test_path = os.path.join(OUTPUT_DIR, 'LC_test.csv')
    test.to_csv(test_path, index=False)
    print(f"Saved LC_test.csv ({len(test):,} rows, {len(test.columns)} cols)")
    
    print("\nDone!")

if __name__ == '__main__':
    main()
