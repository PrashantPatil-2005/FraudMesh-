import pandas as pd
from src.config import TRAIN_TRANSACTION, TRAIN_IDENTITY, TRANSACTION_ID_COL

def load_data():
    """
    Load and merge transaction and identity datasets.
    
    Returns:
        pd.DataFrame: Merged dataset with all transactions
    """
    print("Loading transaction data...")
    trans_df = pd.read_csv(TRAIN_TRANSACTION)
    print(f"  Transaction shape: {trans_df.shape}")
    
    print("Loading identity data...")
    iden_df = pd.read_csv(TRAIN_IDENTITY)
    print(f"  Identity shape: {iden_df.shape}")
    
    print("Merging datasets...")
    # Left join to keep all transactions
    # Only ~24% of transactions have identity information
    merged_df = trans_df.merge(iden_df, on=TRANSACTION_ID_COL, how='left')
    print(f"  Merged shape: {merged_df.shape}")
    
    # Verify no rows were lost
    assert len(merged_df) == len(trans_df), "Row count mismatch after merge!"
    
    print(f"✓ Data loaded successfully: {merged_df.shape[0]:,} transactions")
    return merged_df
