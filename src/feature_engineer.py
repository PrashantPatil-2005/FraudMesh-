import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.config import (
    TARGET_COL, CARD_COL, TRANSACTION_ID_COL, TIME_COL, AMOUNT_COL,
    MERCHANT_COLS, MERCHANT_NODE_COL, MISSING_THRESHOLD, RANDOM_STATE
)

def engineer_features(df):
    """
    Engineer features and prepare data for modeling.
    
    Args:
        df: Raw merged DataFrame
        
    Returns:
        dict: Dictionary containing train/test data and metadata
    """
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # ========================================================================
    # STEP 1: Create Merchant Node Column
    # ========================================================================
    print("\n[Step 1/7] Creating merchant node column...")
    
    # Fill NaN in addr1 and ProductCD before concatenating
    addr1_filled = df['addr1'].fillna('unknown').astype(str)
    product_filled = df['ProductCD'].fillna('unknown').astype(str)
    
    df[MERCHANT_NODE_COL] = addr1_filled + '_' + product_filled
    
    n_merchants = df[MERCHANT_NODE_COL].nunique()
    print(f"  Created {n_merchants:,} unique merchant nodes")
    
    # ========================================================================
    # STEP 2: Time-Based Train/Test Split
    # ========================================================================
    print("\n[Step 2/7] Time-based train/test split...")
    
    # Sort by time
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    
    # Find 80th percentile
    split_point = df[TIME_COL].quantile(0.8)
    
    train_df = df[df[TIME_COL] <= split_point].copy()
    test_df = df[df[TIME_COL] > split_point].copy()
    
    print(f"  Split point: {split_point:,.0f} seconds")
    print(f"  Training set: {len(train_df):,} transactions")
    print(f"  Test set:     {len(test_df):,} transactions")
    
    # Check fraud rates
    train_fraud_rate = train_df[TARGET_COL].mean() * 100
    test_fraud_rate = test_df[TARGET_COL].mean() * 100
    
    print(f"  Training fraud rate: {train_fraud_rate:.4f}%")
    print(f"  Test fraud rate:     {test_fraud_rate:.4f}%")
    
    if abs(train_fraud_rate - test_fraud_rate) > 1.0:
        print(f"  WARNING: Fraud rates differ by {abs(train_fraud_rate - test_fraud_rate):.2f}%")
    
    # ========================================================================
    # STEP 3: Handle Missing Values
    # ========================================================================
    print("\n[Step 3/7] Handling missing values...")
    
    # Calculate missing percentage
    missing_pct = train_df.isnull().sum() / len(train_df) * 100
    
    # Identify columns to drop (>60% missing in training set)
    cols_to_drop = missing_pct[missing_pct > MISSING_THRESHOLD * 100].index.tolist()
    
    # Also drop ID and time columns that won't be used as features
    cols_to_drop.extend([TRANSACTION_ID_COL, TIME_COL])
    
    # Remove duplicates
    cols_to_drop = list(set(cols_to_drop))
    
    print(f"  Dropping {len(cols_to_drop)} columns (>{MISSING_THRESHOLD*100:.0f}% missing or metadata)")
    
    # Drop from both train and test
    train_df = train_df.drop(columns=cols_to_drop, errors='ignore')
    test_df = test_df.drop(columns=cols_to_drop, errors='ignore')
    
    # Separate numeric and categorical columns
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target from numeric list if present
    if TARGET_COL in numeric_cols:
        numeric_cols.remove(TARGET_COL)
    
    print(f"  Numeric columns: {len(numeric_cols)}")
    print(f"  Categorical columns: {len(categorical_cols)}")
    
    # Fill numeric missing values with TRAINING median
    print("  Filling numeric missing values with training median...")
    numeric_medians = train_df[numeric_cols].median()
    train_df[numeric_cols] = train_df[numeric_cols].fillna(numeric_medians)
    test_df[numeric_cols] = test_df[numeric_cols].fillna(numeric_medians)
    
    # Fill categorical missing values with 'missing'
    print("  Filling categorical missing values with 'missing'...")
    train_df[categorical_cols] = train_df[categorical_cols].fillna('missing')
    test_df[categorical_cols] = test_df[categorical_cols].fillna('missing')
    
    print(f"  Remaining columns: {len(train_df.columns)}")
    
    # ========================================================================
    # STEP 4: Encode Categorical Columns
    # ========================================================================
    print("\n[Step 4/7] Encoding categorical variables...")
    
    # Remove merchant_node and card1 from categorical encoding 
    # (they're for graph construction)
    cat_cols_to_encode = [c for c in categorical_cols 
                          if c not in [MERCHANT_NODE_COL, CARD_COL]]
    
    label_encoders = {}
    
    for col in cat_cols_to_encode:
        le = LabelEncoder()
        
        # Fit on training data
        le.fit(train_df[col].astype(str))
        
        # Transform training data
        train_df[col] = le.transform(train_df[col].astype(str))
        
        # Transform test data, handling unseen categories
        test_vals = test_df[col].astype(str)
        # Map unseen categories to -1
        test_df[col] = test_vals.map(lambda x: le.transform([x])[0] 
                                      if x in le.classes_ else -1)
        
        label_encoders[col] = le
    
    print(f"  Encoded {len(cat_cols_to_encode)} categorical columns")
    
    # ========================================================================
    # STEP 5: Select Features for Baseline Models
    # ========================================================================
    print("\n[Step 5/7] Selecting features for baseline models...")
    
    # Columns to exclude from feature set
    exclude_cols = [
        TARGET_COL,           # This is the target
        MERCHANT_NODE_COL,    # For graph construction
        CARD_COL              # For graph construction
    ]
    
    # Keep only columns present in dataframe
    exclude_cols = [c for c in exclude_cols if c in train_df.columns]
    
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    # Ensure all features are numeric
    train_features = train_df[feature_cols].copy()
    test_features = test_df[feature_cols].copy()
    
    # Convert any remaining object columns to numeric
    for col in train_features.columns:
        if train_features[col].dtype == 'object':
            train_features[col] = pd.to_numeric(train_features[col], errors='coerce').fillna(0)
            test_features[col] = pd.to_numeric(test_features[col], errors='coerce').fillna(0)
    
    # Extract targets
    y_train = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL]
    
    print(f"  Feature count: {len(feature_cols)}")
    print(f"  Training samples: {len(train_features)}")
    print(f"  Test samples: {len(test_features)}")
    
    # ========================================================================
    # STEP 6: Calculate Class Weight
    # ========================================================================
    print("\n[Step 6/7] Calculating class weight...")
    
    fraud_count = (y_train == 1).sum()
    non_fraud_count = (y_train == 0).sum()
    class_weight = non_fraud_count / fraud_count
    
    print(f"  Non-fraud count: {non_fraud_count:,}")
    print(f"  Fraud count: {fraud_count:,}")
    print(f"  Class weight: {class_weight:.2f}")
    
    # ========================================================================
    # STEP 7: Create Node Index Mappings for Graph
    # ========================================================================
    print("\n[Step 7/7] Creating node index mappings...")
    
    # Card nodes (from training set only)
    unique_cards = train_df[CARD_COL].dropna().unique()
    card_node_to_idx = {card: idx for idx, card in enumerate(unique_cards)}
    
    # Merchant nodes (from training set only)
    unique_merchants = train_df[MERCHANT_NODE_COL].dropna().unique()
    merchant_node_to_idx = {merchant: idx for idx, merchant in enumerate(unique_merchants)}
    
    print(f"  Card nodes: {len(card_node_to_idx):,}")
    print(f"  Merchant nodes: {len(merchant_node_to_idx):,}")
    
    # ========================================================================
    # Return Everything
    # ========================================================================
    print("\n✓ Feature engineering complete")
    
    return {
        'X_train': train_features,
        'X_test': test_features,
        'y_train': y_train,
        'y_test': y_test,
        'train_df': train_df,  # Full DataFrame with graph columns
        'test_df': test_df,    # Full DataFrame with graph columns
        'card_node_to_idx': card_node_to_idx,
        'merchant_node_to_idx': merchant_node_to_idx,
        'class_weight': class_weight,
        'feature_names': feature_cols
    }
