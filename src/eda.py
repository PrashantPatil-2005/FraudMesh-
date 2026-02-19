import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import (
    TARGET_COL, CARD_COL, AMOUNT_COL, TIME_COL,
    MERCHANT_COLS, PLOTS_DIR
)

def run_eda(df):
    """
    Perform exploratory data analysis and save visualizations.
    
    Args:
        df: Merged DataFrame from data_loader
    """
    print("\n" + "=" * 80)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    # ========================================================================
    # QUESTION 1: Class Imbalance
    # ========================================================================
    print("\n[Q1] CLASS DISTRIBUTION")
    print("-" * 80)
    
    fraud_counts = df[TARGET_COL].value_counts()
    fraud_pct = df[TARGET_COL].value_counts(normalize=True) * 100
    
    print(f"Non-fraud: {fraud_counts[0]:,} ({fraud_pct[0]:.2f}%)")
    print(f"Fraud:     {fraud_counts[1]:,} ({fraud_pct[1]:.2f}%)")
    print(f"Fraud rate: {fraud_pct[1]:.4f}%")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    fraud_counts.plot(kind='bar', ax=ax, color=['green', 'red'])
    ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Is Fraud (0=No, 1=Yes)')
    ax.set_ylabel('Count')
    ax.set_xticklabels(['Non-Fraud', 'Fraud'], rotation=0)
    for i, v in enumerate(fraud_counts):
        ax.text(i, v + 5000, f'{v:,}\n({fraud_pct[i]:.2f}%)', 
                ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'class_distribution.png', dpi=150)
    plt.close()
    print(f"✓ Saved: class_distribution.png")
    
    # ========================================================================
    # QUESTION 2: Missing Values
    # ========================================================================
    print("\n[Q2] MISSING VALUES")
    print("-" * 80)
    
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    top_missing = missing_pct[missing_pct > 0].head(30)
    
    print("Top 20 columns by missing percentage:")
    for col, pct in top_missing.head(20).items():
        print(f"  {col:30s}: {pct:6.2f}%")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    top_missing.plot(kind='barh', ax=ax, color='orange')
    ax.set_title('Top 30 Columns by Missing Values', fontsize=14, fontweight='bold')
    ax.set_xlabel('Missing Percentage (%)')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'missing_values.png', dpi=150)
    plt.close()
    print(f"✓ Saved: missing_values.png")
    
    # ========================================================================
    # QUESTION 3: Transaction Amounts
    # ========================================================================
    print("\n[Q3] TRANSACTION AMOUNT DISTRIBUTION")
    print("-" * 80)
    
    print(f"Statistics:")
    print(f"  Mean:   ${df[AMOUNT_COL].mean():.2f}")
    print(f"  Median: ${df[AMOUNT_COL].median():.2f}")
    print(f"  Std:    ${df[AMOUNT_COL].std():.2f}")
    print(f"  Min:    ${df[AMOUNT_COL].min():.2f}")
    print(f"  Max:    ${df[AMOUNT_COL].max():.2f}")
    
    # Compare fraud vs non-fraud
    fraud_amounts = df[df[TARGET_COL] == 1][AMOUNT_COL]
    non_fraud_amounts = df[df[TARGET_COL] == 0][AMOUNT_COL]
    
    print(f"\nFraud transactions:")
    print(f"  Mean:   ${fraud_amounts.mean():.2f}")
    print(f"  Median: ${fraud_amounts.median():.2f}")
    
    print(f"\nNon-fraud transactions:")
    print(f"  Mean:   ${non_fraud_amounts.mean():.2f}")
    print(f"  Median: ${non_fraud_amounts.median():.2f}")
    
    # Plot (log scale due to heavy right skew)
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.logspace(0, 5, 50)
    ax.hist(non_fraud_amounts, bins=bins, alpha=0.5, label='Non-Fraud', color='green')
    ax.hist(fraud_amounts, bins=bins, alpha=0.5, label='Fraud', color='red')
    ax.set_xscale('log')
    ax.set_xlabel('Transaction Amount ($, log scale)')
    ax.set_ylabel('Frequency')
    ax.set_title('Transaction Amount Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'amount_distribution.png', dpi=150)
    plt.close()
    print(f"✓ Saved: amount_distribution.png")
    
    # ========================================================================
    # QUESTION 4: Card Columns
    # ========================================================================
    print("\n[Q4] CARD COLUMNS")
    print("-" * 80)
    
    print(f"\ncard1 (card identifier):")
    print(f"  Unique values: {df['card1'].nunique():,}")
    print(f"  Top 10 values:")
    print(df['card1'].value_counts().head(10))
    
    if 'card4' in df.columns:
        print(f"\ncard4 (card type):")
        print(df['card4'].value_counts(dropna=False))
        
        print(f"\nFraud rate by card4:")
        fraud_by_card4 = df.groupby('card4', dropna=False)[TARGET_COL].mean() * 100
        for card_type, rate in fraud_by_card4.items():
            print(f"  {card_type}: {rate:.2f}%")
    
    if 'card6' in df.columns:
        print(f"\ncard6 (debit/credit):")
        print(df['card6'].value_counts(dropna=False))
        
        print(f"\nFraud rate by card6:")
        fraud_by_card6 = df.groupby('card6', dropna=False)[TARGET_COL].mean() * 100
        for card_cat, rate in fraud_by_card6.items():
            print(f"  {card_cat}: {rate:.2f}%")
    
    # ========================================================================
    # QUESTION 5: Finding the Merchant Node
    # ========================================================================
    print("\n[Q5] MERCHANT NODE CONSTRUCTION")
    print("-" * 80)
    
    print("The IEEE-CIS dataset does NOT have a direct 'merchant_id' column.")
    print("Investigating candidate columns...\n")
    
    # addr1 - billing address
    if 'addr1' in df.columns:
        print(f"addr1 (billing address region):")
        print(f"  Unique values: {df['addr1'].nunique():,}")
        print(f"  Missing: {df['addr1'].isnull().sum() / len(df) * 100:.2f}%")
        print(f"  Note: This represents BUYER location, not merchant")
    
    # P_emaildomain - payment email
    if 'P_emaildomain' in df.columns:
        print(f"\nP_emaildomain (payment email domain):")
        print(f"  Unique values: {df['P_emaildomain'].nunique():,}")
        print(f"  Missing: {df['P_emaildomain'].isnull().sum() / len(df) * 100:.2f}%")
        print(f"  Top 10:")
        print(df['P_emaildomain'].value_counts().head(10))
        print(f"  Note: This is BUYER email, not merchant")
    
    # R_emaildomain - recipient email
    if 'R_emaildomain' in df.columns:
        print(f"\nR_emaildomain (recipient email domain):")
        print(f"  Unique values: {df['R_emaildomain'].nunique():,}")
        print(f"  Missing: {df['R_emaildomain'].isnull().sum() / len(df) * 100:.2f}%")
        print(f"  Note: Closer to merchant-side, but high missing rate")
    
    # ProductCD - product code
    if 'ProductCD' in df.columns:
        print(f"\nProductCD (product code):")
        print(df['ProductCD'].value_counts(dropna=False))
        print(f"  Note: Only {df['ProductCD'].nunique()} unique values - too few for graph nodes alone")
    
    # Combination approach
    print(f"\n{'='*80}")
    print("DECISION: Create synthetic merchant node")
    print(f"{'='*80}")
    print(f"  Combine: addr1 + ProductCD")
    print(f"  Rationale: Groups transactions by billing region AND product type")
    print(f"  This is a PROXY for merchant, not a real merchant ID")
    
    # Create temporary merchant node to count
    temp_merchant = df['addr1'].fillna('unknown').astype(str) + '_' + df['ProductCD'].fillna('unknown').astype(str)
    print(f"\nMerchant nodes that would be created: {temp_merchant.nunique():,}")
    
    # ========================================================================
    # QUESTION 6: Time Column
    # ========================================================================
    print("\n[Q6] TIME COLUMN")
    print("-" * 80)
    
    print(f"TransactionDT statistics:")
    print(f"  Min: {df[TIME_COL].min():,.0f} seconds")
    print(f"  Max: {df[TIME_COL].max():,.0f} seconds")
    
    time_range_seconds = df[TIME_COL].max() - df[TIME_COL].min()
    time_range_days = time_range_seconds / (24 * 3600)
    print(f"  Range: {time_range_seconds:,.0f} seconds ({time_range_days:.1f} days)")
    
    # Calculate 80% split point
    split_point = df[TIME_COL].quantile(0.8)
    print(f"\n80% time split point: {split_point:,.0f} seconds")
    print(f"  Transactions before split: {(df[TIME_COL] <= split_point).sum():,}")
    print(f"  Transactions after split:  {(df[TIME_COL] > split_point).sum():,}")
    
    print("\nIMPORTANT: Using TIME-BASED split to prevent data leakage")
    print("  Random split would leak future fraud patterns into training")
    
    # ========================================================================
    # QUESTION 7: Correlation with Fraud
    # ========================================================================
    print("\n[Q7] FEATURE CORRELATION WITH FRAUD")
    print("-" * 80)
    
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Remove target column
    numeric_cols = [c for c in numeric_cols if c != TARGET_COL]
    
    # Calculate correlations
    correlations = {}
    for col in numeric_cols:
        if df[col].notna().sum() > 0:  # Only if column has non-null values
            corr = df[[col, TARGET_COL]].corr().iloc[0, 1]
            if not np.isnan(corr):
                correlations[col] = abs(corr)
    
    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 15 features by absolute correlation with fraud:")
    for i, (col, corr) in enumerate(sorted_corr[:15], 1):
        print(f"  {i:2d}. {col:30s}: {corr:.4f}")
    
    # Plot
    if len(sorted_corr) >= 20:
        top_20 = dict(sorted_corr[:20])
        fig, ax = plt.subplots(figsize=(10, 8))
        pd.Series(top_20).sort_values().plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title('Top 20 Features by Correlation with Fraud', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Absolute Correlation')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'fraud_correlations.png', dpi=150)
        plt.close()
        print(f"✓ Saved: fraud_correlations.png")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("EDA SUMMARY")
    print("=" * 80)
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns):,}")
    print(f"Fraud rate: {fraud_pct[1]:.4f}%")
    print(f"Card node column: {CARD_COL}")
    print(f"Merchant node approach: Combine {' + '.join(MERCHANT_COLS)}")
    print(f"Time split (80%): {split_point:,.0f} seconds")
    print(f"Columns with >60% missing: {(missing_pct > 60).sum()}")
    print("=" * 80)
