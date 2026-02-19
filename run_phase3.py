"""
FraudMesh-RL — Phase 3: Reinforcement Learning Pipeline
========================================================
This script runs the complete Phase 3 pipeline:
  1. Loads Phase 1 data (or reruns if needed)
  2. Generates synthetic GNN fraud scores
  3. Creates the RL environment
  4. Trains the DQN agent
  5. Evaluates baselines
  6. Generates comparison plots & saves metrics
"""

import sys
import os
import numpy as np
import pandas as pd
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    TRAIN_TRANSACTION, TRAIN_IDENTITY, OUTPUT_DIR, PLOTS_DIR,
    METRICS_DIR, TARGET_COL, AMOUNT_COL, CARD_COL,
    MERCHANT_NODE_COL, MERCHANT_COLS, TIME_COL,
    RL_NUM_EPISODES, RL_MAX_STEPS_PER_EPISODE
)


def load_phase1_data():
    """Load and prepare Phase 1 data for the RL environment."""
    print("=" * 60)
    print("STEP 1: LOADING DATA")
    print("=" * 60)
    
    # Check for cached processed data from Phase 1
    cached_path = OUTPUT_DIR / 'processed_transactions.csv'
    if cached_path.exists():
        print(f"  Loading cached data from {cached_path}")
        df = pd.read_csv(cached_path)
        print(f"  Loaded {len(df):,} transactions")
    else:
        print("  No cached data found — loading raw data...")
        if not TRAIN_TRANSACTION.exists():
            print(f"  ERROR: {TRAIN_TRANSACTION} not found!")
            print("  Please download the IEEE-CIS Fraud Detection dataset from Kaggle")
            print("  and place train_transaction.csv in the data/ folder.")
            sys.exit(1)
        
        df = pd.read_csv(TRAIN_TRANSACTION)
        print(f"  Loaded {len(df):,} transactions from raw data")
        
        # Merge with identity if available
        if TRAIN_IDENTITY.exists():
            identity_df = pd.read_csv(TRAIN_IDENTITY)
            df = df.merge(identity_df, on='TransactionID', how='left')
            print(f"  Merged with identity data")
        
        # Create merchant node
        if all(col in df.columns for col in MERCHANT_COLS):
            df[MERCHANT_NODE_COL] = df[MERCHANT_COLS[0]].astype(str) + '_' + df[MERCHANT_COLS[1]].astype(str)
        else:
            df[MERCHANT_NODE_COL] = 'merchant_' + (df.index % 100).astype(str)
    
    # Ensure required columns exist
    required_cols = [TARGET_COL, AMOUNT_COL, CARD_COL, TIME_COL, MERCHANT_NODE_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  WARNING: Missing columns {missing}, creating synthetic versions...")
        if TARGET_COL not in df.columns:
            df[TARGET_COL] = np.random.choice([0, 1], size=len(df), p=[0.965, 0.035])
        if AMOUNT_COL not in df.columns:
            df[AMOUNT_COL] = np.random.lognormal(mean=3.5, sigma=1.5, size=len(df))
        if CARD_COL not in df.columns:
            df[CARD_COL] = np.random.randint(1000, 15000, size=len(df))
        if TIME_COL not in df.columns:
            df[TIME_COL] = np.sort(np.random.uniform(0, 86400 * 180, size=len(df)))
        if MERCHANT_NODE_COL not in df.columns:
            df[MERCHANT_NODE_COL] = 'merchant_' + (df.index % 100).astype(str)
    
    print(f"  Final dataset: {len(df):,} transactions")
    print(f"  Fraud rate: {df[TARGET_COL].mean()*100:.2f}%")
    print(f"  Columns used: {required_cols}")
    
    return df


def generate_fraud_scores(df):
    """
    Generate synthetic fraud scores mixing true labels with noise.
    In a real pipeline, these would come from the GNN in Phase 2.
    """
    print("\n" + "=" * 60)
    print("STEP 2: GENERATING FRAUD SCORES")
    print("=" * 60)
    
    # Check if Phase 2 GNN scores exist
    gnn_scores_path = METRICS_DIR / 'gnn_fraud_scores.csv'
    if gnn_scores_path.exists():
        print(f"  Loading GNN fraud scores from {gnn_scores_path}")
        scores_df = pd.read_csv(gnn_scores_path)
        if 'fraud_score' in scores_df.columns and len(scores_df) == len(df):
            fraud_scores = scores_df['fraud_score'].values
            print(f"  Loaded {len(fraud_scores):,} GNN scores")
            return fraud_scores
    
    print("  GNN scores not found — generating synthetic fraud scores")
    print("  (In production, these come from the Phase 2 GNN)")
    
    np.random.seed(42)
    labels = df[TARGET_COL].values.astype(np.float32)
    noise = np.random.normal(0, 0.15, size=len(labels))
    fraud_scores = np.clip(labels * 0.7 + 0.15 + noise, 0.0, 1.0)
    
    # Make fraud samples have higher scores and legit lower
    fraud_mask = labels == 1
    fraud_scores[fraud_mask] = np.clip(fraud_scores[fraud_mask] + 0.2, 0.0, 1.0)
    fraud_scores[~fraud_mask] = np.clip(fraud_scores[~fraud_mask] - 0.1, 0.0, 1.0)
    
    print(f"  Generated {len(fraud_scores):,} fraud scores")
    print(f"  Score range: [{fraud_scores.min():.4f}, {fraud_scores.max():.4f}]")
    print(f"  Mean score (fraud): {fraud_scores[fraud_mask].mean():.4f}")
    print(f"  Mean score (legit): {fraud_scores[~fraud_mask].mean():.4f}")
    
    return fraud_scores


def subsample_data(df, fraud_scores, max_samples=20000):
    """Subsample data to keep training fast while preserving fraud ratio."""
    if len(df) <= max_samples:
        return df, fraud_scores
    
    print(f"\n  Subsampling from {len(df):,} to {max_samples:,} transactions...")
    
    fraud_mask = df[TARGET_COL].values == 1
    n_fraud = int(fraud_mask.sum())
    n_legit = len(df) - n_fraud
    
    fraud_idx = np.where(fraud_mask)[0]
    legit_idx = np.where(~fraud_mask)[0]
    
    np.random.seed(42)
    
    # If fraud alone exceeds max_samples, subsample fraud too
    if n_fraud >= max_samples:
        fraud_sample = np.random.choice(fraud_idx, size=int(max_samples * 0.5), replace=False)
        n_legit_sample = max_samples - len(fraud_sample)
        legit_sample = np.random.choice(legit_idx, size=min(n_legit_sample, n_legit), replace=False)
        all_idx = np.concatenate([fraud_sample, legit_sample])
    else:
        n_legit_sample = max(0, min(max_samples - n_fraud, n_legit))
        if n_legit_sample > 0:
            legit_sample = np.random.choice(legit_idx, size=n_legit_sample, replace=False)
            all_idx = np.concatenate([fraud_idx, legit_sample])
        else:
            all_idx = fraud_idx[:max_samples]
    
    np.random.shuffle(all_idx)
    
    df_sub = df.iloc[all_idx].reset_index(drop=True)
    scores_sub = fraud_scores[all_idx]
    
    print(f"  Subsampled: {len(df_sub):,} transactions "
          f"({df_sub[TARGET_COL].mean()*100:.2f}% fraud)")
    
    return df_sub, scores_sub


def main():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " FraudMesh-RL — Phase 3: Reinforcement Learning ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # Step 1: Load data
    df = load_phase1_data()
    
    # Step 2: Generate fraud scores
    fraud_scores = generate_fraud_scores(df)
    
    # Step 3: Subsample for training speed
    df_sub, scores_sub = subsample_data(df, fraud_scores, max_samples=20000)
    
    # Step 4: Create environment
    print("\n" + "=" * 60)
    print("STEP 3: CREATING RL ENVIRONMENT")
    print("=" * 60)
    
    from src.rl_environment import FraudResponseEnv
    env = FraudResponseEnv(df_sub, scores_sub)
    
    # Step 5: Train DQN
    print("\n" + "=" * 60)
    print("STEP 4: TRAINING DQN AGENT")
    print("=" * 60)
    
    from src.rl_trainer import train_dqn, evaluate_baselines
    agent, training_metrics = train_dqn(env, num_episodes=RL_NUM_EPISODES)
    
    # Step 6: Evaluate baselines
    baseline_results = evaluate_baselines(env, num_episodes=50)
    
    # Step 7: Evaluate DQN
    from src.rl_evaluate import (
        evaluate_dqn, plot_training_curves,
        plot_action_distribution, plot_reward_comparison,
        plot_cost_analysis, save_combined_comparison
    )
    
    print("\n" + "=" * 60)
    print("STEP 5: EVALUATION & VISUALIZATION")
    print("=" * 60)
    
    dqn_metrics = evaluate_dqn(env, agent, num_episodes=50)
    
    # Step 8: Generate all plots
    plot_training_curves(training_metrics)
    plot_action_distribution(env, agent, baseline_results)
    plot_reward_comparison(dqn_metrics, baseline_results)
    plot_cost_analysis(dqn_metrics, baseline_results)
    
    # Step 9: Save combined comparison
    combined = save_combined_comparison(dqn_metrics, baseline_results, training_metrics)
    
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " Phase 3 Complete! ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print(f"\n  Outputs saved to: {OUTPUT_DIR}")
    print(f"  Plots: {PLOTS_DIR}")
    print(f"  Metrics: {METRICS_DIR}")
    print()


if __name__ == '__main__':
    main()
