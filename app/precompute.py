"""
Pre-compute all demo data for the Streamlit app.

Run this ONCE locally before deploying:
    python -m app.precompute

This generates lightweight files that the Streamlit app loads instantly.
"""

import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Project imports
from src.config import (
    TRAIN_TRANSACTION, TRAIN_IDENTITY, TRANSACTION_ID_COL,
    TARGET_COL, CARD_COL, AMOUNT_COL, TIME_COL,
    MERCHANT_NODE_COL, RL_ACTIONS, RL_NUM_ACTIONS, RL_REWARDS,
    METRICS_DIR, GNN_CHECKPOINT_PATH, RL_CHECKPOINT_PATH
)

# Output directory
DEMO_DATA_DIR = Path(__file__).parent / 'demo_data'
DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)

NUM_DEMO_SAMPLES = 500


def load_and_prepare_data():
    """Load raw data and prepare sample transactions."""
    print("[1/7] Loading raw data...")
    
    trans = pd.read_csv(TRAIN_TRANSACTION)
    iden = pd.read_csv(TRAIN_IDENTITY)
    df = trans.merge(iden, on=TRANSACTION_ID_COL, how='left')
    
    # Create merchant node
    df[MERCHANT_NODE_COL] = (
        df['addr1'].fillna('unknown').astype(str) + '_' +
        df['ProductCD'].fillna('unknown').astype(str)
    )
    
    # Time-based split (same as Phase 1)
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    split_point = df[TIME_COL].quantile(0.8)
    
    train_df = df[df[TIME_COL] <= split_point].copy()
    test_df = df[df[TIME_COL] > split_point].copy()
    
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")
    
    return df, train_df, test_df


def sample_transactions(test_df):
    """Pick a balanced sample of fraud and non-fraud transactions."""
    print("[2/7] Sampling transactions...")
    
    fraud = test_df[test_df[TARGET_COL] == 1]
    legit = test_df[test_df[TARGET_COL] == 0]
    
    # Take up to half fraud, half legit
    n_fraud = min(len(fraud), NUM_DEMO_SAMPLES // 4)  # ~25% fraud for interesting demo
    n_legit = NUM_DEMO_SAMPLES - n_fraud
    
    fraud_sample = fraud.sample(n=n_fraud, random_state=42)
    legit_sample = legit.sample(n=n_legit, random_state=42)
    
    samples = pd.concat([fraud_sample, legit_sample]).sample(frac=1, random_state=42)
    samples = samples.reset_index(drop=True)
    
    print(f"  Sampled {len(samples)} transactions")
    print(f"  Fraud: {(samples[TARGET_COL] == 1).sum()}")
    print(f"  Legit: {(samples[TARGET_COL] == 0).sum()}")
    
    return samples


def generate_fraud_scores(samples, train_df):
    """Generate GNN or RF fraud scores for sample transactions."""
    print("[3/7] Generating fraud scores...")
    
    scores = np.zeros(len(samples), dtype=np.float32)
    source = "unknown"
    
    # Try GNN first
    try:
        import torch
        from src.gnn_model import FraudGAT
        from src.config import GNN_HIDDEN_DIM, GNN_NUM_HEADS, GNN_NUM_LAYERS, GNN_DROPOUT
        
        if GNN_CHECKPOINT_PATH.exists():
            print("  Attempting GNN scoring...")
            
            # We need to build the graph and get card-level scores
            # Then map them to transactions
            from src.feature_engineer import engineer_features
            
            # Merge train and test for feature engineering
            full_df = pd.concat([train_df, samples]).reset_index(drop=True)
            
            # Build card-level fraud scores using the graph
            from src.pyg_converter import convert_to_pyg
            
            data_dict = engineer_features(full_df)
            
            pyg_data = convert_to_pyg(
                train_df=data_dict['train_df'],
                test_df=data_dict['test_df'],
                card_node_to_idx=data_dict['card_node_to_idx'],
                merchant_node_to_idx=data_dict['merchant_node_to_idx']
            )
            
            card_feature_dim = pyg_data['card'].x.shape[1]
            merchant_feature_dim = pyg_data['merchant'].x.shape[1]
            
            model = FraudGAT(
                card_feature_dim=card_feature_dim,
                merchant_feature_dim=merchant_feature_dim,
                hidden_dim=GNN_HIDDEN_DIM,
                num_heads=GNN_NUM_HEADS,
                num_layers=GNN_NUM_LAYERS,
                dropout=GNN_DROPOUT
            )
            
            checkpoint = torch.load(GNN_CHECKPOINT_PATH, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            with torch.no_grad():
                x_dict = {
                    'card': pyg_data['card'].x,
                    'merchant': pyg_data['merchant'].x
                }
                edge_index_dict = {
                    ('card', 'transacts_at', 'merchant'):
                        pyg_data['card', 'transacts_at', 'merchant'].edge_index,
                    ('merchant', 'rev_transacts_at', 'card'):
                        pyg_data['merchant', 'rev_transacts_at', 'card'].edge_index
                }
                logits = model(x_dict, edge_index_dict)
                card_probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            
            card_node_to_idx = data_dict['card_node_to_idx']
            
            for i in range(len(samples)):
                card_id = samples.iloc[i][CARD_COL]
                if card_id in card_node_to_idx:
                    idx = card_node_to_idx[card_id]
                    if idx < len(card_probs):
                        scores[i] = card_probs[idx]
                    else:
                        scores[i] = 0.5
                else:
                    scores[i] = 0.5
            
            source = "GNN (GAT)"
            print(f"  ✓ GNN scores generated")
        else:
            raise FileNotFoundError("No GNN checkpoint")
            
    except Exception as e:
        print(f"  GNN failed: {e}")
        print("  Falling back to Random Forest...")
        
        from sklearn.ensemble import RandomForestClassifier
        
        feature_cols = ['TransactionAmt', 'card1', 'card2', 'card3', 'card5',
                       'addr1', 'addr2', 'dist1', 'dist2']
        feature_cols = [c for c in feature_cols if c in train_df.columns]
        
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[TARGET_COL]
        X_sample = samples[feature_cols].fillna(0)
        
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42,
            class_weight='balanced', n_jobs=-1
        )
        rf.fit(X_train, y_train)
        scores = rf.predict_proba(X_sample)[:, 1].astype(np.float32)
        source = "Random Forest"
        print(f"  ✓ RF scores generated")
    
    print(f"  Source: {source}")
    print(f"  Range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    return scores, source


def build_state_vectors(samples, fraud_scores, full_df):
    """Build 8-dim RL state vectors for all sample transactions."""
    print("[4/7] Building state vectors...")
    
    amounts = full_df[AMOUNT_COL].values
    amt_mean = amounts.mean()
    amt_std = amounts.std() if amounts.std() > 0 else 1
    amt_max = amounts.max()
    amt_90 = np.percentile(amounts, 90)
    log_max = np.log1p(amt_max)
    
    card_counts = full_df[CARD_COL].value_counts().to_dict()
    merchant_fraud_rates = full_df.groupby(MERCHANT_NODE_COL)[TARGET_COL].mean().to_dict()
    overall_fraud_rate = full_df[TARGET_COL].mean()
    
    states = np.zeros((len(samples), 8), dtype=np.float32)
    
    for i in range(len(samples)):
        row = samples.iloc[i]
        amount = row[AMOUNT_COL]
        
        # Feature 1: fraud_score
        states[i, 0] = np.clip(fraud_scores[i], 0, 1)
        
        # Feature 2: amount_normalized (log transform)
        log_amt = np.log1p(amount)
        states[i, 1] = log_amt / log_max if log_max > 0 else 0
        
        # Feature 3: hour_of_day
        time_val = row[TIME_COL] if not pd.isna(row[TIME_COL]) else 0
        hour = (time_val % 86400) / 3600.0
        states[i, 2] = hour / 24.0
        
        # Feature 4: is_high_amount
        states[i, 3] = 1.0 if amount > amt_90 else 0.0
        
        # Feature 5: card_frequency
        card_freq = card_counts.get(row[CARD_COL], 1)
        states[i, 4] = min(card_freq / 100.0, 1.0)
        
        # Feature 6: merchant_risk
        merchant_risk = merchant_fraud_rates.get(row[MERCHANT_NODE_COL], overall_fraud_rate)
        states[i, 5] = np.clip(merchant_risk, 0, 1)
        
        # Feature 7: amount_zscore
        zscore = (amount - amt_mean) / amt_std
        states[i, 6] = np.clip((zscore + 3) / 6, 0, 1)
        
        # Feature 8: velocity
        states[i, 7] = min(card_freq / 50.0, 1.0)
    
    print(f"  ✓ Built {len(states)} state vectors")
    return states


def generate_rl_decisions(states):
    """Run DQN agent on all states to get actions and Q-values."""
    print("[5/7] Generating RL decisions...")
    
    actions = np.zeros(len(states), dtype=np.int32)
    q_values_all = np.zeros((len(states), RL_NUM_ACTIONS), dtype=np.float32)
    method = "rule-based"
    
    # Try loading DQN
    try:
        import torch
        from src.dqn_model import QNetwork
        from src.config import RL_STATE_DIM, DQN_HIDDEN_DIM
        
        if RL_CHECKPOINT_PATH.exists():
            model = QNetwork(RL_STATE_DIM, RL_NUM_ACTIONS, DQN_HIDDEN_DIM)
            checkpoint = torch.load(RL_CHECKPOINT_PATH, weights_only=False)
            model.load_state_dict(checkpoint['q_network_state_dict'])
            model.eval()
            
            with torch.no_grad():
                states_tensor = torch.tensor(states, dtype=torch.float32)
                q_vals = model(states_tensor).numpy()
            
            q_values_all = q_vals
            actions = np.argmax(q_vals, axis=1)
            method = "DQN"
            print(f"  ✓ DQN decisions generated")
        else:
            raise FileNotFoundError("No DQN checkpoint")
            
    except Exception as e:
        print(f"  DQN failed: {e}")
        print("  Using rule-based policy...")
        
        for i in range(len(states)):
            fraud_score = states[i, 0]
            if fraud_score < 0.2:
                actions[i] = 0
            elif fraud_score < 0.4:
                actions[i] = 1
            elif fraud_score < 0.6:
                actions[i] = 3
            elif fraud_score < 0.8:
                actions[i] = 2
            else:
                actions[i] = 4
            
            q_values_all[i, actions[i]] = 1.0
        
        method = "rule-based"
        print(f"  ✓ Rule-based decisions generated")
    
    print(f"  Method: {method}")
    print(f"  Action distribution:")
    for a in range(RL_NUM_ACTIONS):
        count = (actions == a).sum()
        print(f"    {RL_ACTIONS[a]:>15}: {count}")
    
    return actions, q_values_all, method


def compute_rewards(samples, actions):
    """Compute rewards for each action-label combination."""
    print("[6/7] Computing rewards...")
    
    rewards = np.zeros(len(samples), dtype=np.float32)
    correct = np.zeros(len(samples), dtype=np.bool_)
    
    for i in range(len(samples)):
        true_label = samples.iloc[i][TARGET_COL]
        action = actions[i]
        action_name = RL_ACTIONS[action]
        
        if true_label == 1:
            key = f"{action_name}_FRAUD"
            correct[i] = action != 0
        else:
            key = f"{action_name}_LEGIT"
            correct[i] = action == 0
        
        rewards[i] = RL_REWARDS.get(key, 0.0)
    
    print(f"  Total reward: {rewards.sum():.1f}")
    print(f"  Correct decisions: {correct.sum()}/{len(correct)}")
    
    return rewards, correct


def build_graph_neighborhoods(samples, train_df):
    """Build small graph neighborhoods for visualization."""
    print("[7/7] Building graph neighborhoods...")
    
    neighborhoods = []
    
    for i in range(min(len(samples), 50)):  # Only first 50 for speed
        row = samples.iloc[i]
        card_id = row[CARD_COL]
        merchant = row[MERCHANT_NODE_COL]
        
        # Find other cards at same merchant
        same_merchant = train_df[train_df[MERCHANT_NODE_COL] == merchant]
        connected_cards = same_merchant[CARD_COL].value_counts().head(10).to_dict()
        
        # Find fraud rate at merchant
        merchant_fraud_rate = same_merchant[TARGET_COL].mean() if len(same_merchant) > 0 else 0
        
        # Find other merchants used by same card
        same_card = train_df[train_df[CARD_COL] == card_id]
        connected_merchants = same_card[MERCHANT_NODE_COL].value_counts().head(5).to_dict()
        
        neighborhoods.append({
            'card_id': int(card_id) if not pd.isna(card_id) else 0,
            'merchant': str(merchant)[:50],
            'merchant_fraud_rate': float(merchant_fraud_rate),
            'merchant_total_txs': int(len(same_merchant)),
            'connected_cards': {str(k): int(v) for k, v in connected_cards.items()},
            'connected_merchants': {str(k)[:30]: int(v) for k, v in connected_merchants.items()},
            'card_total_txs': int(len(same_card)),
            'card_fraud_rate': float(same_card[TARGET_COL].mean()) if len(same_card) > 0 else 0
        })
    
    # Pad remaining with empty
    for i in range(len(neighborhoods), len(samples)):
        neighborhoods.append({
            'card_id': 0, 'merchant': 'unknown',
            'merchant_fraud_rate': 0, 'merchant_total_txs': 0,
            'connected_cards': {}, 'connected_merchants': {},
            'card_total_txs': 0, 'card_fraud_rate': 0
        })
    
    print(f"  ✓ Built {min(len(samples), 50)} neighborhoods")
    return neighborhoods


def save_demo_data(samples, fraud_scores, score_source, states,
                   actions, q_values, rl_method, rewards, correct,
                   neighborhoods):
    """Save all pre-computed data to lightweight files."""
    print("\nSaving demo data...")
    
    # 1. Transaction details (lightweight CSV)
    demo_df = pd.DataFrame({
        'transaction_id': range(len(samples)),
        'amount': samples[AMOUNT_COL].values,
        'card_id': samples[CARD_COL].values,
        'merchant': samples[MERCHANT_NODE_COL].values,
        'product_cd': samples['ProductCD'].values if 'ProductCD' in samples.columns else 'unknown',
        'card_type': samples['card4'].values if 'card4' in samples.columns else 'unknown',
        'card_category': samples['card6'].values if 'card6' in samples.columns else 'unknown',
        'true_label': samples[TARGET_COL].values,
        'fraud_score': fraud_scores,
        'rl_action': actions,
        'rl_action_name': [RL_ACTIONS[a] for a in actions],
        'reward': rewards,
        'correct': correct
    })
    demo_df.to_csv(DEMO_DATA_DIR / 'transactions.csv', index=False)
    print(f"  ✓ transactions.csv ({len(demo_df)} rows)")
    
    # 2. State vectors (numpy)
    np.save(DEMO_DATA_DIR / 'states.npy', states)
    print(f"  ✓ states.npy")
    
    # 3. Q-values (numpy)
    np.save(DEMO_DATA_DIR / 'q_values.npy', q_values)
    print(f"  ✓ q_values.npy")
    
    # 4. Neighborhoods (JSON)
    with open(DEMO_DATA_DIR / 'neighborhoods.json', 'w') as f:
        json.dump(neighborhoods, f)
    print(f"  ✓ neighborhoods.json")
    
    # 5. Metadata
    metadata = {
        'num_transactions': len(samples),
        'fraud_count': int((samples[TARGET_COL] == 1).sum()),
        'legit_count': int((samples[TARGET_COL] == 0).sum()),
        'fraud_score_source': score_source,
        'rl_method': rl_method,
        'total_reward': float(rewards.sum()),
        'accuracy': float(correct.mean()),
        'action_distribution': {
            RL_ACTIONS[a]: int((actions == a).sum()) for a in range(RL_NUM_ACTIONS)
        },
        'reward_structure': RL_REWARDS,
        'actions': RL_ACTIONS
    }
    
    with open(DEMO_DATA_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ metadata.json")
    
    # 6. Copy phase metrics
    for phase_file in ['phase1_baselines.json', 'phase2_gnn.json',
                       'phase3_combined_comparison.json']:
        src = METRICS_DIR / phase_file
        dst = DEMO_DATA_DIR / phase_file
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ {phase_file}")
        else:
            print(f"  ○ {phase_file} (not found, skipping)")
    
    # 7. Copy plots that exist
    plots_src = Path('outputs/plots')
    plots_dst = DEMO_DATA_DIR / 'plots'
    plots_dst.mkdir(exist_ok=True)
    
    important_plots = [
        'class_distribution.png',
        'sample_subgraph.png',
        'roc_curves_phase1.png',
        'gnn_training_curves.png',
        'model_comparison_auc.png',
        'rl_training_curves.png',
        'rl_reward_comparison.png',
        'rl_action_distribution.png',
        'rl_cost_analysis.png',
        'demo_full_pipeline.png'
    ]
    
    for plot in important_plots:
        src = plots_src / plot
        if src.exists():
            shutil.copy2(src, plots_dst / plot)
            print(f"  ✓ plots/{plot}")
    
    print("\n✓ All demo data saved to app/demo_data/")


def main():
    print("\n" + "=" * 60)
    print("PRE-COMPUTING DEMO DATA")
    print("=" * 60 + "\n")
    
    full_df, train_df, test_df = load_and_prepare_data()
    samples = sample_transactions(test_df)
    fraud_scores, score_source = generate_fraud_scores(samples, train_df)
    states = build_state_vectors(samples, fraud_scores, full_df)
    actions, q_values, rl_method = generate_rl_decisions(states)
    rewards, correct = compute_rewards(samples, actions)
    neighborhoods = build_graph_neighborhoods(samples, train_df)
    
    save_demo_data(
        samples, fraud_scores, score_source, states,
        actions, q_values, rl_method, rewards, correct,
        neighborhoods
    )
    
    print("\n" + "=" * 60)
    print("PRE-COMPUTATION COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Test locally: streamlit run streamlit_app.py")
    print("  2. Deploy to Streamlit Cloud or Hugging Face Spaces")
    print()


if __name__ == '__main__':
    main()
