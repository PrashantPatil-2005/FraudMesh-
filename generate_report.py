"""
FraudMesh-RL — Report Generator

Reads all saved metrics and generates a comprehensive project report.

Run: python generate_report.py
Output: outputs/PROJECT_REPORT.md
"""

import json
from pathlib import Path
from datetime import datetime
from src.config import METRICS_DIR, OUTPUT_DIR


def load_json_safe(path):
    """Load JSON file, return empty dict if not found."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def generate_report():
    """Generate the full project report."""
    
    print("Generating project report...")
    
    # Load all metrics
    phase1 = load_json_safe(METRICS_DIR / 'phase1_baselines.json')
    phase2 = load_json_safe(METRICS_DIR / 'phase2_gnn.json')
    phase3 = load_json_safe(METRICS_DIR / 'phase3_combined_comparison.json')
    rl_training = load_json_safe(METRICS_DIR / 'rl_training_metrics.json')
    rl_baselines = load_json_safe(METRICS_DIR / 'rl_baseline_metrics.json')
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    report = []
    
    # ================================================================
    # Title
    # ================================================================
    report.append("# FraudMesh-RL: Project Report")
    report.append("")
    report.append(f"**Generated:** {timestamp}")
    report.append("")
    report.append("---")
    report.append("")
    
    # ================================================================
    # Executive Summary
    # ================================================================
    report.append("## Executive Summary")
    report.append("")
    report.append("FraudMesh-RL is an end-to-end financial fraud detection and response system")
    report.append("that combines three machine learning paradigms:")
    report.append("")
    report.append("1. **Traditional ML** (Logistic Regression, Random Forest) — baseline detection")
    report.append("2. **Graph Neural Networks** (Graph Attention Network) — pattern detection across transaction networks")
    report.append("3. **Reinforcement Learning** (Deep Q-Network) — autonomous response decisions")
    report.append("")
    report.append("The system was trained and evaluated on the IEEE-CIS Fraud Detection dataset")
    report.append("containing 590,540 real-world transactions.")
    report.append("")
    report.append("---")
    report.append("")
    
    # ================================================================
    # Phase 1 Results
    # ================================================================
    report.append("## Phase 1: Baseline Models")
    report.append("")
    
    metrics_to_show = [
        ('AUC-ROC', 'auc'),
        ('F1 Score', 'f1'),
        ('Precision', 'precision'),
        ('Recall', 'recall'),
        ('False Positive Rate', 'false_positive_rate'),
        ('False Negative Rate', 'false_negative_rate')
    ]
    
    if phase1:
        report.append("### Results")
        report.append("")
        report.append("| Metric | Logistic Regression | Random Forest |")
        report.append("|--------|-------------------|---------------|")
        
        for display_name, key in metrics_to_show:
            lr_val = phase1.get('logistic_regression', {}).get(key, 'N/A')
            rf_val = phase1.get('random_forest', {}).get(key, 'N/A')
            
            if isinstance(lr_val, (int, float)):
                lr_str = f"{lr_val:.4f}"
            else:
                lr_str = str(lr_val)
            
            if isinstance(rf_val, (int, float)):
                rf_str = f"{rf_val:.4f}"
            else:
                rf_str = str(rf_val)
            
            report.append(f"| {display_name} | {lr_str} | {rf_str} |")
        
        report.append("")
        report.append("### Key Findings")
        report.append("")
        report.append("- Dataset is highly imbalanced (~3.5% fraud rate)")
        report.append("- Time-based train/test split used to prevent data leakage")
        report.append("- `class_weight='balanced'` used to handle imbalance")
        report.append("- Random Forest outperformed Logistic Regression on most metrics")
        report.append("")
    else:
        report.append("*Phase 1 metrics not available. Run `python run_phase1.py` first.*")
        report.append("")
    
    report.append("---")
    report.append("")
    
    # ================================================================
    # Phase 2 Results
    # ================================================================
    report.append("## Phase 2: Graph Neural Network")
    report.append("")
    
    if phase2:
        gnn = phase2.get('gnn_gat', phase2.get('gnn', {}))
        training = phase2.get('training_info', {})
        
        report.append("### Architecture")
        report.append("")
        report.append("- **Model:** Graph Attention Network (GAT)")
        report.append("- **Graph Type:** Heterogeneous bipartite (card ↔ merchant)")
        
        if training.get('total_parameters'):
            report.append(f"- **Parameters:** {training['total_parameters']:,}")
        if training.get('best_epoch'):
            report.append(f"- **Best Epoch:** {training['best_epoch']}")
        if training.get('total_epochs'):
            report.append(f"- **Total Epochs:** {training['total_epochs']}")
        report.append("")
        
        report.append("### Results")
        report.append("")
        report.append("| Metric | GNN (GAT) |")
        report.append("|--------|-----------|")
        
        for display_name, key in metrics_to_show:
            val = gnn.get(key, 'N/A')
            if isinstance(val, (int, float)):
                val_str = f"{val:.4f}"
            else:
                val_str = str(val)
            report.append(f"| {display_name} | {val_str} |")
        
        report.append("")
        
        report.append("### Why GNN Over Traditional ML")
        report.append("")
        report.append("Traditional models evaluate each transaction independently.")
        report.append("The GNN sees the **neighborhood structure** — if a merchant")
        report.append("is connected to multiple fraudulent cards, that signal propagates")
        report.append("through the graph to flag other transactions at that merchant.")
        report.append("")
    else:
        report.append("*Phase 2 metrics not available. Run `python run_phase2.py` first.*")
        report.append("")
    
    report.append("---")
    report.append("")
    
    # ================================================================
    # Phase 3 Results
    # ================================================================
    report.append("## Phase 3: Reinforcement Learning Agent")
    report.append("")
    
    report.append("### Action Space")
    report.append("")
    report.append("| Action | Description |")
    report.append("|--------|-------------|")
    report.append("| APPROVE | Let transaction through |")
    report.append("| SOFT_BLOCK | Request 2FA verification |")
    report.append("| HARD_BLOCK | Decline the transaction |")
    report.append("| FLAG_REVIEW | Send to human analyst |")
    report.append("| FREEZE_ACCOUNT | Freeze the entire account |")
    report.append("")
    
    if rl_training:
        report.append("### Training")
        report.append("")
        report.append(f"- **Total Episodes:** {len(rl_training.get('episode_rewards', []))}")
        report.append(f"- **Best Avg Reward:** {rl_training.get('best_avg_reward', 'N/A'):.2f}")
        report.append(f"- **Training Time:** {rl_training.get('total_time', 0):.0f}s")
        report.append(f"- **Total Training Steps:** {rl_training.get('total_training_steps', 'N/A'):,}")
        report.append("")
    
    if phase3:
        rl_results = phase3.get('phase3_rl_results', {})
        dqn = rl_results.get('dqn_agent', {})
        baselines = rl_results.get('baselines', {})
        
        report.append("### Policy Comparison")
        report.append("")
        report.append("| Policy | Avg Reward | Fraud Catch Rate | False Positive Rate |")
        report.append("|--------|-----------|-----------------|-------------------|")
        
        if dqn:
            avg_r = dqn.get('avg_reward', 'N/A')
            fcr = dqn.get('avg_fraud_catch_rate', 'N/A')
            fpr_val = dqn.get('avg_false_positive_rate', 'N/A')
            
            avg_r_str = f"{avg_r:.2f}" if isinstance(avg_r, (int, float)) else str(avg_r)
            fcr_str = f"{fcr:.2%}" if isinstance(fcr, (int, float)) else str(fcr)
            fpr_str = f"{fpr_val:.2%}" if isinstance(fpr_val, (int, float)) else str(fpr_val)
            
            report.append(f"| **DQN Agent** | **{avg_r_str}** | **{fcr_str}** | **{fpr_str}** |")
        
        for policy_name, metrics in baselines.items():
            if isinstance(metrics, dict):
                avg_r = metrics.get('avg_reward', 'N/A')
                fcr = metrics.get('avg_fraud_catch_rate', 'N/A')
                fpr_val = metrics.get('avg_false_positive_rate', 'N/A')
                
                avg_r_str = f"{avg_r:.2f}" if isinstance(avg_r, (int, float)) else str(avg_r)
                fcr_str = f"{fcr:.2%}" if isinstance(fcr, (int, float)) else str(fcr)
                fpr_str = f"{fpr_val:.2%}" if isinstance(fpr_val, (int, float)) else str(fpr_val)
                
                report.append(f"| {policy_name} | {avg_r_str} | {fcr_str} | {fpr_str} |")
        
        report.append("")
    elif rl_baselines:
        report.append("### Baseline Results")
        report.append("")
        report.append("| Policy | Avg Reward | Fraud Catch Rate | False Positive Rate |")
        report.append("|--------|-----------|-----------------|-------------------|")
        
        for policy_name, metrics in rl_baselines.items():
            if isinstance(metrics, dict):
                avg_r = metrics.get('avg_reward', 'N/A')
                fcr = metrics.get('avg_fraud_catch_rate', 'N/A')
                fpr_val = metrics.get('avg_false_positive_rate', 'N/A')
                
                avg_r_str = f"{avg_r:.2f}" if isinstance(avg_r, (int, float)) else str(avg_r)
                fcr_str = f"{fcr:.2%}" if isinstance(fcr, (int, float)) else str(fcr)
                fpr_str = f"{fpr_val:.2%}" if isinstance(fpr_val, (int, float)) else str(fpr_val)
                
                report.append(f"| {policy_name} | {avg_r_str} | {fcr_str} | {fpr_str} |")
        
        report.append("")
    else:
        report.append("*Phase 3 metrics not available. Run `python run_phase3.py` first.*")
        report.append("")
    
    report.append("### Key Insight")
    report.append("")
    report.append("The DQN agent learned to **balance** fraud prevention against customer")
    report.append("friction. Unlike a binary classifier that says \"fraud or not,\" the RL")
    report.append("agent chooses the **appropriate response intensity** based on the risk level")
    report.append("and the cost of each possible mistake.")
    report.append("")
    
    report.append("---")
    report.append("")
    
    # ================================================================
    # System Architecture
    # ================================================================
    report.append("## System Architecture")
    report.append("")
    report.append("```")
    report.append("┌─────────────────────────────────────────────────────────────┐")
    report.append("│                    RAW TRANSACTION                          │")
    report.append("│  (amount, card_id, merchant, time, ...)                     │")
    report.append("└──────────────────────┬──────────────────────────────────────┘")
    report.append("                       │")
    report.append("                       ▼")
    report.append("┌─────────────────────────────────────────────────────────────┐")
    report.append("│              LAYER 1: GRAPH CONSTRUCTION                    │")
    report.append("│  Cards and merchants as nodes, transactions as edges        │")
    report.append("│  Bipartite heterogeneous graph                              │")
    report.append("└──────────────────────┬──────────────────────────────────────┘")
    report.append("                       │")
    report.append("                       ▼")
    report.append("┌─────────────────────────────────────────────────────────────┐")
    report.append("│              LAYER 2: GNN FRAUD SCORING                     │")
    report.append("│  Graph Attention Network (GAT)                              │")
    report.append("│  Multi-head attention learns neighbor importance            │")
    report.append("│  Output: fraud probability per card (0.0 to 1.0)            │")
    report.append("└──────────────────────┬──────────────────────────────────────┘")
    report.append("                       │")
    report.append("                       ▼")
    report.append("┌─────────────────────────────────────────────────────────────┐")
    report.append("│              LAYER 3: RL RESPONSE AGENT                     │")
    report.append("│  Deep Q-Network (DQN)                                       │")
    report.append("│  State = [fraud_score, amount, time, card_freq, ...]        │")
    report.append("│  Actions: APPROVE | SOFT_BLOCK | HARD_BLOCK |               │")
    report.append("│           FLAG_REVIEW | FREEZE_ACCOUNT                      │")
    report.append("│  Output: optimal action for this transaction                │")
    report.append("└─────────────────────────────────────────────────────────────┘")
    report.append("```")
    report.append("")
    report.append("---")
    report.append("")
    
    # ================================================================
    # Technical Decisions
    # ================================================================
    report.append("## Technical Decisions and Limitations")
    report.append("")
    report.append("### Data")
    report.append("- **Merchant Node:** The dataset has no explicit merchant ID. Created synthetic")
    report.append("  merchant nodes from `addr1 + ProductCD` combination. This is a proxy.")
    report.append("- **Time Split:** 80/20 time-based split to prevent data leakage.")
    report.append("- **Class Imbalance:** ~3.5% fraud rate handled with balanced class weights.")
    report.append("")
    report.append("### GNN")
    report.append("- **Node vs Transaction Classification:** GNN classifies card nodes, not")
    report.append("  individual transactions. This is a different evaluation granularity than")
    report.append("  the tabular baselines. Comparison is directional, not exact.")
    report.append("- **Transductive Setting:** All nodes are in the graph during training,")
    report.append("  but only train_mask nodes contribute to the loss.")
    report.append("")
    report.append("### RL")
    report.append("- **Reward Function:** Hand-designed to reflect business costs. Real deployment")
    report.append("  would need actual chargeback data and customer lifetime value.")
    report.append("- **Environment:** Simulated from static dataset. Real deployment would be")
    report.append("  online learning with streaming transactions.")
    report.append("- **State Space:** 8-dimensional simplified representation. Production system")
    report.append("  would include more features and historical context.")
    report.append("")
    report.append("---")
    report.append("")
    
    # ================================================================
    # How to Run
    # ================================================================
    report.append("## How to Reproduce")
    report.append("")
    report.append("```bash")
    report.append("# Setup")
    report.append("python -m venv venv")
    report.append("source venv/bin/activate")
    report.append("pip install -r requirements.txt")
    report.append("")
    report.append("# Download IEEE-CIS data from Kaggle into data/ folder")
    report.append("")
    report.append("# Run all phases")
    report.append("python run_phase1.py     # ~15-20 minutes")
    report.append("python run_phase2.py     # ~15-25 minutes")
    report.append("python run_phase3.py     # ~25-35 minutes")
    report.append("")
    report.append("# Demo")
    report.append("python demo.py")
    report.append("")
    report.append("# Generate this report")
    report.append("python generate_report.py")
    report.append("```")
    report.append("")
    report.append("---")
    report.append("")
    report.append(f"*Report generated on {timestamp}*")
    
    # ================================================================
    # Write Report
    # ================================================================
    report_path = OUTPUT_DIR / 'PROJECT_REPORT.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"✓ Report saved to: {report_path}")
    print(f"  Total lines: {len(report)}")
    
    return report_path


if __name__ == "__main__":
    generate_report()
