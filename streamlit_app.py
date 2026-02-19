"""
FraudMesh-RL — Interactive Demo
Streamlit web application for exploring the fraud detection + response system.

Run locally: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================

DEMO_DIR = Path('app/demo_data')

RL_ACTIONS = {
    0: 'APPROVE',
    1: 'SOFT_BLOCK',
    2: 'HARD_BLOCK',
    3: 'FLAG_REVIEW',
    4: 'FREEZE_ACCOUNT'
}

ACTION_EMOJIS = {
    'APPROVE': '✅',
    'SOFT_BLOCK': '🔒',
    'HARD_BLOCK': '🚫',
    'FLAG_REVIEW': '🔍',
    'FREEZE_ACCOUNT': '🧊'
}

ACTION_COLORS = {
    'APPROVE': '#2ecc71',
    'SOFT_BLOCK': '#f1c40f',
    'HARD_BLOCK': '#e74c3c',
    'FLAG_REVIEW': '#3498db',
    'FREEZE_ACCOUNT': '#9b59b6'
}

STATE_NAMES = [
    'Fraud Score', 'Amount', 'Hour of Day', 'High Amount',
    'Card Frequency', 'Merchant Risk', 'Amount Z-Score', 'Velocity'
]

# ============================================================================
# DATA LOADING (cached)
# ============================================================================

@st.cache_data
def load_demo_data():
    """Load all pre-computed demo data."""
    data = {}
    
    # Transactions
    tx_path = DEMO_DIR / 'transactions.csv'
    if tx_path.exists():
        data['transactions'] = pd.read_csv(tx_path)
    else:
        data['transactions'] = None
    
    # States
    states_path = DEMO_DIR / 'states.npy'
    if states_path.exists():
        data['states'] = np.load(states_path)
    else:
        data['states'] = None
    
    # Q-values
    qvals_path = DEMO_DIR / 'q_values.npy'
    if qvals_path.exists():
        data['q_values'] = np.load(qvals_path)
    else:
        data['q_values'] = None
    
    # Metadata
    meta_path = DEMO_DIR / 'metadata.json'
    if meta_path.exists():
        with open(meta_path) as f:
            data['metadata'] = json.load(f)
    else:
        data['metadata'] = None
    
    # Neighborhoods
    neigh_path = DEMO_DIR / 'neighborhoods.json'
    if neigh_path.exists():
        with open(neigh_path) as f:
            data['neighborhoods'] = json.load(f)
    else:
        data['neighborhoods'] = None
    
    # Phase metrics
    for phase in ['phase1_baselines', 'phase2_gnn', 'phase3_combined_comparison']:
        path = DEMO_DIR / f'{phase}.json'
        if path.exists():
            with open(path) as f:
                data[phase] = json.load(f)
        else:
            data[phase] = None
    
    return data


# ============================================================================
# PAGE: HOME
# ============================================================================

def page_home(data):
    """Project overview and key metrics."""
    
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='font-size: 3em;'>🛡️ FraudMesh-RL</h1>
        <p style='font-size: 1.3em; color: #aaa;'>
            Graph Neural Network + Reinforcement Learning<br>
            for Financial Fraud Detection & Autonomous Response
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Key stats
    if data['metadata']:
        meta = data['metadata']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Transactions Analyzed",
                value=f"{meta['num_transactions']:,}"
            )
        
        with col2:
            st.metric(
                label="Fraud Detected",
                value=f"{meta['fraud_count']:,}",
                delta=f"{meta['fraud_count']/meta['num_transactions']*100:.1f}%"
            )
        
        with col3:
            st.metric(
                label="Decision Accuracy",
                value=f"{meta['accuracy']*100:.1f}%"
            )
        
        with col4:
            st.metric(
                label="Total Reward",
                value=f"{meta['total_reward']:+.0f}"
            )
    
    st.divider()
    
    # Pipeline overview
    st.subheader("How It Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔗 Phase 1: Graph
        - Build card-merchant transaction graph
        - Cards and merchants as nodes
        - Transactions as edges
        - Baseline ML models (LR, RF)
        """)
    
    with col2:
        st.markdown("""
        ### 🧠 Phase 2: GNN
        - Graph Attention Network (GAT)
        - Fraud signal propagates through graph
        - Learns which neighbors matter
        - Outputs fraud probability per card
        """)
    
    with col3:
        st.markdown("""
        ### 🎮 Phase 3: RL Agent
        - Deep Q-Network (DQN)
        - 5 possible response actions
        - Balances fraud prevention vs UX
        - Learned cost-optimal decisions
        """)
    
    st.divider()
    
    # The key insight
    st.subheader("Why This Matters")
    
    st.info("""
    **Traditional systems** say: *"This is fraud"* or *"This is not fraud."*
    
    **FraudMesh-RL** says: *"This has 87% fraud probability. Based on the transaction 
    amount, customer history, and merchant risk profile, the optimal action is 
    HARD_BLOCK — declining the transaction. Soft-blocking would risk $50 in fraud 
    losses, while hard-blocking a legitimate customer here only costs $10 in friction."*
    """)
    
    # Show fraud score source
    if data['metadata']:
        st.caption(f"Fraud scores generated using: **{meta['fraud_score_source']}** | "
                  f"RL decisions using: **{meta['rl_method']}**")


# ============================================================================
# PAGE: TRANSACTION EXPLORER
# ============================================================================

def page_transaction_explorer(data):
    """Interactive single transaction analysis."""
    
    st.header("🔍 Transaction Explorer")
    st.markdown("Select a transaction to see the full pipeline in action.")
    
    if data['transactions'] is None:
        st.error("Demo data not found. Run `python -m app.precompute` first.")
        return
    
    df = data['transactions']
    states = data['states']
    q_values = data['q_values']
    neighborhoods = data['neighborhoods']
    
    # Transaction selector
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Filter options
        show_only = st.radio(
            "Show:",
            ["All", "Fraud Only", "Legit Only"],
            index=0
        )
        
        if show_only == "Fraud Only":
            filtered = df[df['true_label'] == 1]
        elif show_only == "Legit Only":
            filtered = df[df['true_label'] == 0]
        else:
            filtered = df
        
        tx_idx = st.selectbox(
            "Transaction #",
            filtered.index.tolist(),
            format_func=lambda x: f"#{x} — {'🔴 FRAUD' if df.iloc[x]['true_label'] == 1 else '🟢 LEGIT'} — ${df.iloc[x]['amount']:.2f}"
        )
    
    with col2:
        tx = df.iloc[tx_idx]
        
        # Transaction details
        st.subheader(f"Transaction #{tx_idx}")
        
        detail_col1, detail_col2, detail_col3 = st.columns(3)
        
        with detail_col1:
            st.markdown(f"**Amount:** ${tx['amount']:,.2f}")
            st.markdown(f"**Card ID:** {int(tx['card_id'])}")
            
        with detail_col2:
            st.markdown(f"**Card Type:** {tx.get('card_type', 'N/A')}")
            st.markdown(f"**Category:** {tx.get('card_category', 'N/A')}")
            
        with detail_col3:
            st.markdown(f"**Product:** {tx.get('product_cd', 'N/A')}")
            true_label = "🔴 FRAUD" if tx['true_label'] == 1 else "🟢 LEGITIMATE"
            st.markdown(f"**True Label:** {true_label}")
    
    st.divider()
    
    # ── Stage 1: GNN Fraud Score ──
    st.subheader("Stage 1: GNN Fraud Score")
    
    fraud_score = tx['fraud_score']
    
    score_col1, score_col2 = st.columns([2, 3])
    
    with score_col1:
        # Fraud score gauge
        if fraud_score < 0.2:
            risk_level = "LOW RISK"
            risk_color = "green"
        elif fraud_score < 0.5:
            risk_level = "MEDIUM RISK"
            risk_color = "orange"
        elif fraud_score < 0.8:
            risk_level = "HIGH RISK"
            risk_color = "red"
        else:
            risk_level = "CRITICAL"
            risk_color = "red"
        
        st.metric("Fraud Probability", f"{fraud_score:.4f}")
        st.markdown(f"**Risk Level:** :{risk_color}[{risk_level}]")
    
    with score_col2:
        # Graph neighborhood
        if neighborhoods and tx_idx < len(neighborhoods):
            neigh = neighborhoods[tx_idx]
            
            st.markdown("**Graph Neighborhood:**")
            st.markdown(f"- Merchant: `{neigh['merchant']}`")
            st.markdown(f"- Merchant fraud rate: **{neigh['merchant_fraud_rate']*100:.1f}%**")
            st.markdown(f"- Transactions at merchant: **{neigh['merchant_total_txs']}**")
            st.markdown(f"- Cards connected to merchant: **{len(neigh['connected_cards'])}**")
            st.markdown(f"- Card total transactions: **{neigh['card_total_txs']}**")
    
    st.divider()
    
    # ── Stage 2: RL State Vector ──
    st.subheader("Stage 2: RL State Vector")
    
    if states is not None and tx_idx < len(states):
        state = states[tx_idx]
        
        # Bar chart of state features
        fig, ax = plt.subplots(figsize=(10, 3))
        
        colors = []
        for val in state:
            if val < 0.3:
                colors.append('#2ecc71')
            elif val < 0.6:
                colors.append('#f1c40f')
            else:
                colors.append('#e74c3c')
        
        bars = ax.barh(range(len(state)), state, color=colors, edgecolor='white', height=0.6)
        ax.set_yticks(range(len(state)))
        ax.set_yticklabels(STATE_NAMES, fontsize=10)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel('Value (0-1)')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, state)):
            ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.divider()
    
    # ── Stage 3: RL Decision ──
    st.subheader("Stage 3: RL Agent Decision")
    
    action_name = tx['rl_action_name']
    action_idx = int(tx['rl_action'])
    reward = tx['reward']
    is_correct = tx['correct']
    
    dec_col1, dec_col2 = st.columns([2, 3])
    
    with dec_col1:
        emoji = ACTION_EMOJIS.get(action_name, '')
        color = ACTION_COLORS.get(action_name, '#ffffff')
        
        st.markdown(f"""
        <div style='background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;'>
            <h2 style='color: black; margin: 0;'>{emoji} {action_name}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        if is_correct:
            st.success(f"✅ Correct decision! Reward: **{reward:+.1f}**")
        else:
            st.error(f"❌ Wrong decision. Reward: **{reward:+.1f}**")
    
    with dec_col2:
        # Q-values bar chart
        if q_values is not None and tx_idx < len(q_values):
            qv = q_values[tx_idx]
            
            fig, ax = plt.subplots(figsize=(8, 3))
            
            action_names = [RL_ACTIONS[a] for a in range(5)]
            bar_colors = [ACTION_COLORS.get(name, 'gray') for name in action_names]
            
            # Highlight chosen action
            edge_colors = ['black' if a == action_idx else 'gray' for a in range(5)]
            linewidths = [3 if a == action_idx else 1 for a in range(5)]
            
            bars = ax.bar(range(5), qv, color=bar_colors, 
                         edgecolor=edge_colors, linewidth=linewidths)
            ax.set_xticks(range(5))
            ax.set_xticklabels(action_names, rotation=30, ha='right', fontsize=9)
            ax.set_ylabel('Q-Value')
            ax.set_title('Q-Values (higher = agent prefers)', fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    # Cost explanation
    st.divider()
    st.subheader("Why This Action?")
    
    if data['metadata']:
        rewards_struct = data['metadata'].get('reward_structure', {})
        
        true_type = "FRAUD" if tx['true_label'] == 1 else "LEGIT"
        
        st.markdown(f"**True label: {true_type}**")
        st.markdown("What would each action have cost?")
        
        cost_data = []
        for a in range(5):
            a_name = RL_ACTIONS[a]
            key = f"{a_name}_{true_type}"
            r = rewards_struct.get(key, 0)
            cost_data.append({
                'Action': f"{ACTION_EMOJIS.get(a_name, '')} {a_name}",
                'Reward': r,
                'Chosen': '← CHOSEN' if a == action_idx else ''
            })
        
        cost_df = pd.DataFrame(cost_data)
        st.dataframe(cost_df, use_container_width=True, hide_index=True)


# ============================================================================
# PAGE: BATCH ANALYSIS
# ============================================================================

def page_batch_analysis(data):
    """View all transactions at once."""
    
    st.header("📊 Batch Analysis")
    
    if data['transactions'] is None:
        st.error("Demo data not found. Run `python -m app.precompute` first.")
        return
    
    df = data['transactions']
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total", len(df))
    with col2:
        st.metric("Fraud", int(df['true_label'].sum()))
    with col3:
        st.metric("Correct", int(df['correct'].sum()))
    with col4:
        accuracy = df['correct'].mean() * 100
        st.metric("Accuracy", f"{accuracy:.1f}%")
    with col5:
        st.metric("Total Reward", f"{df['reward'].sum():+.0f}")
    
    st.divider()
    
    # ── Fraud Score Distribution ──
    st.subheader("Fraud Score Distribution")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    fraud = df[df['true_label'] == 1]['fraud_score']
    legit = df[df['true_label'] == 0]['fraud_score']
    
    ax.hist(legit, bins=30, alpha=0.6, color='green', label=f'Legitimate ({len(legit)})')
    ax.hist(fraud, bins=30, alpha=0.6, color='red', label=f'Fraud ({len(fraud)})')
    ax.axvline(x=0.5, color='orange', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Fraud Score')
    ax.set_ylabel('Count')
    ax.legend()
    ax.set_title('Distribution of GNN Fraud Scores')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.divider()
    
    # ── Action Distribution ──
    st.subheader("Action Distribution")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall action distribution
    action_counts = df['rl_action_name'].value_counts()
    colors = [ACTION_COLORS.get(name, 'gray') for name in action_counts.index]
    
    axes[0].pie(action_counts.values, labels=action_counts.index, colors=colors,
               autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Overall Action Distribution')
    
    # Action by true label
    action_by_label = df.groupby(['true_label', 'rl_action_name']).size().unstack(fill_value=0)
    action_by_label.index = ['Legitimate', 'Fraud']
    
    action_by_label.plot(kind='bar', ax=axes[1], 
                        color=[ACTION_COLORS.get(c, 'gray') for c in action_by_label.columns],
                        edgecolor='black')
    axes[1].set_title('Actions by True Label')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Count')
    axes[1].legend(title='Action', bbox_to_anchor=(1.05, 1))
    axes[1].set_xticklabels(['Legitimate', 'Fraud'], rotation=0)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.divider()
    
    # ── Reward Analysis ──
    st.subheader("Reward Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        
        reward_by_action = df.groupby('rl_action_name')['reward'].sum().sort_values()
        colors = [ACTION_COLORS.get(name, 'gray') for name in reward_by_action.index]
        
        reward_by_action.plot(kind='barh', ax=ax, color=colors, edgecolor='black')
        ax.set_xlabel('Total Reward')
        ax.set_title('Total Reward by Action')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Correct vs incorrect
        n_correct = int(df['correct'].sum())
        n_incorrect = len(df) - n_correct
        
        ax.pie([n_correct, n_incorrect], 
              labels=['Correct', 'Incorrect'],
              colors=['#2ecc71', '#e74c3c'],
              autopct='%1.1f%%', startangle=90,
              explode=(0.05, 0.05))
        ax.set_title('Decision Accuracy')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.divider()
    
    # ── Full Transaction Table ──
    st.subheader("All Transactions")
    
    display_df = df[['transaction_id', 'amount', 'card_id', 'true_label',
                     'fraud_score', 'rl_action_name', 'reward', 'correct']].copy()
    display_df.columns = ['ID', 'Amount', 'Card', 'Fraud?', 
                          'Score', 'Action', 'Reward', 'Correct']
    display_df['Amount'] = display_df['Amount'].apply(lambda x: f"${x:,.2f}")
    display_df['Fraud?'] = display_df['Fraud?'].map({1: '🔴 Yes', 0: '🟢 No'})
    display_df['Score'] = display_df['Score'].apply(lambda x: f"{x:.4f}")
    display_df['Correct'] = display_df['Correct'].map({True: '✅', False: '❌'})
    
    st.dataframe(display_df, use_container_width=True, height=400, hide_index=True)


# ============================================================================
# PAGE: MODEL COMPARISON
# ============================================================================

def page_model_comparison(data):
    """Compare all phases and models."""
    
    st.header("📈 Model Comparison")
    
    # ── Phase 1 vs Phase 2 ──
    st.subheader("Detection: Baselines vs GNN")
    
    phase1 = data.get('phase1_baselines')
    phase2 = data.get('phase2_gnn')
    
    if phase1 or phase2:
        metrics_to_show = [
            ('AUC-ROC', 'auc'),
            ('F1 Score', 'f1'),
            ('Precision', 'precision'),
            ('Recall', 'recall'),
        ]
        
        comparison_data = []
        for display_name, key in metrics_to_show:
            row = {'Metric': display_name}
            
            if phase1:
                lr = phase1.get('logistic_regression', {})
                rf = phase1.get('random_forest', {})
                row['Logistic Reg'] = f"{lr.get(key, 'N/A'):.4f}" if isinstance(lr.get(key), (int, float)) else 'N/A'
                row['Random Forest'] = f"{rf.get(key, 'N/A'):.4f}" if isinstance(rf.get(key), (int, float)) else 'N/A'
            
            if phase2:
                gnn = phase2.get('gnn_gat', {})
                val = gnn.get(key, 'N/A')
                row['GNN (GAT)'] = f"{val:.4f}" if isinstance(val, (int, float)) else 'N/A'
            
            comparison_data.append(row)
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
        
        st.caption("⚠️ Note: Phase 1 evaluates at transaction level. Phase 2 evaluates at card level. "
                   "Comparison is directional, not exact apples-to-apples.")
    else:
        st.info("Phase metrics not available. Run the phases first.")
    
    st.divider()
    
    # ── Phase 3: RL Comparison ──
    st.subheader("Response: RL Agent vs Baselines")
    
    phase3 = data.get('phase3_combined_comparison')
    
    if phase3:
        rl_results = phase3.get('phase3_rl_results', phase3)
        dqn = rl_results.get('dqn_agent', {})
        baselines = rl_results.get('baselines', {})
        
        rl_data = []
        
        if dqn:
            rl_data.append({
                'Policy': '🤖 DQN Agent',
                'Avg Reward': f"{dqn.get('avg_reward', 0):.2f}",
                'Fraud Catch Rate': f"{dqn.get('avg_fraud_catch_rate', 0):.2%}",
                'False Positive Rate': f"{dqn.get('avg_false_positive_rate', 0):.2%}"
            })
        
        for policy_name, metrics in baselines.items():
            if isinstance(metrics, dict):
                rl_data.append({
                    'Policy': policy_name,
                    'Avg Reward': f"{metrics.get('avg_reward', 0):.2f}",
                    'Fraud Catch Rate': f"{metrics.get('avg_fraud_catch_rate', 0):.2%}",
                    'False Positive Rate': f"{metrics.get('avg_false_positive_rate', 0):.2%}"
                })
        
        if rl_data:
            st.dataframe(pd.DataFrame(rl_data), use_container_width=True, hide_index=True)
    else:
        st.info("Phase 3 metrics not available.")
    
    st.divider()
    
    # ── Show saved plots ──
    st.subheader("Visualizations")
    
    plots_dir = DEMO_DIR / 'plots'
    
    plot_pairs = [
        ('model_comparison_auc.png', 'AUC Comparison'),
        ('rl_reward_comparison.png', 'RL Policy Comparison'),
        ('rl_action_distribution.png', 'Action Distribution'),
        ('rl_cost_analysis.png', 'Cost Analysis'),
        ('gnn_training_curves.png', 'GNN Training Curves'),
        ('rl_training_curves.png', 'RL Training Curves'),
    ]
    
    for filename, title in plot_pairs:
        plot_path = plots_dir / filename
        if plot_path.exists():
            st.markdown(f"**{title}**")
            st.image(str(plot_path), use_container_width=True)
            st.markdown("")


# ============================================================================
# PAGE: ARCHITECTURE
# ============================================================================

def page_architecture(data):
    """System architecture explanation."""
    
    st.header("🏗️ System Architecture")
    
    st.markdown("""
    ### The Three-Layer Pipeline
    
    ```
    ┌──────────────────────────────────────────────────────┐
    │              RAW TRANSACTION                         │
    │  (amount, card_id, merchant, time, ...)              │
    └─────────────────────┬────────────────────────────────┘
                          │
                          ▼
    ┌──────────────────────────────────────────────────────┐
    │         LAYER 1: GRAPH CONSTRUCTION                  │
    │  • Cards ←→ Merchants (bipartite graph)              │
    │  • Transactions become edges                         │
    │  • Node features: aggregated statistics              │
    └─────────────────────┬────────────────────────────────┘
                          │
                          ▼
    ┌──────────────────────────────────────────────────────┐
    │         LAYER 2: GNN FRAUD SCORING (GAT)             │
    │  • 2-layer Graph Attention Network                   │
    │  • Multi-head attention on neighbors                 │
    │  • Fraud signal propagates through graph             │
    │  • Output: fraud probability (0.0 → 1.0)            │
    └─────────────────────┬────────────────────────────────┘
                          │
                          ▼
    ┌──────────────────────────────────────────────────────┐
    │         LAYER 3: RL RESPONSE AGENT (DQN)             │
    │  • State: [fraud_score, amount, time, ...]           │
    │  • 5 actions: APPROVE → FREEZE_ACCOUNT               │
    │  • Learned cost-optimal response                     │
    └──────────────────────────────────────────────────────┘
    ```
    """)
    
    st.divider()
    
    st.subheader("Why GNN?")
    
    st.markdown("""
    **Traditional model (Random Forest):**
    > Card #1234 spent $50 at Store X at 3am.  
    > No history for this card. Low amount.  
    > **Decision: LOW RISK** ❌
    
    **Graph model (GAT):**
    > Card #1234 spent $50 at Store X at 3am.  
    > Store X is connected to 47 other cards.  
    > 12 of those cards were confirmed fraud this week.  
    > **Decision: HIGH RISK** ✅
    
    The fraud signal travels through the **graph structure**.
    """)
    
    st.divider()
    
    st.subheader("Why RL?")
    
    st.markdown("""
    **Binary classifier:** *"This is fraud."*  
    **RL agent:** *"This is fraud, and here's exactly what to do about it."*
    
    | Scenario | Cost |
    |----------|------|
    | Approve real fraud | **-$50** lost to fraud |
    | Block legitimate customer | **-$10** lost revenue + trust |
    | Freeze innocent account | **-$30** customer service + churn |
    | Soft block (2FA) on fraud | **+$5** caught fraud, light touch |
    | Approve legitimate transaction | **+$1** customer happy |
    
    The RL agent **learned these tradeoffs** and picks the action
    that minimizes total business cost.
    """)
    
    st.divider()
    
    st.subheader("Technical Stack")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **ML & Deep Learning:**
        - PyTorch
        - PyTorch Geometric
        - scikit-learn
        - Gymnasium
        
        **Data:**
        - IEEE-CIS Fraud Detection (Kaggle)
        - 590,540 real transactions
        - ~3.5% fraud rate
        """)
    
    with col2:
        st.markdown("""
        **Models:**
        - Graph Attention Network (GAT)
        - Deep Q-Network (DQN)
        - Experience Replay Buffer
        - Target Network + Soft Updates
        
        **Visualization:**
        - Streamlit
        - NetworkX
        - matplotlib / seaborn
        """)
    
    st.divider()
    
    st.subheader("Key Design Decisions")
    
    decisions = pd.DataFrame([
        {'Decision': 'Merchant node', 'Choice': 'addr1 + ProductCD', 
         'Why': 'No direct merchant ID in dataset'},
        {'Decision': 'Train/test split', 'Choice': 'Time-based (80/20)', 
         'Why': 'Prevents temporal data leakage'},
        {'Decision': 'GNN type', 'Choice': 'GAT (not GCN)', 
         'Why': 'Attention weights are interpretable'},
        {'Decision': 'RL algorithm', 'Choice': 'DQN (not PPO)', 
         'Why': 'Discrete action space, simpler to debug'},
        {'Decision': 'Reward design', 'Choice': 'Asymmetric costs', 
         'Why': 'Missed fraud costs more than false positive'},
    ])
    
    st.dataframe(decisions, use_container_width=True, hide_index=True)
    
    st.divider()
    
    st.subheader("Limitations (Honest Assessment)")
    
    st.markdown("""
    - **Merchant node is a proxy** — real systems have actual merchant IDs
    - **Reward function is hand-designed** — real costs come from business data
    - **Static dataset** — real deployment would be online/streaming
    - **GNN evaluates cards, baselines evaluate transactions** — comparison is directional
    - **No real-time inference** — demo uses pre-computed scores
    """)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="FraudMesh-RL",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar navigation
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 10px 0;'>
        <h2>🛡️ FraudMesh-RL</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Home", "🔍 Transaction Explorer", "📊 Batch Analysis", 
         "📈 Model Comparison", "🏗️ Architecture"],
        index=0
    )
    
    st.sidebar.divider()
    
    st.sidebar.markdown("""
    **Built with:**
    - PyTorch + PyG
    - Gymnasium
    - Streamlit
    
    **Dataset:**  
    IEEE-CIS Fraud Detection  
    590K real transactions
    """)
    
    st.sidebar.divider()
    
    st.sidebar.markdown("""
    [📂 GitHub](https://github.com/PrashantPatil-2005/FraudMesh-)  
    [📄 Report](outputs/PROJECT_REPORT.md)
    """)
    
    # Load data
    data = load_demo_data()
    
    if data['transactions'] is None:
        st.warning("⚠️ Demo data not found. Please run `python -m app.precompute` first.")
        st.stop()
    
    # Route to pages
    if page == "🏠 Home":
        page_home(data)
    elif page == "🔍 Transaction Explorer":
        page_transaction_explorer(data)
    elif page == "📊 Batch Analysis":
        page_batch_analysis(data)
    elif page == "📈 Model Comparison":
        page_model_comparison(data)
    elif page == "🏗️ Architecture":
        page_architecture(data)


if __name__ == "__main__":
    main()
