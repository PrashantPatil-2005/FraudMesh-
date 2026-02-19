# FraudMesh-RL

**End-to-end financial fraud detection and autonomous response system combining Graph Neural Networks and Reinforcement Learning.**

Built on 590,540 real-world transactions from the IEEE-CIS Fraud Detection dataset.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://prashantpatil-2005-fraudmesh-rl.streamlit.app)

---

## What This System Does

Most fraud detection systems answer: **"Is this fraud?"**

This system answers: **"Is this fraud, AND what should we do about it?"**

```
Transaction → Graph Analysis → Fraud Score → Response Decision
                  (GNN)          (0-1)        (RL Agent)

                                              ┌─ APPROVE
                                              ├─ SOFT BLOCK (request 2FA)
                                     0.87 ──→ ├─ HARD BLOCK (decline)
                                              ├─ FLAG FOR REVIEW
                                              └─ FREEZE ACCOUNT
```

---

## 🚀 Live Demo

**[Try the interactive Streamlit app →](https://prashantpatil-2005-fraudmesh-rl.streamlit.app)**

Explore 500 real transactions through the full GNN → RL pipeline — no setup required.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    RAW TRANSACTION                        │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│              LAYER 1: GRAPH CONSTRUCTION                  │
│  • Cards and merchants as nodes                           │
│  • Transactions as edges                                  │
│  • Bipartite heterogeneous graph                          │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│              LAYER 2: GNN FRAUD SCORING (GAT)             │
│  • Multi-head attention on neighbor nodes                 │
│  • Fraud signal propagates through graph                  │
│  • Output: fraud probability per card                     │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│              LAYER 3: RL RESPONSE AGENT (DQN)             │
│  • State: [fraud_score, amount, time, velocity, ...]      │
│  • 5 actions with different business costs                │
│  • Learned to balance fraud prevention vs customer UX     │
└──────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/PrashantPatil-2005/FraudMesh-.git
cd FraudMesh-
python -m venv venv

# Windows
.\venv\Scripts\activate

# Mac/Linux
# source venv/bin/activate

pip install -r requirements.txt

# 2. Download data
# Go to https://www.kaggle.com/c/ieee-fraud-detection/data
# Download train_transaction.csv and train_identity.csv
# Place in data/ folder

# 3. Run all phases
python run_phase1.py     # Baselines (~15 min)
python run_phase2.py     # GNN (~20 min)
python run_phase3.py     # RL (~30 min)

# 4. Demo
python demo.py

# 5. Generate report
python generate_report.py

# 6. Validate everything
python validate_project.py

# 7. Launch Streamlit app
python -m app.precompute                       # Pre-compute demo data (run once)
.\venv\Scripts\streamlit run streamlit_app.py  # Windows
# streamlit run streamlit_app.py               # Mac/Linux
```

---

## Streamlit Web App

The project includes an interactive Streamlit dashboard with 5 pages:

| Page | Description |
|------|-------------|
| 🏠 **Home** | Project overview, key metrics, pipeline explanation |
| 🔍 **Transaction Explorer** | Select any transaction → see GNN score, RL state vector, DQN decision with Q-values |
| 📊 **Batch Analysis** | Fraud score distributions, action breakdowns, reward analysis across all 500 transactions |
| 📈 **Model Comparison** | Phase 1 vs 2 vs 3 results side-by-side with saved plots |
| 🏗️ **Architecture** | System design, why GNN, why RL, design decisions, limitations |

### Running Locally

```bash
# Step 1: Pre-compute demo data (one time, ~5 min)
python -m app.precompute

# Step 2: Launch the app
.\venv\Scripts\streamlit run streamlit_app.py   # Windows
# streamlit run streamlit_app.py                # Mac/Linux

# Opens at http://localhost:8501
```

### Deploying to Streamlit Cloud

1. Push to GitHub (make sure `app/demo_data/` is included)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select your repo → Branch: `main` → Main file: `streamlit_app.py`
4. Click **Deploy**

---

## Results

### Phase 1: Baseline Detection

| Model | AUC-ROC | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| Logistic Regression | — | — | — | — |
| Random Forest | — | — | — | — |

*Run `python run_phase1.py` to populate these numbers.*

### Phase 2: GNN Detection

| Model | AUC-ROC | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| GAT (2-layer) | — | — | — | — |

*Run `python run_phase2.py` to populate these numbers.*

### Phase 3: RL Response Agent

| Policy | Avg Reward | Fraud Catch Rate | False Positive Rate |
|--------|-----------|-----------------|-------------------|
| DQN Agent | — | — | — |
| Rule-Based | — | — | — |
| Random | — | — | — |

*Run `python run_phase3.py` to populate these numbers.*

---

## Why GNN Over Traditional ML

A traditional model sees each transaction in isolation:

> Card #1234 spent $50 at Store X at 3am → **Low risk** (new card, small amount)

The GNN sees the graph neighborhood:

> Store X is connected to 47 cards, 12 of which were confirmed fraud this week.
> Card #1234 is the 13th card at this merchant.
> → **High risk** (fraud cluster detected)

The fraud signal travels through the graph structure.

---

## Why RL Over Binary Classification

Binary classifiers output: **fraud** or **not fraud**.

But the cost of each mistake is different:

| Scenario | Cost |
|----------|------|
| Approve real fraud | $50 loss |
| Block legitimate customer | $10 lost revenue + trust damage |
| Freeze innocent account | $30 customer service + churn risk |
| Flag for review (fraud) | $7 analyst cost but fraud caught |
| Soft block (legit) | $2 minor 2FA friction |

The RL agent learned these cost tradeoffs and picks the action
that minimizes total business cost.

---

## Project Structure

```
fraudmesh-rl/
├── data/                          # IEEE-CIS dataset (not in git)
│   ├── train_transaction.csv
│   └── train_identity.csv
├── src/
│   ├── config.py                  # All constants and paths
│   ├── data_loader.py             # Load and merge CSVs
│   ├── eda.py                     # Exploratory data analysis
│   ├── feature_engineer.py        # Cleaning, encoding, splitting
│   ├── graph_builder.py           # NetworkX graph construction
│   ├── baseline_model.py          # Logistic Regression + Random Forest
│   ├── evaluate.py                # Phase 1 evaluation
│   ├── pyg_converter.py           # NetworkX → PyTorch Geometric
│   ├── gnn_model.py               # GAT model definition
│   ├── gnn_trainer.py             # GNN training loop
│   ├── gnn_evaluate.py            # GNN evaluation
│   ├── rl_environment.py          # Custom Gymnasium environment
│   ├── replay_buffer.py           # Experience replay
│   ├── dqn_model.py               # Q-Network architecture
│   ├── dqn_agent.py               # DQN agent
│   ├── rl_trainer.py              # RL training loop
│   ├── rl_evaluate.py             # RL evaluation
│   └── rl_baselines.py            # Random and rule-based policies
├── app/
│   ├── precompute.py              # Pre-compute demo data for Streamlit
│   └── demo_data/                 # Pre-computed data (pushed to git)
├── outputs/
│   ├── plots/                     # All visualizations
│   └── metrics/                   # All metrics as JSON
├── models/                        # Saved model weights
├── .streamlit/
│   └── config.toml                # Streamlit theme config
├── streamlit_app.py               # Streamlit web app (5 pages)
├── run_phase1.py                  # Phase 1 entry point
├── run_phase2.py                  # Phase 2 entry point
├── run_phase3.py                  # Phase 3 entry point
├── demo.py                        # Full system demonstration
├── generate_report.py             # Report generator
├── validate_project.py            # Project validator
├── Procfile                       # Hosting config
├── packages.txt                   # System dependencies
├── requirements.txt
└── README.md
```

---

## Technical Stack

- **Python 3.10+**
- **PyTorch** — neural network framework
- **PyTorch Geometric** — graph neural networks
- **Gymnasium** — RL environment interface
- **scikit-learn** — baseline models
- **Streamlit** — interactive web dashboard
- **NetworkX** — graph construction and analysis
- **pandas / numpy** — data processing
- **matplotlib / seaborn** — visualization

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Merchant node | `addr1 + ProductCD` | No direct merchant ID in dataset |
| Train/test split | Time-based (80/20) | Prevents temporal data leakage |
| GNN type | GAT (not GCN) | Attention weights are interpretable |
| RL algorithm | DQN (not PPO) | Discrete action space, simpler to debug |
| Graph type | Heterogeneous bipartite | Two node types require HeteroConv |
| Reward design | Asymmetric costs | Missed fraud costs more than false positive |
| Demo app | Pre-computed data only | Fast load on free hosting (<10s) |

---

## Limitations

- **Merchant node is a proxy** — real systems have actual merchant IDs
- **Reward function is hand-designed** — real costs come from business data
- **Static dataset** — real deployment would be online/streaming
- **GNN evaluates cards, baselines evaluate transactions** — comparison is directional
- **Environment is simulated** — real RL would learn from live feedback

---

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)

- 590,540 transactions
- 394 features (transaction) + 41 features (identity)
- ~3.5% fraud rate
- Real-world e-commerce transactions

---

## License

MIT