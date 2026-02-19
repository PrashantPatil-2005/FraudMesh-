
import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Data paths
DATA_DIR = ROOT_DIR / 'data'
TRAIN_TRANSACTION = DATA_DIR / 'train_transaction.csv'
TRAIN_IDENTITY = DATA_DIR / 'train_identity.csv'

# Output paths
OUTPUT_DIR = ROOT_DIR / 'outputs'
PLOTS_DIR = OUTPUT_DIR / 'plots'
METRICS_DIR = OUTPUT_DIR / 'metrics'

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset columns
TARGET_COL = 'isFraud'
CARD_COL = 'card1'
TRANSACTION_ID_COL = 'TransactionID'
TIME_COL = 'TransactionDT'
AMOUNT_COL = 'TransactionAmt'

# Merchant node construction
# The IEEE-CIS dataset does NOT have a direct merchant ID column
# We will create a synthetic merchant node using addr1 + ProductCD
MERCHANT_COLS = ['addr1', 'ProductCD']
MERCHANT_NODE_COL = 'merchant_node'

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model parameters
# Class weight will be calculated dynamically in feature_engineer.py
CLASS_WEIGHT = None

# Missing value threshold
MISSING_THRESHOLD = 0.6  # Drop columns with >60% missing values

# Graph visualization
MAX_SUBGRAPH_NODES = 50

# ============================================================================
# PHASE 2 — GNN CONFIGURATION
# ============================================================================

# Model save path
MODELS_DIR = ROOT_DIR / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# GNN Architecture
GNN_HIDDEN_DIM = 64          # Hidden layer dimension
GNN_NUM_HEADS = 4            # Number of attention heads in GAT
GNN_NUM_LAYERS = 2           # Number of GAT layers
GNN_DROPOUT = 0.3            # Dropout rate
GNN_OUTPUT_DIM = 2           # Binary classification (fraud / not fraud)

# Training
GNN_LEARNING_RATE = 0.001
GNN_WEIGHT_DECAY = 5e-4      # L2 regularization
GNN_EPOCHS = 100              # Maximum training epochs
GNN_PATIENCE = 15             # Early stopping patience
GNN_BATCH_SIZE = 512          # For mini-batch training if needed

# Node feature dimensions
# These will be set dynamically during graph conversion
CARD_FEATURE_DIM = None
MERCHANT_FEATURE_DIM = None

# Model checkpoint path
GNN_CHECKPOINT_PATH = MODELS_DIR / 'best_gnn_model.pt'

# Logging
LOG_EVERY_N_EPOCHS = 5       # Print metrics every N epochs

# ============================================================================
# PHASE 3 — RL CONFIGURATION
# ============================================================================

# Action Space
RL_ACTIONS = {
    0: 'APPROVE',
    1: 'SOFT_BLOCK',
    2: 'HARD_BLOCK',
    3: 'FLAG_REVIEW',
    4: 'FREEZE_ACCOUNT'
}
RL_NUM_ACTIONS = 5

# Reward Structure
RL_REWARDS = {
    'APPROVE_FRAUD':        -50.0,
    'SOFT_BLOCK_FRAUD':     +5.0,
    'HARD_BLOCK_FRAUD':     +10.0,
    'FLAG_REVIEW_FRAUD':    +7.0,
    'FREEZE_ACCOUNT_FRAUD': +8.0,
    'APPROVE_LEGIT':        +1.0,
    'SOFT_BLOCK_LEGIT':     -2.0,
    'HARD_BLOCK_LEGIT':     -10.0,
    'FLAG_REVIEW_LEGIT':    -3.0,
    'FREEZE_ACCOUNT_LEGIT': -30.0,
}

# State Space
RL_STATE_DIM = 8

# DQN Hyperparameters
DQN_HIDDEN_DIM = 128
DQN_LEARNING_RATE = 0.001
DQN_GAMMA = 0.99
DQN_TAU = 0.005
DQN_BATCH_SIZE = 64
DQN_BUFFER_SIZE = 50000
DQN_MIN_BUFFER_SIZE = 1000

# Epsilon-Greedy Exploration
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Training
RL_NUM_EPISODES = 500
RL_MAX_STEPS_PER_EPISODE = 200
RL_LOG_EVERY = 25
RL_EVAL_EVERY = 50

# Model save
RL_CHECKPOINT_PATH = MODELS_DIR / 'best_dqn_model.pt'
