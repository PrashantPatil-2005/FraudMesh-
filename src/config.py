
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
