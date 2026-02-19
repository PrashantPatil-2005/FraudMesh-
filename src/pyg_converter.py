
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from src.config import (
    TARGET_COL, CARD_COL, MERCHANT_NODE_COL, AMOUNT_COL, TIME_COL
)


def create_node_features(train_df, card_node_to_idx, merchant_node_to_idx):
    """
    Create feature vectors for card and merchant nodes
    using ONLY training data to prevent leakage.
    
    Args:
        train_df: Training DataFrame
        card_node_to_idx: Dict mapping card IDs to indices
        merchant_node_to_idx: Dict mapping merchant IDs to indices
        
    Returns:
        tuple: (card_features, merchant_features, card_labels)
            card_features: torch.Tensor of shape [num_cards, num_card_features]
            merchant_features: torch.Tensor of shape [num_merchants, num_merchant_features]
            card_labels: torch.Tensor of shape [num_cards] with 0/1 labels
    """
    print("  Creating node features from training data...")
    
    # ====================================================================
    # Card Node Features
    # ====================================================================
    # For each card, compute aggregated statistics from its transactions
    
    card_agg = train_df.groupby(CARD_COL).agg({
        AMOUNT_COL: ['mean', 'std', 'min', 'max', 'count'],
        TARGET_COL: ['mean', 'sum'],         # fraud rate and fraud count
        TIME_COL: ['min', 'max']             # time range of activity
    })
    
    # Flatten multi-level column names
    card_agg.columns = [
        'amt_mean', 'amt_std', 'amt_min', 'amt_max', 'tx_count',
        'fraud_rate', 'fraud_count',
        'first_tx_time', 'last_tx_time'
    ]
    
    # Add derived features
    card_agg['time_active'] = card_agg['last_tx_time'] - card_agg['first_tx_time']
    card_agg['amt_range'] = card_agg['amt_max'] - card_agg['amt_min']
    
    # Fill NaN in std (cards with single transaction have NaN std)
    card_agg['amt_std'] = card_agg['amt_std'].fillna(0)
    
    # Card labels: 1 if ANY transaction is fraud, else 0
    card_labels_series = (card_agg['fraud_count'] > 0).astype(int)
    
    # Select feature columns (exclude fraud_rate and fraud_count from features
    # because they ARE the target — using them would be data leakage)
    #
    # WAIT — This requires careful thinking:
    # 
    # If we are doing TRANSDUCTIVE learning (all nodes in graph during training,
    # but only train on train_mask nodes), then we cannot use fraud_rate as
    # a feature because test nodes would have it too.
    #
    # DECISION: Do NOT use fraud_rate or fraud_count as node features.
    # Use only transaction-level aggregated statistics.
    
    card_feature_cols = [
        'amt_mean', 'amt_std', 'amt_min', 'amt_max', 'tx_count',
        'first_tx_time', 'last_tx_time', 'time_active', 'amt_range'
    ]
    
    # Create feature tensor
    num_cards = len(card_node_to_idx)
    num_card_features = len(card_feature_cols)
    
    card_features = np.zeros((num_cards, num_card_features), dtype=np.float32)
    card_labels = np.zeros(num_cards, dtype=np.int64)
    
    for card_id, idx in card_node_to_idx.items():
        if card_id in card_agg.index:
            card_features[idx] = card_agg.loc[card_id, card_feature_cols].values
            card_labels[idx] = card_labels_series.loc[card_id]
    
    # Normalize features to zero mean and unit variance
    means = card_features.mean(axis=0)
    stds = card_features.std(axis=0)
    stds[stds == 0] = 1  # Prevent division by zero
    card_features = (card_features - means) / stds
    
    print(f"    Card features shape: {card_features.shape}")
    print(f"    Card labels — fraud: {card_labels.sum()}, "
          f"non-fraud: {len(card_labels) - card_labels.sum()}")
    
    # ====================================================================
    # Merchant Node Features
    # ====================================================================
    
    merchant_agg = train_df.groupby(MERCHANT_NODE_COL).agg({
        AMOUNT_COL: ['mean', 'std', 'min', 'max', 'count'],
        TARGET_COL: ['mean', 'sum'],
        TIME_COL: ['min', 'max']
    })
    
    merchant_agg.columns = [
        'amt_mean', 'amt_std', 'amt_min', 'amt_max', 'tx_count',
        'fraud_rate', 'fraud_count',
        'first_tx_time', 'last_tx_time'
    ]
    
    merchant_agg['time_active'] = merchant_agg['last_tx_time'] - merchant_agg['first_tx_time']
    merchant_agg['amt_range'] = merchant_agg['amt_max'] - merchant_agg['amt_min']
    merchant_agg['amt_std'] = merchant_agg['amt_std'].fillna(0)
    
    # For merchant nodes, we CAN use fraud_rate as a feature
    # because merchants are not what we're classifying
    # We're classifying CARD nodes
    # Merchant fraud rate is a legitimate signal
    merchant_feature_cols = [
        'amt_mean', 'amt_std', 'amt_min', 'amt_max', 'tx_count',
        'fraud_rate', 'fraud_count',
        'first_tx_time', 'last_tx_time', 'time_active', 'amt_range'
    ]
    
    num_merchants = len(merchant_node_to_idx)
    num_merchant_features = len(merchant_feature_cols)
    
    merchant_features = np.zeros((num_merchants, num_merchant_features), dtype=np.float32)
    
    for merchant_id, idx in merchant_node_to_idx.items():
        if merchant_id in merchant_agg.index:
            merchant_features[idx] = merchant_agg.loc[merchant_id, merchant_feature_cols].values
    
    # Normalize
    m_means = merchant_features.mean(axis=0)
    m_stds = merchant_features.std(axis=0)
    m_stds[m_stds == 0] = 1
    merchant_features = (merchant_features - m_means) / m_stds
    
    print(f"    Merchant features shape: {merchant_features.shape}")
    
    return (
        torch.tensor(card_features, dtype=torch.float),
        torch.tensor(merchant_features, dtype=torch.float),
        torch.tensor(card_labels, dtype=torch.long)
    )


def create_edge_index(train_df, card_node_to_idx, merchant_node_to_idx):
    """
    Create edge index tensor for card-merchant connections.
    
    In PyG HeteroData, edges are stored as:
        ('card', 'transacts_at', 'merchant') → edge_index of shape [2, num_edges]
        Row 0 = source node indices (card side)
        Row 1 = target node indices (merchant side)
    
    We also create reverse edges:
        ('merchant', 'rev_transacts_at', 'card')
    Because message passing in GNN needs to go both directions.
    
    Args:
        train_df: Training DataFrame
        card_node_to_idx: Dict mapping card IDs to indices
        merchant_node_to_idx: Dict mapping merchant IDs to indices
        
    Returns:
        tuple: (edge_index_card_to_merchant, edge_index_merchant_to_card, edge_features)
    """
    print("  Creating edge indices...")
    
    # Get unique card-merchant pairs (not individual transactions)
    # Multiple transactions between same card and merchant = one edge
    edge_df = train_df.groupby([CARD_COL, MERCHANT_NODE_COL]).agg({
        AMOUNT_COL: ['mean', 'sum', 'count'],
        TARGET_COL: ['sum']
    }).reset_index()
    
    edge_df.columns = [
        'card_id', 'merchant_id',
        'avg_amount', 'total_amount', 'tx_count',
        'fraud_count'
    ]
    
    # Build edge index
    src_indices = []  # card indices
    dst_indices = []  # merchant indices
    edge_features_list = []
    
    skipped = 0
    for _, row in edge_df.iterrows():
        card_id = row['card_id']
        merchant_id = row['merchant_id']
        
        # Skip if either node not in mapping
        if card_id not in card_node_to_idx:
            skipped += 1
            continue
        if merchant_id not in merchant_node_to_idx:
            skipped += 1
            continue
        
        src_indices.append(card_node_to_idx[card_id])
        dst_indices.append(merchant_node_to_idx[merchant_id])
        
        edge_features_list.append([
            float(row['avg_amount']),
            float(row['total_amount']),
            float(row['tx_count']),
            float(row['fraud_count'])
        ])
    
    if skipped > 0:
        print(f"    Skipped {skipped} edges (nodes not in mapping)")
    
    # Card → Merchant edges
    edge_index_c2m = torch.tensor(
        [src_indices, dst_indices],
        dtype=torch.long
    )
    
    # Merchant → Card edges (reverse)
    edge_index_m2c = torch.tensor(
        [dst_indices, src_indices],
        dtype=torch.long
    )
    
    # Edge features
    edge_features = torch.tensor(edge_features_list, dtype=torch.float)
    
    # Normalize edge features
    e_means = edge_features.mean(dim=0)
    e_stds = edge_features.std(dim=0)
    e_stds[e_stds == 0] = 1
    edge_features = (edge_features - e_means) / e_stds
    
    print(f"    Edges (card → merchant): {edge_index_c2m.shape[1]:,}")
    print(f"    Edges (merchant → card): {edge_index_m2c.shape[1]:,}")
    print(f"    Edge features dim: {edge_features.shape[1]}")
    
    return edge_index_c2m, edge_index_m2c, edge_features


def create_masks(train_df, test_df, card_node_to_idx):
    """
    Create train/test masks for card nodes.
    
    Cards that appear ONLY in training data → train_mask = True
    Cards that appear ONLY in test data → test_mask = True
    Cards that appear in BOTH → train_mask = True (we train on them,
                                 but we also have test data for them)
    
    For evaluation, we need a separate test set.
    Cards in test that were NOT in training cannot have graph features,
    so we evaluate on cards that exist in the graph.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        card_node_to_idx: Dict mapping card IDs to indices
        
    Returns:
        tuple: (train_mask, test_mask) both torch.BoolTensor of shape [num_cards]
    """
    print("  Creating train/test masks...")
    
    num_cards = len(card_node_to_idx)
    
    train_cards = set(train_df[CARD_COL].dropna().unique())
    test_cards = set(test_df[CARD_COL].dropna().unique())
    
    # Cards in both sets
    overlap_cards = train_cards & test_cards
    train_only_cards = train_cards - test_cards
    test_only_cards = test_cards - train_cards
    
    print(f"    Cards in training only: {len(train_only_cards):,}")
    print(f"    Cards in test only: {len(test_only_cards):,}")
    print(f"    Cards in both: {len(overlap_cards):,}")
    
    # Train mask: all cards that appear in training
    train_mask = torch.zeros(num_cards, dtype=torch.bool)
    for card_id in train_cards:
        if card_id in card_node_to_idx:
            train_mask[card_node_to_idx[card_id]] = True
    
    # Test mask: cards that appear in test AND exist in graph
    # Cards that are test-only don't exist in our graph (built from training)
    # So test_mask only covers overlap cards
    test_mask = torch.zeros(num_cards, dtype=torch.bool)
    for card_id in overlap_cards:
        if card_id in card_node_to_idx:
            test_mask[card_node_to_idx[card_id]] = True
    
    print(f"    Train mask: {train_mask.sum().item():,} nodes")
    print(f"    Test mask: {test_mask.sum().item():,} nodes")
    
    # Create test labels for overlap cards
    # Their label = did they have fraud in the TEST period?
    test_card_fraud = test_df.groupby(CARD_COL)[TARGET_COL].max()
    
    test_labels = torch.zeros(num_cards, dtype=torch.long)
    for card_id in overlap_cards:
        if card_id in card_node_to_idx and card_id in test_card_fraud.index:
            test_labels[card_node_to_idx[card_id]] = int(test_card_fraud[card_id])
    
    print(f"    Test fraud cards: {test_labels[test_mask].sum().item():,}")
    print(f"    Test non-fraud cards: {(test_mask.sum() - test_labels[test_mask].sum()).item():,}")
    
    return train_mask, test_mask, test_labels


def convert_to_pyg(train_df, test_df, card_node_to_idx, merchant_node_to_idx):
    """
    Main function: convert everything into a PyG HeteroData object.
    
    Args:
        train_df: Training DataFrame (with card, merchant, amount, fraud columns)
        test_df: Test DataFrame
        card_node_to_idx: Dict mapping card IDs to indices
        merchant_node_to_idx: Dict mapping merchant IDs to indices
        
    Returns:
        HeteroData: PyTorch Geometric heterogeneous graph data object
    """
    print("\n" + "=" * 80)
    print("CONVERTING TO PYTORCH GEOMETRIC FORMAT")
    print("=" * 80)
    
    # ====================================================================
    # Step 1: Create Node Features
    # ====================================================================
    print("\n[Step 1/4] Creating node features...")
    card_features, merchant_features, card_labels = create_node_features(
        train_df, card_node_to_idx, merchant_node_to_idx
    )
    
    # ====================================================================
    # Step 2: Create Edge Indices
    # ====================================================================
    print("\n[Step 2/4] Creating edge indices...")
    edge_index_c2m, edge_index_m2c, edge_features = create_edge_index(
        train_df, card_node_to_idx, merchant_node_to_idx
    )
    
    # ====================================================================
    # Step 3: Create Masks
    # ====================================================================
    print("\n[Step 3/4] Creating train/test masks...")
    train_mask, test_mask, test_labels = create_masks(
        train_df, test_df, card_node_to_idx
    )
    
    # ====================================================================
    # Step 4: Assemble HeteroData Object
    # ====================================================================
    print("\n[Step 4/4] Assembling HeteroData object...")
    
    data = HeteroData()
    
    # --- Card nodes ---
    data['card'].x = card_features
    data['card'].y = card_labels                    # Training labels
    data['card'].train_mask = train_mask
    data['card'].test_mask = test_mask
    data['card'].test_labels = test_labels          # Test period labels
    
    # --- Merchant nodes ---
    data['merchant'].x = merchant_features
    
    # --- Edges: Card → Merchant ---
    data['card', 'transacts_at', 'merchant'].edge_index = edge_index_c2m
    data['card', 'transacts_at', 'merchant'].edge_attr = edge_features
    
    # --- Edges: Merchant → Card (reverse for message passing) ---
    data['merchant', 'rev_transacts_at', 'card'].edge_index = edge_index_m2c
    data['merchant', 'rev_transacts_at', 'card'].edge_attr = edge_features
    
    # ====================================================================
    # Print Summary
    # ====================================================================
    print("\n" + "-" * 80)
    print("HeteroData Summary:")
    print(f"  Card nodes: {data['card'].x.shape[0]:,} "
          f"(features: {data['card'].x.shape[1]})")
    print(f"  Merchant nodes: {data['merchant'].x.shape[0]:,} "
          f"(features: {data['merchant'].x.shape[1]})")
    print(f"  Card → Merchant edges: {data['card', 'transacts_at', 'merchant'].edge_index.shape[1]:,}")
    print(f"  Merchant → Card edges: {data['merchant', 'rev_transacts_at', 'card'].edge_index.shape[1]:,}")
    print(f"  Edge features: {edge_features.shape[1]}")
    print(f"  Train mask: {train_mask.sum().item():,} cards")
    print(f"  Test mask: {test_mask.sum().item():,} cards")
    
    print("\n✓ PyG HeteroData conversion complete")
    
    return data
