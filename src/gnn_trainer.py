
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from src.gnn_model import FraudGAT
from src.config import (
    GNN_HIDDEN_DIM, GNN_NUM_HEADS, GNN_NUM_LAYERS, GNN_DROPOUT,
    GNN_OUTPUT_DIM, GNN_LEARNING_RATE, GNN_WEIGHT_DECAY,
    GNN_EPOCHS, GNN_PATIENCE, GNN_CHECKPOINT_PATH, LOG_EVERY_N_EPOCHS
)


def train_gnn(data):
    """
    Train the FraudGAT model.
    
    Args:
        data: PyG HeteroData object from pyg_converter
        
    Returns:
        dict: {
            'model': trained FraudGAT model,
            'history': training history dict,
            'best_metrics': dict of best validation metrics
        }
    """
    print("\n" + "=" * 80)
    print("TRAINING GRAPH ATTENTION NETWORK")
    print("=" * 80)
    
    # ====================================================================
    # Step 1: Setup Device
    # ====================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Setup] Device: {device}")
    
    # ====================================================================
    # Step 2: Get Data Dimensions
    # ====================================================================
    card_feature_dim = data['card'].x.shape[1]
    merchant_feature_dim = data['merchant'].x.shape[1]
    
    print(f"[Setup] Card feature dim: {card_feature_dim}")
    print(f"[Setup] Merchant feature dim: {merchant_feature_dim}")
    
    # ====================================================================
    # Step 3: Calculate Class Weights
    # ====================================================================
    train_mask = data['card'].train_mask
    train_labels = data['card'].y[train_mask]
    
    num_fraud = (train_labels == 1).sum().item()
    num_non_fraud = (train_labels == 0).sum().item()
    total = num_fraud + num_non_fraud
    
    # Weight = total / (2 * count_per_class)
    weight_non_fraud = total / (2 * num_non_fraud) if num_non_fraud > 0 else 1.0
    weight_fraud = total / (2 * num_fraud) if num_fraud > 0 else 1.0
    
    class_weights = torch.tensor([weight_non_fraud, weight_fraud], dtype=torch.float)
    
    print(f"[Setup] Training cards — fraud: {num_fraud:,}, non-fraud: {num_non_fraud:,}")
    print(f"[Setup] Class weights: [{weight_non_fraud:.4f}, {weight_fraud:.4f}]")
    
    # ====================================================================
    # Step 4: Initialize Model
    # ====================================================================
    model = FraudGAT(
        card_feature_dim=card_feature_dim,
        merchant_feature_dim=merchant_feature_dim,
        hidden_dim=GNN_HIDDEN_DIM,
        num_heads=GNN_NUM_HEADS,
        num_layers=GNN_NUM_LAYERS,
        dropout=GNN_DROPOUT,
        output_dim=GNN_OUTPUT_DIM
    )
    
    model = model.to(device)
    data = data.to(device)
    class_weights = class_weights.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Setup] Total parameters: {total_params:,}")
    print(f"[Setup] Trainable parameters: {trainable_params:,}")
    
    # ====================================================================
    # Step 5: Setup Training
    # ====================================================================
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=GNN_LEARNING_RATE,
        weight_decay=GNN_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7, verbose=False
    )
    
    # ====================================================================
    # Step 6: Training Loop
    # ====================================================================
    history = {
        'train_loss': [],
        'train_auc': [],
        'train_f1': [],
        'test_auc': [],
        'test_f1': [],
        'lr': []
    }
    
    best_test_auc = 0
    best_epoch = 0
    patience_counter = 0
    
    print(f"\n{'='*80}")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train AUC':>10} | "
          f"{'Test AUC':>10} | {'Test F1':>10} | {'LR':>10}")
    print(f"{'='*80}")
    
    # Prepare inputs
    x_dict = {
        'card': data['card'].x,
        'merchant': data['merchant'].x
    }
    edge_index_dict = {
        ('card', 'transacts_at', 'merchant'): data['card', 'transacts_at', 'merchant'].edge_index,
        ('merchant', 'rev_transacts_at', 'card'): data['merchant', 'rev_transacts_at', 'card'].edge_index
    }
    
    train_mask = data['card'].train_mask
    test_mask = data['card'].test_mask
    train_labels = data['card'].y
    test_labels = data['card'].test_labels
    
    for epoch in range(1, GNN_EPOCHS + 1):
        # ================================================================
        # Training Step
        # ================================================================
        model.train()
        optimizer.zero_grad()
        
        logits = model(x_dict, edge_index_dict)
        
        # Loss only on training nodes
        loss = criterion(logits[train_mask], train_labels[train_mask])
        
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # ================================================================
        # Evaluation Step
        # ================================================================
        model.eval()
        with torch.no_grad():
            logits = model(x_dict, edge_index_dict)
            probs = torch.softmax(logits, dim=1)[:, 1]  # Fraud probability
            
            # Training metrics
            train_probs = probs[train_mask].cpu().numpy()
            train_true = train_labels[train_mask].cpu().numpy()
            
            # Check if both classes present in training
            if len(np.unique(train_true)) > 1:
                train_auc = roc_auc_score(train_true, train_probs)
            else:
                train_auc = 0.0
            
            train_preds = (train_probs > 0.5).astype(int)
            train_f1 = f1_score(train_true, train_preds, zero_division=0)
            
            # Test metrics
            if test_mask.sum().item() > 0:
                test_probs = probs[test_mask].cpu().numpy()
                test_true = test_labels[test_mask].cpu().numpy()
                
                if len(np.unique(test_true)) > 1:
                    test_auc = roc_auc_score(test_true, test_probs)
                else:
                    test_auc = 0.0
                
                test_preds = (test_probs > 0.5).astype(int)
                test_f1 = f1_score(test_true, test_preds, zero_division=0)
            else:
                test_auc = 0.0
                test_f1 = 0.0
        
        # Record history
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(loss.item())
        history['train_auc'].append(train_auc)
        history['train_f1'].append(train_f1)
        history['test_auc'].append(test_auc)
        history['test_f1'].append(test_f1)
        history['lr'].append(current_lr)
        
        # Learning rate scheduler
        scheduler.step(test_auc)
        
        # Print progress
        if epoch % LOG_EVERY_N_EPOCHS == 0 or epoch == 1:
            print(f"{epoch:>6} | {loss.item():>10.4f} | {train_auc:>10.4f} | "
                  f"{test_auc:>10.4f} | {test_f1:>10.4f} | {current_lr:>10.6f}")
        
        # ================================================================
        # Early Stopping + Checkpointing
        # ================================================================
        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_auc': test_auc,
                'test_f1': test_f1,
                'train_auc': train_auc,
                'train_loss': loss.item()
            }, GNN_CHECKPOINT_PATH)
        else:
            patience_counter += 1
            if patience_counter >= GNN_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {GNN_PATIENCE} epochs)")
                break
    
    # ====================================================================
    # Step 7: Load Best Model
    # ====================================================================
    print(f"\n{'='*80}")
    print(f"Loading best model from epoch {best_epoch} "
          f"(test AUC: {best_test_auc:.4f})")
    
    checkpoint = torch.load(GNN_CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    best_metrics = {
        'best_epoch': best_epoch,
        'best_test_auc': checkpoint['test_auc'],
        'best_test_f1': checkpoint['test_f1'],
        'best_train_auc': checkpoint['train_auc'],
        'best_train_loss': checkpoint['train_loss'],
        'total_epochs': epoch,
        'total_params': total_params
    }
    
    print(f"Best epoch: {best_epoch}")
    print(f"Best test AUC: {best_metrics['best_test_auc']:.4f}")
    print(f"Best test F1: {best_metrics['best_test_f1']:.4f}")
    
    print("\n✓ GNN training complete")
    
    return {
        'model': model,
        'history': history,
        'best_metrics': best_metrics,
        'device': device,
        'data': data
    }
