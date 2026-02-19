
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve
)
from src.config import PLOTS_DIR, METRICS_DIR


def evaluate_gnn(train_result):
    """
    Evaluate trained GNN and compare with Phase 1 baselines.
    
    Args:
        train_result: Dictionary returned by train_gnn()
    """
    print("\n" + "=" * 80)
    print("EVALUATING GNN MODEL")
    print("=" * 80)
    
    model = train_result['model']
    history = train_result['history']
    best_metrics = train_result['best_metrics']
    device = train_result['device']
    data = train_result['data']
    
    # ====================================================================
    # Step 1: Get GNN Predictions
    # ====================================================================
    print("\n[Step 1/5] Getting GNN predictions...")
    
    model.eval()
    with torch.no_grad():
        x_dict = {
            'card': data['card'].x,
            'merchant': data['merchant'].x
        }
        edge_index_dict = {
            ('card', 'transacts_at', 'merchant'): 
                data['card', 'transacts_at', 'merchant'].edge_index,
            ('merchant', 'rev_transacts_at', 'card'): 
                data['merchant', 'rev_transacts_at', 'card'].edge_index
        }
        
        logits = model(x_dict, edge_index_dict)
        probs = torch.softmax(logits, dim=1)[:, 1]
    
    # Get test predictions
    test_mask = data['card'].test_mask
    test_labels = data['card'].test_labels
    
    test_probs = probs[test_mask].cpu().numpy()
    test_true = test_labels[test_mask].cpu().numpy()
    test_preds = (test_probs > 0.5).astype(int)
    
    print(f"  Test samples: {len(test_true):,}")
    print(f"  Test fraud: {test_true.sum():,}")
    print(f"  Test non-fraud: {(1 - test_true).sum():,}")
    
    # ====================================================================
    # Step 2: Calculate GNN Metrics
    # ====================================================================
    print("\n[Step 2/5] Calculating GNN metrics...")
    
    # Check if both classes are present
    if len(np.unique(test_true)) < 2:
        print("  WARNING: Only one class in test set!")
        print("  AUC cannot be calculated. Using accuracy instead.")
        gnn_auc = 0.0
    else:
        gnn_auc = roc_auc_score(test_true, test_probs)
    
    gnn_f1 = f1_score(test_true, test_preds, zero_division=0)
    gnn_precision = precision_score(test_true, test_preds, zero_division=0)
    gnn_recall = recall_score(test_true, test_preds, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(test_true, test_preds).ravel() \
        if len(np.unique(test_preds)) > 1 else (0, 0, 0, 0)
    
    gnn_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    gnn_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    gnn_metrics = {
        'auc': float(gnn_auc),
        'f1': float(gnn_f1),
        'precision': float(gnn_precision),
        'recall': float(gnn_recall),
        'fpr': float(gnn_fpr),
        'fnr': float(gnn_fnr),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        }
    }
    
    # ====================================================================
    # Step 3: Load Phase 1 Baselines and Compare
    # ====================================================================
    print("\n[Step 3/5] Comparing with Phase 1 baselines...")
    
    # Load Phase 1 results
    phase1_path = METRICS_DIR / 'phase1_baselines.json'
    
    try:
        with open(phase1_path, 'r') as f:
            phase1_results = json.load(f)
        phase1_loaded = True
    except FileNotFoundError:
        print(f"  WARNING: {phase1_path} not found. Showing GNN results only.")
        phase1_loaded = False
        phase1_results = {}
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("MODEL COMPARISON — PHASE 1 vs PHASE 2")
    print("=" * 80)
    
    if phase1_loaded:
        print(f"{'Metric':<25} {'LogReg':<12} {'RandForest':<12} {'GNN (GAT)':<12}")
        print("-" * 80)
        
        metrics_map = {
            'AUC-ROC': 'auc',
            'F1 Score': 'f1',
            'Precision': 'precision',
            'Recall': 'recall',
            'False Positive Rate': ('false_positive_rate', 'fpr'),
            'False Negative Rate': ('false_negative_rate', 'fnr')
        }
        
        for display_name, key_info in metrics_map.items():
            if isinstance(key_info, tuple):
                p1_key, gnn_key = key_info
            else:
                p1_key = key_info
                gnn_key = key_info
            
            lr_val = phase1_results.get('logistic_regression', {}).get(p1_key, 'N/A')
            rf_val = phase1_results.get('random_forest', {}).get(p1_key, 'N/A')
            gnn_val = gnn_metrics[gnn_key]
            
            if isinstance(lr_val, (int, float)):
                lr_str = f"{lr_val:.4f}"
            else:
                lr_str = str(lr_val)
                
            if isinstance(rf_val, (int, float)):
                rf_str = f"{rf_val:.4f}"
            else:
                rf_str = str(rf_val)
            
            print(f"{display_name:<25} {lr_str:<12} {rf_str:<12} {gnn_val:<12.4f}")
        
        print("=" * 80)
    else:
        print(f"{'Metric':<25} {'GNN (GAT)':<12}")
        print("-" * 80)
        for name, key in [('AUC-ROC', 'auc'), ('F1 Score', 'f1'),
                          ('Precision', 'precision'), ('Recall', 'recall'),
                          ('False Positive Rate', 'fpr'), ('False Negative Rate', 'fnr')]:
            print(f"{name:<25} {gnn_metrics[key]:<12.4f}")
        print("=" * 80)
    
    print("\nIMPORTANT NOTE:")
    print("  Phase 1 baselines were evaluated on TRANSACTION-level predictions.")
    print("  GNN was evaluated on CARD-level predictions (node classification).")
    print("  These are different evaluation setups.")
    print("  The comparison shows relative model capability, not exact apples-to-apples.")
    
    # ====================================================================
    # Step 4: Plot Training Curves
    # ====================================================================
    print("\n[Step 4/5] Plotting training curves...")
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss curve
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training loss
    axes[0, 0].plot(epochs, history['train_loss'], color='red', linewidth=1.5)
    axes[0, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(alpha=0.3)
    
    # AUC curves
    axes[0, 1].plot(epochs, history['train_auc'], label='Train AUC', 
                    color='blue', linewidth=1.5)
    axes[0, 1].plot(epochs, history['test_auc'], label='Test AUC', 
                    color='green', linewidth=1.5)
    axes[0, 1].axhline(y=best_metrics['best_test_auc'], color='green', 
                       linestyle='--', alpha=0.5, label=f"Best: {best_metrics['best_test_auc']:.4f}")
    axes[0, 1].set_title('AUC-ROC Over Training', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # F1 curves
    axes[1, 0].plot(epochs, history['train_f1'], label='Train F1', 
                    color='blue', linewidth=1.5)
    axes[1, 0].plot(epochs, history['test_f1'], label='Test F1', 
                    color='green', linewidth=1.5)
    axes[1, 0].set_title('F1 Score Over Training', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Learning rate
    axes[1, 1].plot(epochs, history['lr'], color='purple', linewidth=1.5)
    axes[1, 1].set_title('Learning Rate', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('LR')
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle('GNN Training Curves', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'gnn_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: gnn_training_curves.png")
    
    # Plot 2: GNN Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(test_true, test_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, square=True)
    ax.set_title('GNN Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticklabels(['Non-Fraud', 'Fraud'])
    ax.set_yticklabels(['Non-Fraud', 'Fraud'])
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'gnn_confusion_matrix.png', dpi=150)
    plt.close()
    print("  ✓ Saved: gnn_confusion_matrix.png")
    
    # Plot 3: ROC Curve (GNN only, Phase 1 curves already saved)
    if len(np.unique(test_true)) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        fpr_curve, tpr_curve, _ = roc_curve(test_true, test_probs)
        ax.plot(fpr_curve, tpr_curve, color='red', linewidth=2,
               label=f'GNN GAT (AUC = {gnn_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve — GNN', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'gnn_roc_curve.png', dpi=150)
        plt.close()
        print("  ✓ Saved: gnn_roc_curve.png")
    
    # Plot 4: Comparison Bar Chart
    if phase1_loaded:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = ['Logistic\nRegression', 'Random\nForest', 'GNN\n(GAT)']
        
        lr_auc = phase1_results.get('logistic_regression', {}).get('auc', 0)
        rf_auc = phase1_results.get('random_forest', {}).get('auc', 0)
        aucs = [lr_auc, rf_auc, gnn_auc]
        
        colors = ['steelblue', 'forestgreen', 'crimson']
        bars = ax.bar(models, aucs, color=colors, width=0.5, edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                   f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title('Model Comparison — AUC-ROC', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'model_comparison_auc.png', dpi=150)
        plt.close()
        print("  ✓ Saved: model_comparison_auc.png")
    
    # ====================================================================
    # Step 5: Save Metrics
    # ====================================================================
    print("\n[Step 5/5] Saving metrics...")
    
    # Save GNN metrics
    gnn_results = {
        'gnn_gat': gnn_metrics,
        'training_info': {
            'best_epoch': best_metrics['best_epoch'],
            'total_epochs': best_metrics['total_epochs'],
            'total_parameters': best_metrics['total_params'],
            'best_train_auc': best_metrics['best_train_auc'],
            'best_train_loss': best_metrics['best_train_loss']
        },
        'note': 'GNN evaluation is node-level (per card). '
                'Phase 1 baselines are transaction-level. '
                'Not directly comparable but shows relative capability.'
    }
    
    with open(METRICS_DIR / 'phase2_gnn.json', 'w') as f:
        json.dump(gnn_results, f, indent=2)
    
    print("  ✓ Saved: phase2_gnn.json")
    
    # Save combined comparison
    if phase1_loaded:
        combined = {
            'phase1_baselines': phase1_results,
            'phase2_gnn': gnn_results
        }
        
        with open(METRICS_DIR / 'full_comparison.json', 'w') as f:
            json.dump(combined, f, indent=2)
        
        print("  ✓ Saved: full_comparison.json")
    
    print("\n✓ GNN evaluation complete")
