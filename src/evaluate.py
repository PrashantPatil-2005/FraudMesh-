import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve
)
from src.config import PLOTS_DIR, METRICS_DIR

def evaluate_models(y_test, predictions_dict):
    """
    Evaluate baseline models and save results.
    
    Args:
        y_test: True test labels
        predictions_dict: Dictionary containing predictions from both models
    """
    print("\n" + "=" * 80)
    print("EVALUATING MODELS")
    print("=" * 80)
    
    results = {}
    
    # ========================================================================
    # Calculate Metrics for Each Model
    # ========================================================================
    for model_name, preds in predictions_dict.items():
        y_prob = preds['y_prob']
        y_pred = preds['y_pred']
        
        # Calculate metrics
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Confusion matrix for FPR and FNR
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        results[model_name] = {
            'auc': auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'fnr': fnr,
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            }
        }
    
    # ========================================================================
    # STEP 1: Print Comparison Table
    # ========================================================================
    print("\n" + "=" * 80)
    print("MODEL COMPARISON — PHASE 1 BASELINES")
    print("=" * 80)
    print(f"{'Metric':<25} {'LogReg':<12} {'RandomForest':<12}")
    print("-" * 80)
    
    metrics_to_show = ['auc', 'f1', 'precision', 'recall', 'fpr', 'fnr']
    metric_labels = {
        'auc': 'AUC-ROC',
        'f1': 'F1 Score',
        'precision': 'Precision',
        'recall': 'Recall',
        'fpr': 'False Positive Rate',
        'fnr': 'False Negative Rate'
    }
    
    for metric in metrics_to_show:
        label = metric_labels[metric]
        lr_val = results['lr'][metric]
        rf_val = results['rf'][metric]
        print(f"{label:<25} {lr_val:<12.4f} {rf_val:<12.4f}")
    
    print("=" * 80)
    
    # ========================================================================
    # STEP 2: Plot ROC Curves
    # ========================================================================
    print("\nPlotting ROC curves...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot diagonal baseline
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
    
    # Plot each model
    colors = {'lr': 'blue', 'rf': 'green'}
    labels = {'lr': 'Logistic Regression', 'rf': 'Random Forest'}
    
    for model_name, preds in predictions_dict.items():
        fpr_curve, tpr_curve, _ = roc_curve(y_test, preds['y_prob'])
        auc = results[model_name]['auc']
        
        ax.plot(fpr_curve, tpr_curve, color=colors[model_name],
               label=f"{labels[model_name]} (AUC = {auc:.4f})",
               linewidth=2)
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — Phase 1 Baselines', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'roc_curves_phase1.png', dpi=150)
    plt.close()
    print("✓ Saved: roc_curves_phase1.png")
    
    # ========================================================================
    # STEP 3: Plot Confusion Matrices
    # ========================================================================
    print("Plotting confusion matrices...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (model_name, preds) in enumerate(predictions_dict.items()):
        cm = confusion_matrix(y_test, preds['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   cbar=True, square=True)
        
        axes[idx].set_title(labels[model_name], fontsize=13, fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontsize=11)
        axes[idx].set_ylabel('Actual', fontsize=11)
        axes[idx].set_xticklabels(['Non-Fraud', 'Fraud'])
        axes[idx].set_yticklabels(['Non-Fraud', 'Fraud'])
    
    plt.suptitle('Confusion Matrices — Phase 1 Baselines', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'confusion_matrices_phase1.png', dpi=150)
    plt.close()
    print("✓ Saved: confusion_matrices_phase1.png")
    
    # ========================================================================
    # STEP 4: Plot Feature Importances (Random Forest)
    # ========================================================================
    print("Plotting feature importances...")
    
    feature_importances = predictions_dict['rf']['feature_importances']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    feature_importances.head(20).sort_values().plot(kind='barh', ax=ax, color='forestgreen')
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Top 20 Features — Random Forest', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'feature_importances_rf.png', dpi=150)
    plt.close()
    print("✓ Saved: feature_importances_rf.png")
    
    # ========================================================================
    # STEP 5: Save Metrics to JSON
    # ========================================================================
    print("Saving metrics to JSON...")
    
    # Prepare JSON-serializable results
    json_results = {
        'logistic_regression': {
            'auc': float(results['lr']['auc']),
            'f1': float(results['lr']['f1']),
            'precision': float(results['lr']['precision']),
            'recall': float(results['lr']['recall']),
            'false_positive_rate': float(results['lr']['fpr']),
            'false_negative_rate': float(results['lr']['fnr']),
            'confusion_matrix': results['lr']['confusion_matrix']
        },
        'random_forest': {
            'auc': float(results['rf']['auc']),
            'f1': float(results['rf']['f1']),
            'precision': float(results['rf']['precision']),
            'recall': float(results['rf']['recall']),
            'false_positive_rate': float(results['rf']['fpr']),
            'false_negative_rate': float(results['rf']['fnr']),
            'confusion_matrix': results['rf']['confusion_matrix']
        }
    }
    
    with open(METRICS_DIR / 'phase1_baselines.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print("✓ Saved: phase1_baselines.json")
    print("\n✓ Evaluation complete")
