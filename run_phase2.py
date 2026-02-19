
"""
FraudMesh-RL Phase 2
Run GNN pipeline: reuse Phase 1 data, convert to PyG, train GAT, evaluate.
"""

from src.data_loader import load_data
from src.feature_engineer import engineer_features
from src.pyg_converter import convert_to_pyg
from src.gnn_trainer import train_gnn
from src.gnn_evaluate import evaluate_gnn


def main():
    """Run Phase 2 pipeline."""
    
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  FRAUDMESH-RL — PHASE 2".center(78) + "║")
    print("║" + "  Graph Neural Network (GAT)".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")
    
    try:
        # ====================================================================
        # Step 1: Load Data (reuse Phase 1)
        # ====================================================================
        print("[1/5] LOADING DATA")
        print("-" * 80)
        df = load_data()
        
        # ====================================================================
        # Step 2: Feature Engineering (reuse Phase 1)
        # ====================================================================
        print("\n[2/5] FEATURE ENGINEERING")
        print("-" * 80)
        data = engineer_features(df)
        
        # Free raw dataframe memory
        del df
        
        # ====================================================================
        # Step 3: Convert to PyG Format
        # ====================================================================
        print("\n[3/5] CONVERTING TO PYTORCH GEOMETRIC")
        print("-" * 80)
        pyg_data = convert_to_pyg(
            train_df=data['train_df'],
            test_df=data['test_df'],
            card_node_to_idx=data['card_node_to_idx'],
            merchant_node_to_idx=data['merchant_node_to_idx']
        )
        
        # Free Phase 1 data from memory (keep only PyG data)
        del data
        
        # ====================================================================
        # Step 4: Train GNN
        # ====================================================================
        print("\n[4/5] TRAINING GNN")
        print("-" * 80)
        train_result = train_gnn(pyg_data)
        
        # ====================================================================
        # Step 5: Evaluate GNN
        # ====================================================================
        print("\n[5/5] EVALUATING GNN")
        print("-" * 80)
        evaluate_gnn(train_result)
        
        # ====================================================================
        # Final Summary
        # ====================================================================
        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "  PHASE 2 COMPLETE ✓".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "═" * 78 + "╝")
        print("\n")
        
        best = train_result['best_metrics']
        
        print("RESULTS SUMMARY:")
        print("-" * 80)
        print(f"GNN Architecture: Graph Attention Network (GAT)")
        print(f"  • Best Epoch: {best['best_epoch']}")
        print(f"  • Test AUC: {best['best_test_auc']:.4f}")
        print(f"  • Test F1: {best['best_test_f1']:.4f}")
        print(f"  • Parameters: {best['total_params']:,}")
        print(f"\nOutput Files:")
        print(f"  • Training curves: outputs/plots/gnn_training_curves.png")
        print(f"  • Confusion matrix: outputs/plots/gnn_confusion_matrix.png")
        print(f"  • ROC curve: outputs/plots/gnn_roc_curve.png")
        print(f"  • Model comparison: outputs/plots/model_comparison_auc.png")
        print(f"  • GNN metrics: outputs/metrics/phase2_gnn.json")
        print(f"  • Full comparison: outputs/metrics/full_comparison.json")
        print(f"  • Model checkpoint: models/best_gnn_model.pt")
        print(f"\nNext Steps:")
        print(f"  • Review the comparison table")
        print(f"  • Check training curves for overfitting")
        print(f"  • Ready for Phase 3: RL response agent")
        print("-" * 80)
        print("\n")
        
    except FileNotFoundError as e:
        print("\n❌ ERROR: Required files not found!")
        print(f"   {str(e)}")
        print("\nMake sure Phase 1 has been run successfully first.")
        print("Run: python run_phase1.py")
        print()
        
    except ImportError as e:
        print("\n❌ ERROR: Missing dependency!")
        print(f"   {str(e)}")
        print("\nPlease install PyTorch Geometric:")
        print("  pip install torch torch-geometric")
        print("  pip install torch-scatter torch-sparse torch-cluster "
              "-f https://data.pyg.org/whl/torch-2.1.2+cpu.html")
        print()
        
    except Exception as e:
        print("\n❌ ERROR occurred during execution:")
        print(f"   {type(e).__name__}: {str(e)}")
        print()
        raise


if __name__ == "__main__":
    main()
