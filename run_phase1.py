"""
FraudMesh-RL Phase 1
Run complete baseline pipeline: data loading, EDA, feature engineering,
graph construction, baseline training, and evaluation.
"""

from src.data_loader import load_data
from src.eda import run_eda
from src.feature_engineer import engineer_features
from src.graph_builder import build_graph
from src.baseline_model import train_baselines
from src.evaluate import evaluate_models

def main():
    """Run Phase 1 pipeline."""
    
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  FRAUDMESH-RL — PHASE 1".center(78) + "║")
    print("║" + "  Data Exploration + Baseline Models".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")
    
    try:
        # ====================================================================
        # Step 1: Load Data
        # ====================================================================
        print("[1/6] LOADING DATA")
        print("-" * 80)
        df = load_data()
        
        # ====================================================================
        # Step 2: Exploratory Data Analysis
        # ====================================================================
        print("\n[2/6] RUNNING EDA")
        print("-" * 80)
        run_eda(df)
        
        # ====================================================================
        # Step 3: Feature Engineering
        # ====================================================================
        print("\n[3/6] FEATURE ENGINEERING")
        print("-" * 80)
        data = engineer_features(df)
        
        # Free up memory
        del df
        
        # ====================================================================
        # Step 4: Build Transaction Graph
        # ====================================================================
        print("\n[4/6] BUILDING TRANSACTION GRAPH")
        print("-" * 80)
        G = build_graph(
            data['train_df'],
            data['card_node_to_idx'],
            data['merchant_node_to_idx']
        )
        
        # ====================================================================
        # Step 5: Train Baseline Models
        # ====================================================================
        print("\n[5/6] TRAINING BASELINE MODELS")
        print("-" * 80)
        predictions = train_baselines(
            data['X_train'],
            data['y_train'],
            data['X_test'],
            data['y_test'],
            data['class_weight']
        )
        
        # ====================================================================
        # Step 6: Evaluate Models
        # ====================================================================
        print("\n[6/6] EVALUATING MODELS")
        print("-" * 80)
        evaluate_models(data['y_test'], predictions)
        
        # ====================================================================
        # Final Summary
        # ====================================================================
        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "  PHASE 1 COMPLETE ✓".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "═" * 78 + "╝")
        print("\n")
        
        print("RESULTS SUMMARY:")
        print("-" * 80)
        print(f"Graph Statistics:")
        print(f"  • Nodes: {G.number_of_nodes():,}")
        print(f"  • Edges: {G.number_of_edges():,}")
        print(f"\nOutput Files:")
        print(f"  • Plots saved to: outputs/plots/")
        print(f"  • Metrics saved to: outputs/metrics/phase1_baselines.json")
        print(f"\nNext Steps:")
        print(f"  • Review the baseline metrics")
        print(f"  • Check the visualizations")
        print(f"  • Ready for Phase 2: GNN implementation")
        print("-" * 80)
        print("\n")
        
    except FileNotFoundError as e:
        print("\n❌ ERROR: Data files not found!")
        print(f"   {str(e)}")
        print("\nPlease ensure you have downloaded the IEEE-CIS Fraud Detection dataset:")
        print("  1. Go to: https://www.kaggle.com/c/ieee-fraud-detection/data")
        print("  2. Download train_transaction.csv and train_identity.csv")
        print("  3. Place them in the 'data/' folder")
        print()
        
    except Exception as e:
        print("\n❌ ERROR occurred during execution:")
        print(f"   {type(e).__name__}: {str(e)}")
        print("\nPlease check the error message above and fix the issue.")
        print()
        raise

if __name__ == "__main__":
    main()
