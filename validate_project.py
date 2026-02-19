"""
FraudMesh-RL — Project Validator

Checks that all project files and outputs exist.

Run: python validate_project.py
"""

from pathlib import Path
import json


def check_file(path, required=True):
    """Check if a file exists and return status."""
    exists = Path(path).exists()
    status = "✓" if exists else ("✗ MISSING" if required else "○ optional")
    return exists, status


def validate():
    """Run all validation checks."""
    
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + "  FRAUDMESH-RL — PROJECT VALIDATION".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")
    
    root = Path('.')
    all_good = True
    
    # ================================================================
    # Source Files
    # ================================================================
    print("SOURCE FILES")
    print("-" * 60)
    
    source_files = [
        ('src/__init__.py', True),
        ('src/config.py', True),
        ('src/data_loader.py', True),
        ('src/eda.py', True),
        ('src/feature_engineer.py', True),
        ('src/graph_builder.py', True),
        ('src/baseline_model.py', True),
        ('src/evaluate.py', True),
        ('src/pyg_converter.py', True),
        ('src/gnn_model.py', True),
        ('src/gnn_trainer.py', True),
        ('src/gnn_evaluate.py', True),
        ('src/rl_environment.py', True),
        ('src/replay_buffer.py', True),
        ('src/dqn_model.py', True),
        ('src/dqn_agent.py', True),
        ('src/rl_trainer.py', True),
        ('src/rl_evaluate.py', True),
        ('src/rl_baselines.py', True),
    ]
    
    for filepath, required in source_files:
        exists, status = check_file(root / filepath, required)
        print(f"  {status}  {filepath}")
        if required and not exists:
            all_good = False
    
    # ================================================================
    # Entry Points
    # ================================================================
    print("\nENTRY POINTS")
    print("-" * 60)
    
    entry_files = [
        ('run_phase1.py', True),
        ('run_phase2.py', True),
        ('run_phase3.py', True),
        ('demo.py', True),
        ('generate_report.py', True),
        ('validate_project.py', True),
    ]
    
    for filepath, required in entry_files:
        exists, status = check_file(root / filepath, required)
        print(f"  {status}  {filepath}")
        if required and not exists:
            all_good = False
    
    # ================================================================
    # Config Files
    # ================================================================
    print("\nCONFIG FILES")
    print("-" * 60)
    
    config_files = [
        ('requirements.txt', True),
        ('.gitignore', True),
        ('README.md', True),
    ]
    
    for filepath, required in config_files:
        exists, status = check_file(root / filepath, required)
        print(f"  {status}  {filepath}")
        if required and not exists:
            all_good = False
    
    # ================================================================
    # Data Files
    # ================================================================
    print("\nDATA FILES")
    print("-" * 60)
    
    data_files = [
        ('data/train_transaction.csv', True),
        ('data/train_identity.csv', True),
    ]
    
    for filepath, required in data_files:
        exists, status = check_file(root / filepath, required)
        size = ""
        if exists:
            size_mb = Path(filepath).stat().st_size / (1024 * 1024)
            size = f" ({size_mb:.1f} MB)"
        print(f"  {status}  {filepath}{size}")
        if required and not exists:
            all_good = False
    
    # ================================================================
    # Output Plots
    # ================================================================
    print("\nOUTPUT PLOTS")
    print("-" * 60)
    
    plot_files = [
        # Phase 1
        ('outputs/plots/class_distribution.png', False),
        ('outputs/plots/missing_values.png', False),
        ('outputs/plots/amount_distribution.png', False),
        ('outputs/plots/fraud_correlations.png', False),
        ('outputs/plots/sample_subgraph.png', False),
        ('outputs/plots/roc_curves_phase1.png', False),
        ('outputs/plots/confusion_matrices_phase1.png', False),
        ('outputs/plots/feature_importances_rf.png', False),
        # Phase 2
        ('outputs/plots/gnn_training_curves.png', False),
        ('outputs/plots/gnn_confusion_matrix.png', False),
        ('outputs/plots/gnn_roc_curve.png', False),
        ('outputs/plots/model_comparison_auc.png', False),
        # Phase 3
        ('outputs/plots/rl_training_curves.png', False),
        ('outputs/plots/rl_reward_comparison.png', False),
        ('outputs/plots/rl_action_distribution.png', False),
        ('outputs/plots/rl_cost_analysis.png', False),
        # Demo
        ('outputs/plots/demo_full_pipeline.png', False),
    ]
    
    plot_count = 0
    for filepath, required in plot_files:
        exists, status = check_file(root / filepath, required)
        print(f"  {status}  {filepath}")
        if exists:
            plot_count += 1
    
    # ================================================================
    # Output Metrics
    # ================================================================
    print("\nOUTPUT METRICS")
    print("-" * 60)
    
    metric_files = [
        ('outputs/metrics/phase1_baselines.json', False),
        ('outputs/metrics/phase2_gnn.json', False),
        ('outputs/metrics/rl_training_metrics.json', False),
        ('outputs/metrics/rl_baseline_metrics.json', False),
        ('outputs/metrics/phase3_combined_comparison.json', False),
    ]
    
    metric_count = 0
    for filepath, required in metric_files:
        exists, status = check_file(root / filepath, required)
        
        # Try to peek inside JSON
        detail = ""
        if exists:
            metric_count += 1
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                detail = f" ({len(data)} top-level keys)"
            except Exception:
                detail = ""
        
        print(f"  {status}  {filepath}{detail}")
    
    # ================================================================
    # Model Checkpoints
    # ================================================================
    print("\nMODEL CHECKPOINTS")
    print("-" * 60)
    
    model_files = [
        ('models/best_gnn_model.pt', False),
        ('models/best_dqn_model.pt', False),
    ]
    
    for filepath, required in model_files:
        exists, status = check_file(root / filepath, required)
        size = ""
        if exists:
            size_mb = Path(filepath).stat().st_size / (1024 * 1024)
            size = f" ({size_mb:.1f} MB)"
        print(f"  {status}  {filepath}{size}")
    
    # ================================================================
    # Report
    # ================================================================
    print("\nREPORT")
    print("-" * 60)
    
    report_exists, status = check_file('outputs/PROJECT_REPORT.md', False)
    print(f"  {status}  outputs/PROJECT_REPORT.md")
    
    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\n  Source files:     19 required")
    print(f"  Plots generated:  {plot_count}/17")
    print(f"  Metrics saved:    {metric_count}/5")
    print(f"  Report:           {'✓' if report_exists else '✗ Run: python generate_report.py'}")
    
    # Phase completion
    p1 = Path('outputs/metrics/phase1_baselines.json').exists()
    p2 = Path('outputs/metrics/phase2_gnn.json').exists()
    p3 = Path('outputs/metrics/phase3_combined_comparison.json').exists()
    
    print(f"\n  Phase 1 (Baselines): {'✓ COMPLETE' if p1 else '✗ Run: python run_phase1.py'}")
    print(f"  Phase 2 (GNN):       {'✓ COMPLETE' if p2 else '✗ Run: python run_phase2.py'}")
    print(f"  Phase 3 (RL):        {'✓ COMPLETE' if p3 else '✗ Run: python run_phase3.py'}")
    
    if p1 and p2 and p3:
        print("\n  🎉 ALL PHASES COMPLETE — PROJECT READY FOR GITHUB")
    elif all_good:
        print("\n  Source files ready. Run the phases to generate outputs.")
    else:
        print("\n  ⚠ Some required files are missing. Check above.")
    
    print("\n")


if __name__ == "__main__":
    validate()
