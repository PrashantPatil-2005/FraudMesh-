import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_baselines(X_train, y_train, X_test, y_test, class_weight):
    """
    Train baseline models (Logistic Regression and Random Forest).
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        class_weight: Class weight ratio
        
    Returns:
        dict: Predictions and models for both baselines
    """
    print("\n" + "=" * 80)
    print("TRAINING BASELINE MODELS")
    print("=" * 80)
    
    # ========================================================================
    # STEP 1: Data Validation
    # ========================================================================
    print("\n[Step 1/3] Validating data...")
    
    # Replace infinite values with NaN, then fill with 0
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"  Training shape: {X_train.shape}")
    print(f"  Test shape: {X_test.shape}")
    print(f"  Features: {X_train.shape[1]}")
    
    # Verify all numeric
    assert X_train.select_dtypes(include=[np.number]).shape[1] == X_train.shape[1], \
        "Non-numeric columns found in training features!"
    assert X_test.select_dtypes(include=[np.number]).shape[1] == X_test.shape[1], \
        "Non-numeric columns found in test features!"
    
    print("✓ Data validation passed")
    
    # ========================================================================
    # STEP 2: Train Logistic Regression
    # ========================================================================
    print("\n[Step 2/3] Training Logistic Regression...")
    
    lr_model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        solver='lbfgs',
        n_jobs=-1,
        verbose=0
    )
    
    print("  Fitting model...")
    lr_model.fit(X_train, y_train)
    
    print("  Generating predictions...")
    y_prob_lr = lr_model.predict_proba(X_test)[:, 1]
    y_pred_lr = lr_model.predict(X_test)
    
    print("✓ Logistic Regression trained")
    
    # ========================================================================
    # STEP 3: Train Random Forest
    # ========================================================================
    print("\n[Step 3/3] Training Random Forest...")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        max_depth=10,
        min_samples_leaf=50,
        verbose=0
    )
    
    print("  Fitting model...")
    rf_model.fit(X_train, y_train)
    
    print("  Generating predictions...")
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    y_pred_rf = rf_model.predict(X_test)
    
    # Extract feature importances
    feature_importances = pd.Series(
        rf_model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)
    
    print("✓ Random Forest trained")
    print("\nTop 15 most important features:")
    for i, (feat, importance) in enumerate(feature_importances.head(15).items(), 1):
        print(f"  {i:2d}. {feat:30s}: {importance:.4f}")
    
    # ========================================================================
    # Return Predictions
    # ========================================================================
    print("\n✓ Baseline training complete")
    
    return {
        'lr': {
            'y_prob': y_prob_lr,
            'y_pred': y_pred_lr,
            'model': lr_model
        },
        'rf': {
            'y_prob': y_prob_rf,
            'y_pred': y_pred_rf,
            'model': rf_model,
            'feature_importances': feature_importances
        }
    }
