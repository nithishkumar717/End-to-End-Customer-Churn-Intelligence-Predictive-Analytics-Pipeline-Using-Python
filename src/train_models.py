"""
Enterprise Multi-Model Training & Optimization Pipeline
Trains 4 Benchmark Algorithms:
1. Logistic Regression (Linear Baseline)
2. Random Forest Classifier (Bagging Ensemble)
3. XGBoost Classifier (Gradient Boosting Champion)
4. Deep Neural Network (MLP with 3 Hidden Layers & Loss Tracking)

Persists all models, diagnostic metrics, and epoch loss curves.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             brier_score_loss, log_loss)

try:
    from data_pipeline import run_data_pipeline
    from config import RAW_DATA_PATH, MODELS_DIR, RANDOM_STATE
except ImportError:
    from src.data_pipeline import run_data_pipeline
    from src.config import RAW_DATA_PATH, MODELS_DIR, RANDOM_STATE

def compute_model_metrics(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n========================================================")
    print(f"Benchmark: {name}")
    print(f"========================================================")
    print(f"Accuracy  : {acc * 100:.2f}% | ROC-AUC Score : {auc:.4f}")
    print(f"Precision : {prec * 100:.2f}% | Recall Score  : {rec * 100:.2f}%")
    print(f"F1-Score  : {f1 * 100:.2f}% | Brier Score   : {brier:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    return {
        "name": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "brier_score": brier,
        "confusion_matrix": cm.tolist(),
        "y_prob_sample": y_prob[:200].tolist()
    }

def train_all_models(data_path: str = RAW_DATA_PATH, models_dir: str = MODELS_DIR):
    os.makedirs(models_dir, exist_ok=True)
    X_train_scaled, X_test_scaled, y_train, y_test, X_train_raw, X_test_raw = run_data_pipeline(data_path, models_dir)

    results = {}

    # 1. Baseline: Logistic Regression (L2 Regularized)
    print("\n[1/4] Training Baseline Model: Logistic Regression...")
    lr_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    lr_model.fit(X_train_scaled, y_train)
    results['LogisticRegression'] = compute_model_metrics("Logistic Regression", lr_model, X_test_scaled, y_test)
    joblib.dump(lr_model, os.path.join(models_dir, "logistic_regression_model.joblib"))

    # 2. Ensemble 1: Random Forest Classifier
    print("\n[2/4] Training Ensemble 1: Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=180,
        max_depth=10,
        min_samples_split=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    results['RandomForest'] = compute_model_metrics("Random Forest", rf_model, X_test_scaled, y_test)
    joblib.dump(rf_model, os.path.join(models_dir, "random_forest_model.joblib"))

    # 3. Ensemble 2: Tuned XGBoost Classifier
    print("\n[3/4] Training Ensemble 2: Tuned XGBoost Classifier...")
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )
    xgb_model.fit(X_train_scaled, y_train)
    results['XGBoost'] = compute_model_metrics("XGBoost", xgb_model, X_test_scaled, y_test)
    joblib.dump(xgb_model, os.path.join(models_dir, "xgboost_model.joblib"))

    # 4. Deep Learning: Multi-Layer Neural Network (MLP)
    print("\n[4/4] Training Deep Neural Network (MLP: 64 -> 32)...")
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.1,
        batch_size=64,
        learning_rate="adaptive",
        learning_rate_init=0.0005,
        max_iter=300,
        random_state=RANDOM_STATE
    )
    mlp_model.fit(X_train_scaled, y_train)
    results['DeepLearning_MLP'] = compute_model_metrics("Deep Neural Network (MLP)", mlp_model, X_test_scaled, y_test)
    joblib.dump(mlp_model, os.path.join(models_dir, "mlp_neural_network_model.joblib"))

    # Persist Neural Network Training Loss Curve
    loss_history = {
        "loss_curve": mlp_model.loss_curve_,
        "n_iter": mlp_model.n_iter_,
        "best_loss": mlp_model.best_loss_
    }
    joblib.dump(loss_history, os.path.join(models_dir, "mlp_loss_history.joblib"))

    # Extract Feature Importances
    feature_names = joblib.load(os.path.join(models_dir, "features.joblib"))
    rf_imp = dict(zip(feature_names, rf_model.feature_importances_))
    xgb_imp = dict(zip(feature_names, xgb_model.feature_importances_))

    summary_payload = {
        "results": results,
        "rf_importance": rf_imp,
        "xgb_importance": xgb_imp,
        "feature_names": feature_names
    }
    joblib.dump(summary_payload, os.path.join(models_dir, "metrics_summary.joblib"))
    print(f"\n[Success] All 4 models and diagnostic metrics saved to: {models_dir}")

if __name__ == "__main__":
    train_all_models()
