"""
Model Training & Optimization Pipeline
Trains Random Forest, XGBoost, and Deep Neural Network (MLP) Classifiers,
evaluates performance metrics, and saves trained models for real-time inference.
"""
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, classification_report)

from data_pipeline import run_data_pipeline

def evaluate_predictions(name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n==========================================")
    print(f"Model: {name}")
    print(f"==========================================")
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1-Score  : {f1 * 100:.2f}%")
    print(f"ROC-AUC   : {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    return {
        "name": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm.tolist()
    }

def train_all_models(data_path: str, models_dir: str):
    X_train_scaled, X_test_scaled, y_train, y_test, X_train_raw, X_test_raw = run_data_pipeline(data_path, models_dir)
    
    results = {}
    
    # 1. Random Forest Classifier
    print("\nTraining Model 1/3: Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=9,
        min_samples_split=8,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_preds = rf_model.predict(X_test_scaled)
    rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
    results['RandomForest'] = evaluate_predictions("Random Forest", y_test, rf_preds, rf_probs)
    joblib.dump(rf_model, os.path.join(models_dir, "random_forest_model.joblib"))
    
    # 2. XGBoost Classifier
    print("\nTraining Model 2/3: XGBoost Classifier...")
    xgb_model = XGBClassifier(
        n_estimators=160,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42
    )
    xgb_model.fit(X_train_scaled, y_train)
    xgb_preds = xgb_model.predict(X_test_scaled)
    xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]
    results['XGBoost'] = evaluate_predictions("XGBoost", y_test, xgb_preds, xgb_probs)
    joblib.dump(xgb_model, os.path.join(models_dir, "xgboost_model.joblib"))
    
    # 3. Deep Learning Multi-Layer Perceptron (Neural Network)
    print("\nTraining Model 3/3: Deep Neural Network (MLPClassifier)...")
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=64,
        learning_rate="adaptive",
        max_iter=350,
        early_stopping=True,
        random_state=42
    )
    mlp_model.fit(X_train_scaled, y_train)
    mlp_preds = mlp_model.predict(X_test_scaled)
    mlp_probs = mlp_model.predict_proba(X_test_scaled)[:, 1]
    results['DeepLearning_MLP'] = evaluate_predictions("Deep Neural Network (MLP)", y_test, mlp_preds, mlp_probs)
    joblib.dump(mlp_model, os.path.join(models_dir, "mlp_neural_network_model.joblib"))
    
    # Feature Importances from Random Forest & XGBoost
    feature_names = joblib.load(os.path.join(models_dir, "features.joblib"))
    rf_importance = dict(zip(feature_names, rf_model.feature_importances_))
    xgb_importance = dict(zip(feature_names, xgb_model.feature_importances_))
    
    metrics_summary = {
        "results": results,
        "rf_importance": rf_importance,
        "xgb_importance": xgb_importance
    }
    joblib.dump(metrics_summary, os.path.join(models_dir, "metrics_summary.joblib"))
    print("\n All models and evaluation metrics successfully saved to:", models_dir)

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_file = os.path.join(base_dir, "data", "customer_churn_data.csv")
    models_path = os.path.join(base_dir, "models")
    train_all_models(data_file, models_path)
