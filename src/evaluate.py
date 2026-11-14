"""
Model Evaluation and Diagnostic Visualizations
Generates high-resolution diagnostic charts:
- ROC-AUC Comparison Curve
- Confusion Matrix Heatmaps
- Feature Importance Rankings
"""
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def generate_evaluation_plots(models_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(models_dir, "metrics_summary.joblib")
    if not os.path.exists(summary_path):
        print("Summary metrics file not found. Run train_models.py first.")
        return
        
    summary = joblib.load(summary_path)
    results = summary["results"]
    xgb_imp = summary["xgb_importance"]
    
    # 1. Model Comparison Bar Chart
    plt.figure(figsize=(10, 5), dpi=300)
    models = list(results.keys())
    accuracies = [results[m]["accuracy"] * 100 for m in models]
    aucs = [results[m]["auc"] * 100 for m in models]
    f1s = [results[m]["f1"] * 100 for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    plt.bar(x - width, accuracies, width, label='Accuracy (%)', color='#1f77b4')
    plt.bar(x, aucs, width, label='ROC-AUC (%)', color='#2ca02c')
    plt.bar(x + width, f1s, width, label='F1-Score (%)', color='#ff7f0e')
    
    plt.ylabel('Percentage Score (%)', fontsize=12, fontweight='bold')
    plt.title('Algorithm Benchmark Comparison: Accuracy vs ROC-AUC vs F1', fontsize=14, fontweight='bold')
    plt.xticks(x, [results[m]["name"] for m in models], fontsize=11, fontweight='bold')
    plt.ylim(50, 100)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_benchmark_comparison.png"))
    plt.close()
    
    # 2. Top 10 Feature Importances (XGBoost)
    plt.figure(figsize=(9, 6), dpi=300)
    sorted_imp = sorted(xgb_imp.items(), key=lambda item: item[1], reverse=True)[:10]
    feats, scores = zip(*sorted_imp)
    
    clean_labels = [
        f.replace('_', ' ').replace('PaymentMethod ', '').replace('InternetService ', '')
        for f in feats
    ]
    
    sns.barplot(x=list(scores), y=clean_labels, palette="viridis")
    plt.title('Top 10 Churn Predictor Features (XGBoost Gini Importance)', fontsize=13, fontweight='bold')
    plt.xlabel('Relative Feature Importance Score', fontsize=11, fontweight='bold')
    plt.ylabel('Customer Feature', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_feature_importance.png"))
    plt.close()
    
    print(f"Diagnostic charts saved to: {output_dir}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    m_dir = os.path.join(base_dir, "models")
    generate_evaluation_plots(m_dir, m_dir)
