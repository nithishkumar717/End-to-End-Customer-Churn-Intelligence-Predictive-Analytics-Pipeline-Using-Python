"""
Comprehensive Evaluation & Diagnostic Visualizer
Generates 4 publication-quality academic evaluation charts:
1. Algorithm Performance Benchmark (Accuracy, ROC-AUC, F1, Precision)
2. XGBoost Feature Importance Attribution
3. Deep Neural Network Epoch Loss Convergence Curve
4. Confusion Matrices Comparison Grid
"""
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

try:
    from config import MODELS_DIR
except ImportError:
    from src.config import MODELS_DIR

def generate_all_diagnostic_plots(models_dir: str = MODELS_DIR):
    os.makedirs(models_dir, exist_ok=True)
    summary_path = os.path.join(models_dir, "metrics_summary.joblib")
    loss_path = os.path.join(models_dir, "mlp_loss_history.joblib")

    if not os.path.exists(summary_path):
        print(f"[Error] Metrics summary not found at {summary_path}. Run train_models.py first.")
        return

    summary = joblib.load(summary_path)
    results = summary["results"]
    xgb_imp = summary["xgb_importance"]

    # 1. Benchmark Bar Chart (4 Models)
    plt.figure(figsize=(11, 5.5), dpi=300)
    models = list(results.keys())
    labels = [results[m]["name"] for m in models]
    accuracies = [results[m]["accuracy"] * 100 for m in models]
    aucs = [results[m]["auc"] * 100 for m in models]
    f1s = [results[m]["f1"] * 100 for m in models]
    recalls = [results[m]["recall"] * 100 for m in models]

    x = np.arange(len(models))
    width = 0.20

    plt.bar(x - 1.5 * width, accuracies, width, label='Accuracy (%)', color='#2563EB')
    plt.bar(x - 0.5 * width, aucs, width, label='ROC-AUC (%)', color='#059669')
    plt.bar(x + 0.5 * width, f1s, width, label='F1-Score (%)', color='#D97706')
    plt.bar(x + 1.5 * width, recalls, width, label='Recall (%)', color='#DC2626')

    plt.ylabel('Score Percentage (%)', fontsize=12, fontweight='bold')
    plt.title('Multi-Algorithm Benchmark Comparison (MCA Capstone Evaluation)', fontsize=13, fontweight='bold')
    plt.xticks(x, labels, fontsize=10, fontweight='bold')
    plt.ylim(50, 100)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    chart1_path = os.path.join(models_dir, "model_benchmark_comparison.png")
    plt.savefig(chart1_path)
    plt.close()

    # 2. Top-10 Feature Importances
    plt.figure(figsize=(9.5, 6), dpi=300)
    sorted_imp = sorted(xgb_imp.items(), key=lambda item: item[1], reverse=True)[:10]
    feats, scores = zip(*sorted_imp)
    clean_labels = [
        f.replace('_', ' ').replace('PaymentMethod ', '').replace('InternetService ', '')
        for f in feats
    ]

    sns.barplot(x=list(scores), y=clean_labels, palette="crest")
    plt.title('Top 10 Churn Predictors (XGBoost Gini Importance)', fontsize=13, fontweight='bold')
    plt.xlabel('Normalized Relative Importance', fontsize=11, fontweight='bold')
    plt.ylabel('Engineered Feature', fontsize=11, fontweight='bold')
    plt.tight_layout()
    chart2_path = os.path.join(models_dir, "top_feature_importance.png")
    plt.savefig(chart2_path)
    plt.close()

    # 3. Deep Neural Network Epoch Loss Convergence Curve
    if os.path.exists(loss_path):
        loss_data = joblib.load(loss_path)
        losses = loss_data["loss_curve"]

        plt.figure(figsize=(9, 4.5), dpi=300)
        plt.plot(range(1, len(losses) + 1), losses, color='#7C3AED', linewidth=2.2, label='Cross-Entropy Loss (Adam)')
        plt.title('Deep Neural Network (MLP) Training Loss Convergence', fontsize=13, fontweight='bold')
        plt.xlabel('Epochs (Iterations)', fontsize=11, fontweight='bold')
        plt.ylabel('Loss Value', fontsize=11, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(frameon=True)
        plt.tight_layout()
        chart3_path = os.path.join(models_dir, "neural_network_learning_curve.png")
        plt.savefig(chart3_path)
        plt.close()

    # 4. Confusion Matrices Grid (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    for idx, (m_key, ax) in enumerate(zip(models, axes.flatten())):
        cm = np.array(results[m_key]["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax, annot_kws={"size": 13, "weight": "bold"})
        ax.set_title(f"{results[m_key]['name']}\n(Acc: {results[m_key]['accuracy']*100:.1f}%)", fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted Label (0=Retained, 1=Churned)', fontsize=9)
        ax.set_ylabel('True Label', fontsize=9)

    plt.suptitle('Confusion Matrix Diagnostics Across All 4 Models', fontsize=13, fontweight='bold')
    plt.tight_layout()
    chart4_path = os.path.join(models_dir, "confusion_matrices_grid.png")
    plt.savefig(chart4_path)
    plt.close()

    print(f"[Success] All 4 diagnostic charts generated and saved in: {models_dir}")

if __name__ == "__main__":
    generate_all_diagnostic_plots()
