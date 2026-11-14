"""
Automated Jupyter Research Notebook Generator
Creates 2 publication-grade, fully commented experimental notebooks:
1. notebooks/01_Statistical_Hypothesis_Testing_and_EDA.ipynb
2. notebooks/02_Model_Training_Benchmarking_and_XAI.ipynb
"""
import os
import json

def create_notebook(cells, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nb_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, indent=2)
    print(f"[Success] Generated research notebook: {output_path}")

def make_md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

def make_code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }

def build_notebook_1(output_path):
    cells = [
        make_md_cell("""# 01. Exploratory Data Analysis & Statistical Hypothesis Testing
**Project:** End-to-End Customer Churn Intelligence & Predictive Analytics Pipeline Using Python  
**Candidate:** Nithishkumar A S (Register Number: P25CA0044)  
**Degree:** Master of Computer Applications (MCA), BIHER

---
## Objective:
This notebook performs rigorous statistical analysis on the 7,043 subscriber benchmark dataset:
1. Data Ingestion & Missing Value Imputation
2. Pearson's Chi-Square ($\chi^2$) Test of Independence with Cramér's V
3. Two-Sample Welch's $t$-Test on continuous features (Tenure, Charges, CLV)
4. Correlation Heatmaps & Multicollinearity Assessment"""),

        make_code_cell("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

print("Libraries imported successfully.")"""),

        make_md_cell("""## 1. Data Ingestion & Schema Inspection"""),

        make_code_cell("""data_path = os.path.join("..", "data", "customer_churn_data.csv")
df = pd.read_csv(data_path)
print(f"Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
df.head(5)"""),

        make_md_cell("""## 2. Target Variable Distribution (Class Imbalance)"""),

        make_code_cell("""churn_counts = df['Churn'].value_counts()
churn_pct = df['Churn'].value_counts(normalize=True) * 100

print(f"Retained Accounts: {churn_counts.get('No', 0)} ({churn_pct.get('No', 0):.2f}%)")
print(f"Churned Accounts : {churn_counts.get('Yes', 0)} ({churn_pct.get('Yes', 0):.2f}%)")

plt.figure(figsize=(6, 4))
sns.barplot(x=churn_counts.index, y=churn_counts.values, palette=["#059669", "#DC2626"])
plt.title("Target Distribution: Churned vs. Retained Subscribers", fontsize=12, fontweight='bold')
plt.xlabel("Churn Status")
plt.ylabel("Number of Subscribers")
plt.show()"""),

        make_md_cell("""## 3. Pearson's Chi-Square Test of Independence ($\chi^2$)
Testing $H_0$: *Customer subscription choices are independent of churn behavior.*"""),

        make_code_cell("""from src.statistical_tests import run_statistical_suite
stat_results = run_statistical_suite(data_path)
chi2_df = pd.DataFrame(stat_results['chi2_results'])
chi2_df"""),

        make_md_cell("""## 4. Two-Sample Welch's $t$-Test & Effect Sizes (Continuous Features)"""),

        make_code_cell("""ttest_df = pd.DataFrame(stat_results['ttest_results'])
ttest_df"""),

        make_md_cell("""## 5. Tenure Cohort vs. Monthly Charges Analysis"""),

        make_code_cell("""plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x='Contract', y='MonthlyCharges', hue='Churn', palette=["#059669", "#DC2626"])
plt.title("Monthly Charges Distribution Across Contract Types by Churn Status", fontsize=12, fontweight='bold')
plt.xlabel("Contract Duration")
plt.ylabel("Monthly Charges ($)")
plt.show()""")
    ]
    create_notebook(cells, output_path)

def build_notebook_2(output_path):
    cells = [
        make_md_cell("""# 02. Multi-Model Benchmark, Deep Learning & Explainable AI (XAI)
**Project:** End-to-End Customer Churn Intelligence & Predictive Analytics Pipeline Using Python  
**Candidate:** Nithishkumar A S (Register Number: P25CA0044)  
**Degree:** Master of Computer Applications (MCA), BIHER

---
## Objective:
1. Feature Engineering & Stratified Train/Test Split
2. Multi-Model Benchmarking (Logistic Regression, Random Forest, XGBoost, Deep Neural Network)
3. Model Diagnostics (ROC-AUC, Confusion Matrices, Learning Curves)
4. Local Feature Attribution (XAI) and Counterfactual What-If Simulation"""),

        make_code_cell("""import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

print("Inference libraries loaded.")"""),

        make_md_cell("""## 1. Load Pretrained Models & Feature Artifacts"""),

        make_code_cell("""models_dir = os.path.join("..", "models")
scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
features = joblib.load(os.path.join(models_dir, "features.joblib"))
summary = joblib.load(os.path.join(models_dir, "metrics_summary.joblib"))

results_df = pd.DataFrame([
    {
        "Model": v["name"],
        "Accuracy (%)": round(v["accuracy"] * 100, 2),
        "ROC-AUC": round(v["auc"], 4),
        "Recall (%)": round(v["recall"] * 100, 2),
        "Precision (%)": round(v["precision"] * 100, 2),
        "F1-Score (%)": round(v["f1"] * 100, 2),
        "Brier Score": round(v["brier_score"], 4)
    }
    for k, v in summary["results"].items()
])
results_df"""),

        make_md_cell("""## 2. Global Feature Importance Ranking (XGBoost Gini Importance)"""),

        make_code_cell("""xgb_imp = summary["xgb_importance"]
sorted_imp = sorted(xgb_imp.items(), key=lambda x: x[1], reverse=True)[:10]

plt.figure(figsize=(9, 5))
sns.barplot(x=[s[1] for s in sorted_imp], y=[s[0].replace('_', ' ') for s in sorted_imp], palette="crest")
plt.title("Top 10 Churn Predictors (XGBoost Gini Impurity Gain)", fontsize=12, fontweight='bold')
plt.xlabel("Normalized Importance Score")
plt.ylabel("Engineered Feature")
plt.show()"""),

        make_md_cell("""## 3. Deep Neural Network (MLP) Training Loss Convergence Curve"""),

        make_code_cell("""loss_data = joblib.load(os.path.join(models_dir, "mlp_loss_history.joblib"))
losses = loss_data["loss_curve"]

plt.figure(figsize=(8.5, 4))
plt.plot(range(1, len(losses) + 1), losses, color='#7C3AED', linewidth=2.0)
plt.title("Deep Neural Network (MLP: 64 -> 32) Training Loss vs. Epochs", fontsize=12, fontweight='bold')
plt.xlabel("Epochs")
plt.ylabel("Cross-Entropy Loss (Adam Optimizer)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()"""),

        make_md_cell("""## 4. Local Explainability (XAI) & Counterfactual Simulation Demo"""),

        make_code_cell("""from src.explainability import explain_prediction, simulate_what_if_counterfactual
xgb_model = joblib.load(os.path.join(models_dir, "xgboost_model.joblib"))

sample_cust = {
    'customerID': 'CUST-NOTEBOOK-01',
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 4,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 89.0,
    'TotalCharges': 356.0
}

# 1. Local Explainability
exp = explain_prediction(sample_cust, xgb_model, scaler, features)
print(f"Base Predicted Churn Probability: {exp['churn_probability'] * 100:.1f}% ({exp['risk_tier']})")
print("\\nTop Churn Drivers:")
for d in exp['top_churn_drivers']:
    print(f"  [+] {d['feature']}: impact = +{d['impact']:.3f}")

# 2. Counterfactual Simulation
sim = simulate_what_if_counterfactual(
    sample_cust,
    {'Contract': 'Two year', 'TechSupport': 'Yes'},
    xgb_model,
    scaler
)
print(f"\\nCounterfactual Intervention Applied: Two-Year Contract + Dedicated Tech Support")
print(f"Simulated New Churn Probability : {sim['simulated_churn_prob'] * 100:.1f}%")
print(f"Risk Reduction Delta (ΔP)       : -{sim['absolute_reduction'] * 100:.1f}% ({sim['percentage_risk_reduction']:.1f}% relative reduction)")
print(f"Annual Recurring Revenue Saved   : ${sim['annual_revenue_preserved']:,.2f}")""")
    ]
    create_notebook(cells, output_path)

if __name__ == "__main__":
    base_dir = r"c:\Users\Nithish\Downloads\final year project\End-to-End-Customer-Churn-Intelligence-Predictive-Analytics-Pipeline-Using-Python"
    nb1 = os.path.join(base_dir, "notebooks", "01_Statistical_Hypothesis_Testing_and_EDA.ipynb")
    nb2 = os.path.join(base_dir, "notebooks", "02_Model_Training_Benchmarking_and_XAI.ipynb")
    build_notebook_1(nb1)
    build_notebook_2(nb2)
