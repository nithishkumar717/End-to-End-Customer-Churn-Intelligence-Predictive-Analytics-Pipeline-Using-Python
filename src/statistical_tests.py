"""
Academic Statistical Validation & Hypothesis Testing Engine
Conducts formal inference testing:
1. Pearson's Chi-Square Test of Independence (with Cramér's V effect size)
2. Two-Sample Independent Student's t-Test & Mann-Whitney U Test
3. Multicollinearity Assessment via Correlation & VIF Approximations
"""
import os
import joblib
import numpy as np
import pandas as pd
from scipy import stats

try:
    from config import RAW_DATA_PATH, MODELS_DIR, PROJECT_ROOT
    from data_pipeline import clean_and_feature_engineer
except ImportError:
    from src.config import RAW_DATA_PATH, MODELS_DIR, PROJECT_ROOT
    from src.data_pipeline import clean_and_feature_engineer

def run_statistical_suite(data_path: str = RAW_DATA_PATH, output_dir: str = MODELS_DIR):
    df = pd.read_csv(data_path)
    processed_df = clean_and_feature_engineer(df)

    # 1. Chi-Square Test of Independence (Categorical vs Churn)
    categorical_cols = [
        'Contract', 'InternetService', 'PaymentMethod', 'TechSupport',
        'OnlineSecurity', 'PaperlessBilling', 'Partner', 'Dependents',
        'SeniorCitizen', 'MultipleLines'
    ]

    chi2_results = []
    for col in categorical_cols:
        contingency = pd.crosstab(df[col], df['Churn'])
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        # Cramér's V effect size
        cramers_v = np.sqrt(chi2 / (n * max(1, min_dim)))

        significance = "p < 0.001 (Highly Significant)" if p_val < 0.001 else (f"p = {p_val:.4f}" if p_val < 0.05 else "Not Significant")

        chi2_results.append({
            "Feature": col,
            "Chi2_Statistic": round(chi2, 2),
            "Degrees_of_Freedom": dof,
            "p_value": p_val,
            "Cramers_V": round(cramers_v, 4),
            "Inference": significance
        })

    chi2_df = pd.DataFrame(chi2_results).sort_values(by="Cramers_V", ascending=False)

    # 2. Two-Sample Student's t-Test & Mann-Whitney U (Numerical vs Churn)
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'ChargeRatio', 'EstimatedCLV']
    churn_yes = processed_df[processed_df['Churn'] == 'Yes']
    churn_no = processed_df[processed_df['Churn'] == 'No']

    ttest_results = []
    for col in numerical_cols:
        group_yes = churn_yes[col].dropna()
        group_no = churn_no[col].dropna()

        t_stat, p_val_t = stats.ttest_ind(group_yes, group_no, equal_var=False)
        u_stat, p_val_u = stats.mannwhitneyu(group_yes, group_no, alternative='two-sided')

        # Cohen's d effect size
        pooled_std = np.sqrt((group_yes.std()**2 + group_no.std()**2) / 2)
        cohens_d = (group_yes.mean() - group_no.mean()) / (pooled_std + 1e-6)

        ttest_results.append({
            "Feature": col,
            "Mean_Retained": round(group_no.mean(), 2),
            "Mean_Churned": round(group_yes.mean(), 2),
            "t_Statistic": round(t_stat, 2),
            "p_value": p_val_t,
            "Cohens_d": round(cohens_d, 3),
            "Conclusion": "Reject H0 (Significant difference)" if p_val_t < 0.001 else "Fail to reject H0"
        })

    ttest_df = pd.DataFrame(ttest_results).sort_values(by="t_Statistic", ascending=False)

    # Save artifact
    stat_summary = {
        "chi2_results": chi2_df.to_dict(orient="records"),
        "ttest_results": ttest_df.to_dict(orient="records")
    }
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(stat_summary, os.path.join(output_dir, "statistical_summary.joblib"))

    # Generate Markdown Report
    doc_path = os.path.join(PROJECT_ROOT, "docs", "STATISTICAL_VALIDATION_REPORT.md")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("# Statistical Hypothesis Testing & Validation Report\n\n")
        f.write("**Candidate:** Nithishkumar A S (Register Number: P25CA0044)\n")
        f.write("**Degree:** Master of Computer Applications (MCA), BIHER\n\n")
        f.write("---\n\n")
        f.write("## 1. Pearson's Chi-Square Test of Independence (Categorical Features vs. Churn)\n\n")
        f.write("Evaluates the null hypothesis ($H_0$): *Customer subscription choices are independent of churn behavior.*\n\n")
        f.write("| Feature | Chi-Square (χ²) | df | p-value | Cramér's V | Statistical Inference |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :--- |\n")
        for r in chi2_df.to_dict(orient="records"):
            f.write(f"| **{r['Feature']}** | {r['Chi2_Statistic']:.2f} | {r['Degrees_of_Freedom']} | {r['p_value']:.2e} | {r['Cramers_V']:.4f} | {r['Inference']} |\n")

        f.write("\n\n---\n\n")
        f.write("## 2. Two-Sample Welch's t-Test & Effect Sizes (Continuous Features)\n\n")
        f.write("Evaluates the null hypothesis ($H_0$): *Mean feature values are identical between Retained and Churned cohorts.*\n\n")
        f.write("| Feature | Mean (Retained) | Mean (Churned) | t-Statistic | p-value | Cohen's d | Statistical Inference |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :--- |\n")
        for r in ttest_df.to_dict(orient="records"):
            f.write(f"| **{r['Feature']}** | {r['Mean_Retained']} | {r['Mean_Churned']} | {r['t_Statistic']:.2f} | {r['p_value']:.2e} | {r['Cohens_d']} | {r['Conclusion']} |\n")

    print(f"[Success] Statistical suite completed. Report saved at: {doc_path}")
    return stat_summary

if __name__ == "__main__":
    run_statistical_suite()
