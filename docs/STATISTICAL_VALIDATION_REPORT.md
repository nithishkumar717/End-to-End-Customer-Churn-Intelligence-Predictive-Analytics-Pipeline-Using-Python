# Statistical Hypothesis Testing & Validation Report

**Candidate:** Nithishkumar A S (Register Number: P25CA0044)
**Degree:** Master of Computer Applications (MCA), BIHER

---

## 1. Pearson's Chi-Square Test of Independence (Categorical Features vs. Churn)

Evaluates the null hypothesis ($H_0$): *Customer subscription choices are independent of churn behavior.*

| Feature | Chi-Square (χ²) | df | p-value | Cramér's V | Statistical Inference |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Contract** | 1528.18 | 2 | 0.00e+00 | 0.4658 | p < 0.001 (Highly Significant) |
| **InternetService** | 269.84 | 2 | 2.54e-59 | 0.1957 | p < 0.001 (Highly Significant) |
| **TechSupport** | 198.75 | 2 | 6.97e-44 | 0.1680 | p < 0.001 (Highly Significant) |
| **OnlineSecurity** | 194.81 | 2 | 4.99e-43 | 0.1663 | p < 0.001 (Highly Significant) |
| **PaymentMethod** | 56.77 | 3 | 2.87e-12 | 0.0898 | p < 0.001 (Highly Significant) |
| **MultipleLines** | 5.56 | 2 | 6.20e-02 | 0.0281 | Not Significant |
| **PaperlessBilling** | 3.88 | 1 | 4.89e-02 | 0.0235 | p = 0.0489 |
| **SeniorCitizen** | 0.52 | 1 | 4.71e-01 | 0.0086 | Not Significant |
| **Partner** | 0.01 | 1 | 9.18e-01 | 0.0012 | Not Significant |
| **Dependents** | 0.00 | 1 | 9.44e-01 | 0.0008 | Not Significant |


---

## 2. Two-Sample Welch's t-Test & Effect Sizes (Continuous Features)

Evaluates the null hypothesis ($H_0$): *Mean feature values are identical between Retained and Churned cohorts.*

| Feature | Mean (Retained) | Mean (Churned) | t-Statistic | p-value | Cohen's d | Statistical Inference |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **MonthlyCharges** | 65.45 | 74.09 | 14.22 | 3.33e-45 | 0.344 | Reject H0 (Significant difference) |
| **ChargeRatio** | 1.0 | 1.0 | 1.83 | 6.70e-02 | 0.043 | Fail to reject H0 |
| **EstimatedCLV** | 2750.14 | 2282.64 | -13.38 | 3.12e-40 | -0.327 | Reject H0 (Significant difference) |
| **TotalCharges** | 2096.01 | 1265.58 | -21.87 | 9.09e-102 | -0.534 | Reject H0 (Significant difference) |
| **tenure** | 31.21 | 16.84 | -31.75 | 1.35e-203 | -0.775 | Reject H0 (Significant difference) |
