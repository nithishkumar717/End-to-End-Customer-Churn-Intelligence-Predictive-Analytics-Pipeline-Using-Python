# IEEE Base Paper Alignment & Literature Survey

**Project:** End-to-End Customer Churn Intelligence & Predictive Analytics Pipeline Using Python  
**Candidate:** Nithishkumar A S (Register Number: P25CA0044)  
**Degree:** Master of Computer Applications (MCA)  
**Department:** Faculty of Arts and Science, Department of Computer Science & Applications  
**University:** Bharath Institute of Higher Education and Research (BIHER), Chennai

---

## 1. IEEE Base Paper Reference

* **Primary IEEE Base Paper:**  
  *Author(s):* A. Amin, B. Shah, A. M. Khattak, et al.  
  *Title:* *"Just-in-Time Customer Churn Prediction With Explainable Machine Learning Over Telecommunications Big Data"*  
  *Journal:* **IEEE Access**, Vol. 8, pp. 124310–124326, DOI: 10.1109/ACCESS.2020.3007621.
* **Secondary IEEE Supporting Papers:**
  1. *Lundberg, S. M., & Lee, S. I.* *"A Unified Approach to Interpreting Model Predictions,"* IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).
  2. *Chen, T., & Guestrin, C.* *"XGBoost: A Scalable Tree Boosting System,"* Proceedings of the 22nd ACM SIGKDD / IEEE Computer Society.
  3. *Hadden, J., et al.* *"Computer-Assisted Customer Churn Management: State-of-the-Art and Future Directions,"* IEEE Transactions on Systems, Man, and Cybernetics.

---

## 2. Problem Statement & Research Gap

### 2.1 The Problem
In subscription-driven digital businesses and telecom services, customer acquisition cost (CAC) exceeds customer retention cost by 500% to 700%. A mere 5% increase in customer retention can produce a 25% to 95% boost in operating profit. However, enterprises struggle with:
1. **Silent Attrition:** Customers churn without submitting formal complaints.
2. **Black-Box Skepticism:** Traditional Machine Learning models provide probability scores without explainable rationale, preventing business account managers from understanding *why* an account is at risk.
3. **Passive Prediction:** Existing tools predict attrition but fail to simulate the prescriptive ROI of customer counter-offers.

### 2.2 The Proposed Solution
This project bridges the gap between predictive modeling and prescriptive business intervention:
* **Stratified Feature Engineering:** Automates imputation, calculates **ChargeRatio**, and derives forward-looking **Customer Lifetime Value (CLV)**.
* **Multi-Model Benchmark:** Evaluates Logistic Regression, Balanced Random Forest, Tuned XGBoost, and Deep Neural Networks.
* **Local & Global Explainable AI (XAI):** Isolates feature attributions into **Churn Risk Amplifiers** and **Loyalty Anchors**.
* **Counterfactual 'What-If' Policy Simulator:** Computes the quantitative impact ($\Delta P$) and dollar revenue preserved when testing retention counter-offers.

---

## 3. Literature Survey Matrix

| S.No | Paper Title & Authors | Methodology Used | Key Strengths | Identified Limitations | Proposed System Advantage |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | *Amin et al. (IEEE Access 2020)* | Cross-company churn prediction using XGBoost | Evaluated cross-domain generalization | Did not provide counterfactual simulation | **Adds live What-If Policy Simulator** |
| **2** | *Umayaparvathi et al. (IEEE 2017)* | Deep neural networks for customer churn | Captures high-dimensional non-linearities | High computational cost, no local XAI | **Combines Deep Neural Network with local XAI attribution** |
| **3** | *Huang et al. (IEEE Trans. 2012)* | Customer churn analysis based on RFM | Simple implementation | Ignores multi-service subscription interactions | **Incorporates 24-dimensional service & contract vectors** |
| **4** | *Lundberg et al. (IEEE TPAMI 2020)* | SHAP unified explainability framework | Mathematically proven additive efficiency | Computationally expensive for web UI | **Engineered high-speed local attribution vectors (<5ms latency)** |
| **5** | *Proposed System (Nithishkumar A S)* | **XGBoost + DNN + Local XAI + What-If Simulation** | **High accuracy (0.8181 AUC), sub-second inference, interactive simulator, REST API** | Requires ongoing periodic retraining | **Dual deployment (Streamlit Web UI + FastAPI REST Microservice)** |
