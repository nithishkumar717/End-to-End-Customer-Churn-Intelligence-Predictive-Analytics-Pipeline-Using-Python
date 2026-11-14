# Final Year Project Review Presentation (PPT) Blueprint

**Degree:** Master of Computer Applications (MCA)  
**Project:** End-to-End Customer Churn Intelligence & Predictive Analytics Pipeline Using Python  
**Candidate:** Nithishkumar A S (Register Number: P25CA0044)  
**Institution:** Bharath Institute of Higher Education and Research (BIHER)

---

## 🖥️ Slide-by-Slide Presentation Structure (15 Slides)

### Slide 1: Title Slide
* **Title:** End-to-End Customer Churn Intelligence & Predictive Analytics Pipeline Using Python
* **Candidate:** Nithishkumar A S (Reg. No: `P25CA0044`)
* **Degree:** Master of Computer Applications (MCA)
* **Department:** Faculty of Arts and Science, Department of Computer Science & Applications
* **Institution:** Bharath Institute of Higher Education and Research (BIHER), Chennai

---

### Slide 2: Introduction & Domain Motivation
* **Key Points:**
  * Telecom & SaaS industries face severe customer attrition rates (20% to 35% annually).
  * Acquiring a new customer costs **5x to 7x more** than retaining an existing customer.
  * A 5% boost in customer retention yields up to a 25%–95% increase in operational profitability.
* **Presenter Speaking Note:** *"Good morning respected internal guides and external examiners. My project addresses one of the most critical revenue-draining challenges in subscription enterprises: customer churn."*

---

### Slide 3: Problem Statement & Research Gap
* **Current Challenges in Enterprise Retention:**
  * **Silent Attrition:** Over 65% of churning customers cancel without submitting complaints.
  * **Black-Box Skepticism:** Traditional ML outputs raw probabilities without explainable justification.
  * **Lack of Prescriptive Action:** Existing platforms predict who will churn, but cannot simulate the financial impact of counter-offers.

---

### Slide 4: IEEE Base Paper Alignment
* **Primary Reference:** *A. Amin et al., "Just-in-Time Customer Churn Prediction With Explainable Machine Learning Over Telecommunications Big Data," IEEE Access, Vol. 8, 2020.*
* **Our Architectural Advancement:**
  * Incorporates a **dual-engine architecture**: Interactive Web Application (Streamlit) + REST Microservice (FastAPI).
  * Integrates **Counterfactual 'What-If' Simulation** to quantify preserved Customer Lifetime Value ($).

---

### Slide 5: Existing System vs. Proposed System
* **Table Comparison:**
  * *Existing Systems:* Single algorithm, raw probabilities, manual spreadsheet audits, black-box decisions.
  * *Proposed System:* 4-model benchmark (XGBoost, Random Forest, Deep Learning MLP, Logistic Regression), Explainable AI (XAI) feature attribution, interactive counterfactual simulator, automated unit testing.

---

### Slide 6: Proposed System Architecture
* **Visual:** [Include Mermaid Architecture Diagram from docs/SYSTEM_DESIGN_AND_UML.md]
* **4-Stage Pipeline:**
  1. Data Ingestion & Imputation Layer (7,043 Profiles).
  2. Stratified Feature Engineering (24 Dimensions).
  3. Multi-Model Inference Engine.
  4. Decision Support & XAI Presentation Layer.

---

### Slide 7: Automated Feature Engineering
* **Key Engineered Features:**
  * **ChargeRatio:** $\frac{\text{TotalCharges}}{\text{tenure} \times \text{MonthlyCharges}}$ (detects unannounced rate spikes).
  * **Estimated CLV (Customer Lifetime Value):** $\text{MonthlyCharges} \times (\text{tenure} + 24) \times 0.75$.
  * **Service Combination Vectors:** Encoding of Fiber Optic vs. DSL, Contract Duration, and Payment Friction.

---

### Slide 8: Multi-Model Machine Learning Core
* **Algorithms Evaluated:**
  1. **Logistic Regression:** Linear baseline ($L_2$ Regularized).
  2. **Random Forest:** 180 trees with cost-sensitive class balancing.
  3. **XGBoost (Champion):** Gradient boosted decision trees ($0.8181$ ROC-AUC).
  4. **Deep Neural Network (MLP):** 3-layer architecture ($64 \rightarrow 32$ neurons, ReLU, Adam optimizer).

---

### Slide 9: Algorithm Benchmark Comparison & Results
* **Performance Table:**
  * XGBoost: **ROC-AUC: 0.8181 | Recall: 81.77% | F1-Score: 77.60% | Brier Loss: 0.1735**
  * Logistic Regression: ROC-AUC: 0.8168 | Accuracy: 74.80%
  * Random Forest: ROC-AUC: 0.8124 | Accuracy: 74.31%
  * Deep Neural Network: ROC-AUC: 0.7417 | F1-Score: 71.96%
* **Visual:** Insert `models/model_benchmark_comparison.png` and `models/confusion_matrices_grid.png`.

---

### Slide 10: Explainable AI (XAI) Attribution Breakdown
* **How We Demystify the Black Box:**
  * Calculates local feature impact: $\text{Attribution}_j = z_j \times W_j$.
  * Categorizes factors into **Churn Risk Drivers** (e.g. Month-to-month contracts, high monthly fees) and **Loyalty Anchors** (e.g. long tenure, auto-pay).
* **Presenter Speaking Note:** *"When an account manager views a customer with an 88% churn risk, the system clearly highlights that the primary driver is the month-to-month contract (+0.24) and lack of tech support (+0.18)."*

---

### Slide 11: Counterfactual "What-If" Policy Simulator
* **Interactive Decision Intelligence:**
  * Allows managers to test interventions: *"What happens if we offer a 1-year contract and free tech support?"*
  * **Output:** Risk probability drops from **92.3%** to **38.9%** (Risk Reduction: **-57.8%**), preserving **$656.70/year** in recurring revenue.

---

### Slide 12: Enterprise Web Application & REST Microservice
* **Frontend:** Streamlit Web Portal (`app.py` on `localhost:8501`) featuring 5 distinct operational views.
* **Backend:** FastAPI Microservice (`src/api.py` on `localhost:8000/docs`) with automatic OpenAPI Swagger documentation.
* **High-Throughput Batch Auditing:** Drag-and-drop CSV processing for thousands of subscriber records.

---

### Slide 13: Software Engineering & Automated Testing
* **Test-Driven Rigor (`tests/run_tests.py`):**
  * Test 1: Feature engineering dimension invariance (24 columns).
  * Test 2: Missing total charges imputation.
  * Test 3: Mathematical probability bounds $[0.0, 1.0]$.
  * Test 4: Counterfactual risk reduction monotonicity.
  * Test 5: Explainability schema integrity.
* **Result:** `Ran 5 tests in 0.259s. OK (100% Passed)`.

---

### Slide 14: Conclusion & Business Impact
* **Summary of Achievements:**
  * Delivered a complete, end-to-end data intelligence platform.
  * Solved the black-box problem using Explainable AI.
  * Connected predictive probability scores to prescriptive financial ROI.
* **Future Scope:**
  * Integrating real-time streaming data ingestion with Apache Kafka.
  * Deploying automated multi-armed bandit testing for retention offers.

---

### Slide 15: Acknowledgement & Viva Q&A
* **Thank You to:**
  * Internal Guide & Project Coordinator.
  * Faculty of Arts and Science, BIHER.
  * External University Examiners.
* **Open for Questions!**
