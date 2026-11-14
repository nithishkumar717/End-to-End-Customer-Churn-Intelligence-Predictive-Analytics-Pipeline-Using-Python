# End-to-End Customer Churn Intelligence & Predictive Analytics Pipeline Using Python

A production-grade, enterprise data science and machine learning platform for customer attrition forecasting, feature explainability, and prescriptive retention analytics.

---

## 📌 Architecture & Features

1. **Automated Feature Engineering Pipeline**:
   - Handles missing total charges, calculates charge-to-tenure ratios, encodes contract types, and standardizes multi-channel service attributes.
2. **Multi-Model Machine Learning Core**:
   - **XGBoost Classifier**: Gradient boosted decision trees for state-of-the-art tabular accuracy.
   - **Random Forest Classifier**: Ensemble bagging for robust risk variance reduction.
   - **Deep Neural Network (MLP)**: Multi-layer perceptron architecture with adaptive Adam optimizer and early stopping.
3. **Interactive Streamlit Web Dashboard**:
   - **Executive KPI Monitoring**: High-level churn rates, retention revenue impact, and ROC-AUC benchmarks.
   - **Single Customer Scoring**: Real-time sliders/dropdowns with instant probability gauge and risk badges (Critical, Moderate, Low).
   - **Prescriptive Retention Recommendations**: Actionable discount and contract extension offers tailored to the user's specific churn drivers.
   - **Batch Auditing**: Drag-and-drop CSV upload to score thousands of accounts with downloadable risk reports.
   - **Diagnostic Visualizer**: ROC-AUC curves, confusion matrices, and top 10 feature importance plots.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+ (Verified on Python 3.12)

### 2. Environment Setup
```bash
cd Customer_Churn_Platform
# Activate the virtual environment
.venv\Scripts\activate
```

### 3. Model Training & Evaluation
```bash
python src/train_models.py
python src/evaluate.py
```

### 4. Launch the Web Application
```bash
streamlit run app.py
```
Or double-click `run_app.bat` on Windows!

The application will open automatically in your browser at `http://localhost:8501`.
