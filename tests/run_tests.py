"""
Automated Test Runner (Python unittest)
Runs without external dependencies.
Validates data cleaning, feature engineering, model inference,
counterfactual simulation, and local explainability.
"""
import sys
import os
import unittest
import numpy as np
import pandas as pd
import joblib

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from config import ALL_FEATURE_COLUMNS, MODELS_DIR
from data_pipeline import clean_and_feature_engineer
from explainability import explain_prediction, simulate_what_if_counterfactual

class TestChurnPlatform(unittest.TestCase):

    def setUp(self):
        self.sample_customer = {
            'customerID': 'TEST-UNIT-01',
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'No',
            'tenure': 6,
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
            'MonthlyCharges': 95.0,
            'TotalCharges': 570.0
        }
        self.scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
        self.model = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.joblib"))
        self.features = joblib.load(os.path.join(MODELS_DIR, "features.joblib"))

    def test_01_feature_engineering_dimensions(self):
        """Verify that engineered feature space contains exactly 24 dimensions."""
        df = pd.DataFrame([self.sample_customer])
        processed = clean_and_feature_engineer(df)
        for col in ALL_FEATURE_COLUMNS:
            self.assertIn(col, processed.columns)
        self.assertEqual(len(ALL_FEATURE_COLUMNS), 24)
        print("\n [PASS] Feature engineering dimensions verified (24 features).")

    def test_02_total_charges_imputation(self):
        """Verify that missing/string TotalCharges are imputed via tenure * monthly charges."""
        raw = {'MonthlyCharges': 80.0, 'tenure': 10, 'TotalCharges': "   "}
        df = pd.DataFrame([raw])
        processed = clean_and_feature_engineer(df)
        self.assertFalse(np.isnan(processed['TotalCharges'].iloc[0]))
        self.assertEqual(processed['TotalCharges'].iloc[0], 800.0)
        print("\n [PASS] Total charges imputation verified.")

    def test_03_inference_probability_bounds(self):
        """Verify that inference model outputs valid probabilities within [0.0, 1.0]."""
        df = pd.DataFrame([self.sample_customer])
        proc = clean_and_feature_engineer(df)[ALL_FEATURE_COLUMNS]
        scaled = self.scaler.transform(proc)
        prob = self.model.predict_proba(scaled)[0, 1]
        self.assertTrue(0.0 <= prob <= 1.0)
        print(f"\n [PASS] Inference probability bounds verified (p={prob:.4f}).")

    def test_04_what_if_simulation_efficacy(self):
        """Verify that positive intervention (2-yr contract + support) reduces churn probability."""
        res = simulate_what_if_counterfactual(
            self.sample_customer,
            {'Contract': 'Two year', 'TechSupport': 'Yes'},
            self.model,
            self.scaler
        )
        self.assertLess(res['simulated_churn_prob'], res['base_churn_prob'])
        self.assertGreater(res['percentage_risk_reduction'], 0.0)
        self.assertGreater(res['annual_revenue_preserved'], 0.0)
        print(f"\n [PASS] Counterfactual simulation verified (delta: -{res['absolute_reduction']*100:.1f}%).")

    def test_05_explainability_attribution_structure(self):
        """Verify that local explainability generates drivers and risk tiers."""
        exp = explain_prediction(self.sample_customer, self.model, self.scaler, self.features)
        self.assertIn("churn_probability", exp)
        self.assertIn("top_churn_drivers", exp)
        self.assertIn("risk_tier", exp)
        self.assertGreater(len(exp["top_churn_drivers"]), 0)
        print(f"\n [PASS] Local explainability attribution verified ({len(exp['top_churn_drivers'])} drivers detected).")

if __name__ == '__main__':
    unittest.main(verbosity=2)
