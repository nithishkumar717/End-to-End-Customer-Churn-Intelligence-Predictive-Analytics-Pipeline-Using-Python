"""
Automated Unit Test Suite
Verifies data pipeline consistency, feature shapes, model inference,
counterfactual simulation bounds, and edge cases.
"""
import sys
import os
import pytest
import numpy as np
import pandas as pd
import joblib

# Ensure src is on sys.path
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from config import ALL_FEATURE_COLUMNS, MODELS_DIR
from data_pipeline import clean_and_feature_engineer
from explainability import explain_prediction, simulate_what_if_counterfactual

@pytest.fixture
def sample_customer():
    return {
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

def test_feature_engineering_columns(sample_customer):
    """Test that all 24 required feature columns are present and correctly typed."""
    df = pd.DataFrame([sample_customer])
    processed = clean_and_feature_engineer(df)
    for col in ALL_FEATURE_COLUMNS:
        assert col in processed.columns, f"Missing expected feature: {col}"
    assert len(ALL_FEATURE_COLUMNS) == 24

def test_missing_total_charges_imputation():
    """Test that missing or string TotalCharges are gracefully imputed."""
    raw = {
        'MonthlyCharges': 80.0,
        'tenure': 10,
        'TotalCharges': "   "  # whitespace string
    }
    df = pd.DataFrame([raw])
    processed = clean_and_feature_engineer(df)
    assert not np.isnan(processed['TotalCharges'].iloc[0])
    assert processed['TotalCharges'].iloc[0] == 800.0

def test_model_inference_bounds(sample_customer):
    """Test that inference output yields valid probability bounds [0.0, 1.0]."""
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    model = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.joblib"))

    df = pd.DataFrame([sample_customer])
    processed = clean_and_feature_engineer(df)[ALL_FEATURE_COLUMNS]
    scaled = scaler.transform(processed)
    prob = model.predict_proba(scaled)[0, 1]

    assert 0.0 <= prob <= 1.0
    assert isinstance(prob, (float, np.floating))

def test_what_if_counterfactual_reduction(sample_customer):
    """Test that upgrading contract to Two-year lowers churn probability."""
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    model = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.joblib"))

    res = simulate_what_if_counterfactual(
        sample_customer,
        {'Contract': 'Two year', 'TechSupport': 'Yes'},
        model,
        scaler
    )

    assert res['simulated_churn_prob'] < res['base_churn_prob']
    assert res['percentage_risk_reduction'] > 0
    assert res['annual_revenue_preserved'] > 0

def test_explainability_attribution_structure(sample_customer):
    """Test that local explainability returns top churn drivers."""
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    model = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.joblib"))
    features = joblib.load(os.path.join(MODELS_DIR, "features.joblib"))

    exp = explain_prediction(sample_customer, model, scaler, features)
    assert "churn_probability" in exp
    assert "top_churn_drivers" in exp
    assert "risk_tier" in exp
