"""
Production FastAPI Microservice
Exposes REST endpoints for real-time inference, Explainable AI (XAI),
and counterfactual simulation. Includes interactive Swagger OpenAPI docs.
"""
import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from config import MODELS_DIR, ALL_FEATURE_COLUMNS, RISK_THRESHOLD_HIGH, RISK_THRESHOLD_LOW
    from data_pipeline import clean_and_feature_engineer
    from explainability import explain_prediction, simulate_what_if_counterfactual, generate_prescriptive_strategy
except ImportError:
    from src.config import MODELS_DIR, ALL_FEATURE_COLUMNS, RISK_THRESHOLD_HIGH, RISK_THRESHOLD_LOW
    from src.data_pipeline import clean_and_feature_engineer
    from src.explainability import explain_prediction, simulate_what_if_counterfactual, generate_prescriptive_strategy

app = FastAPI(
    title="Customer Churn Intelligence & Predictive Analytics API",
    description="Enterprise Machine Learning REST Microservice for real-time customer attrition scoring, XAI attributions, and what-if simulation.",
    version="2.0.0"
)

# Load Artifacts
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
features = joblib.load(os.path.join(MODELS_DIR, "features.joblib"))
xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.joblib"))
rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.joblib"))
mlp_model = joblib.load(os.path.join(MODELS_DIR, "mlp_neural_network_model.joblib"))
lr_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression_model.joblib"))

class CustomerProfile(BaseModel):
    customerID: Optional[str] = "CUST-API-001"
    gender: str = Field(default="Male", description="Male or Female")
    SeniorCitizen: int = Field(default=0, ge=0, le=1)
    Partner: str = Field(default="No", description="Yes or No")
    Dependents: str = Field(default="No", description="Yes or No")
    tenure: int = Field(default=6, ge=1, le=72)
    PhoneService: str = Field(default="Yes")
    MultipleLines: str = Field(default="No")
    InternetService: str = Field(default="Fiber optic")
    OnlineSecurity: str = Field(default="No")
    OnlineBackup: str = Field(default="No")
    DeviceProtection: str = Field(default="No")
    TechSupport: str = Field(default="No")
    StreamingTV: str = Field(default="Yes")
    StreamingMovies: str = Field(default="Yes")
    Contract: str = Field(default="Month-to-month")
    PaperlessBilling: str = Field(default="Yes")
    PaymentMethod: str = Field(default="Electronic check")
    MonthlyCharges: float = Field(default=85.0, ge=18.0, le=140.0)
    TotalCharges: Optional[float] = None
    model_choice: Optional[str] = "XGBoost"

class WhatIfRequest(BaseModel):
    customer: CustomerProfile
    interventions: Dict[str, Any]

@app.get("/health", tags=["System"])
def healthcheck():
    return {
        "status": "HEALTHY",
        "service": "Customer Churn Intelligence Engine",
        "active_models": ["XGBoost", "Random Forest", "Deep Neural Network", "Logistic Regression"],
        "feature_dimensions": len(features)
    }

def resolve_model(choice: str):
    if "rf" in choice.lower() or "random" in choice.lower():
        return rf_model
    elif "mlp" in choice.lower() or "neural" in choice.lower():
        return mlp_model
    elif "logistic" in choice.lower() or "lr" in choice.lower():
        return lr_model
    return xgb_model

@app.post("/api/v1/predict", tags=["Inference"])
def predict_churn(customer: CustomerProfile):
    c_dict = customer.model_dump()
    if c_dict.get('TotalCharges') is None:
        c_dict['TotalCharges'] = c_dict['MonthlyCharges'] * c_dict['tenure']

    model = resolve_model(customer.model_choice)
    df = pd.DataFrame([c_dict])
    proc = clean_and_feature_engineer(df)[ALL_FEATURE_COLUMNS]
    scaled = scaler.transform(proc)

    prob = float(model.predict_proba(scaled)[0, 1])
    tier = "Critical Risk" if prob >= RISK_THRESHOLD_HIGH else ("Moderate Risk" if prob >= RISK_THRESHOLD_LOW else "Low Risk / Loyal")

    clv = round(customer.MonthlyCharges * (customer.tenure + 24) * 0.75, 2)
    revenue_at_risk = round(customer.MonthlyCharges * 12, 2)

    return {
        "customerID": customer.customerID,
        "churn_probability": round(prob, 4),
        "churn_percentage": f"{prob * 100:.2f}%",
        "risk_tier": tier,
        "estimated_clv": clv,
        "annual_revenue_at_risk": revenue_at_risk,
        "model_used": customer.model_choice
    }

@app.post("/api/v1/explain", tags=["Explainable AI"])
def explain_churn(customer: CustomerProfile):
    c_dict = customer.model_dump()
    if c_dict.get('TotalCharges') is None:
        c_dict['TotalCharges'] = c_dict['MonthlyCharges'] * c_dict['tenure']

    model = resolve_model(customer.model_choice)
    explanation = explain_prediction(c_dict, model, scaler, features)
    prescriptions = generate_prescriptive_strategy(c_dict, explanation["churn_probability"])

    return {
        "customerID": customer.customerID,
        "explanation": explanation,
        "prescriptions": prescriptions
    }

@app.post("/api/v1/simulate", tags=["Prescriptive Intelligence"])
def simulate_counterfactual(request: WhatIfRequest):
    c_dict = request.customer.model_dump()
    if c_dict.get('TotalCharges') is None:
        c_dict['TotalCharges'] = c_dict['MonthlyCharges'] * c_dict['tenure']

    model = resolve_model(request.customer.model_choice)
    sim_res = simulate_what_if_counterfactual(c_dict, request.interventions, model, scaler)
    return sim_res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
