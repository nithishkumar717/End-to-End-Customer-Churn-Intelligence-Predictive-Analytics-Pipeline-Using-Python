"""
Explainable AI (XAI) & Prescriptive Retention Intelligence Engine
Implements local feature attribution analysis, counterfactual 'what-if' simulation,
and automated retention action plan synthesis.
"""
import numpy as np
import pandas as pd

try:
    from data_pipeline import clean_and_feature_engineer
    from config import ALL_FEATURE_COLUMNS, RISK_THRESHOLD_HIGH, RISK_THRESHOLD_LOW
except ImportError:
    from src.data_pipeline import clean_and_feature_engineer
    from src.config import ALL_FEATURE_COLUMNS, RISK_THRESHOLD_HIGH, RISK_THRESHOLD_LOW

def explain_prediction(input_row: dict, model, scaler, feature_names: list) -> dict:
    """
    Computes local feature attribution by comparing the scaled customer feature
    vector against baseline expected coefficients and model feature importances.
    Returns categorized risk amplifiers and risk mitigators.
    """
    df = pd.DataFrame([input_row])
    processed = clean_and_feature_engineer(df)[ALL_FEATURE_COLUMNS]
    scaled = scaler.transform(processed)[0]

    # Model probability
    churn_prob = float(model.predict_proba(scaler.transform(processed))[0, 1])

    # Global feature importances
    if hasattr(model, 'feature_importances_'):
        global_imp = model.feature_importances_
    elif hasattr(model, 'coef_'):
        global_imp = np.abs(model.coef_[0])
    else:
        global_imp = np.ones(len(feature_names)) / len(feature_names)

    # Local feature impact score: scaled_value * global_importance
    impact_scores = scaled * global_imp

    amplifiers = []
    mitigators = []

    for name, score, raw_val in zip(feature_names, impact_scores, processed.iloc[0]):
        clean_name = name.replace('_', ' ').replace('PaymentMethod ', '').replace('InternetService ', '')
        if score > 0.015:
            amplifiers.append({
                "feature": clean_name,
                "impact": float(score),
                "raw_value": raw_val,
                "type": "Churn Driver"
            })
        elif score < -0.015:
            mitigators.append({
                "feature": clean_name,
                "impact": float(abs(score)),
                "raw_value": raw_val,
                "type": "Loyalty Anchor"
            })

    amplifiers = sorted(amplifiers, key=lambda x: x['impact'], reverse=True)
    mitigators = sorted(mitigators, key=lambda x: x['impact'], reverse=True)

    return {
        "churn_probability": churn_prob,
        "risk_tier": "Critical Risk" if churn_prob >= RISK_THRESHOLD_HIGH else ("Moderate Risk" if churn_prob >= RISK_THRESHOLD_LOW else "Low Risk / Loyal"),
        "top_churn_drivers": amplifiers[:5],
        "top_loyalty_anchors": mitigators[:5]
    }

def simulate_what_if_counterfactual(base_input: dict, interventions: dict, model, scaler) -> dict:
    """
    Simulates a counterfactual intervention:
    e.g., changes contract type, adds tech support, or offers monthly discount.
    Calculates the resulting delta in churn probability and preserved Customer Lifetime Value.
    """
    # 1. Base prediction
    base_df = pd.DataFrame([base_input])
    base_proc = clean_and_feature_engineer(base_df)[ALL_FEATURE_COLUMNS]
    base_prob = float(model.predict_proba(scaler.transform(base_proc))[0, 1])

    # 2. Counterfactual prediction
    cf_input = base_input.copy()
    cf_input.update(interventions)
    cf_df = pd.DataFrame([cf_input])
    cf_proc = clean_and_feature_engineer(cf_df)[ALL_FEATURE_COLUMNS]
    cf_prob = float(model.predict_proba(scaler.transform(cf_proc))[0, 1])

    prob_delta = base_prob - cf_prob
    pct_reduction = (prob_delta / (base_prob + 1e-6)) * 100

    # Financial estimation
    monthly = float(base_input.get('MonthlyCharges', 75.0))
    annual_revenue_at_risk = monthly * 12
    revenue_preserved = annual_revenue_at_risk * max(0.0, prob_delta)

    return {
        "base_churn_prob": base_prob,
        "simulated_churn_prob": cf_prob,
        "absolute_reduction": prob_delta,
        "percentage_risk_reduction": pct_reduction,
        "annual_revenue_at_risk": annual_revenue_at_risk,
        "annual_revenue_preserved": revenue_preserved,
        "interventions_applied": interventions
    }

def generate_prescriptive_strategy(customer_data: dict, churn_prob: float) -> list:
    """
    Generates actionable, rule-governed business prescriptions tailored
    to individual customer contract, service, and payment behaviors.
    """
    strategies = []
    contract = customer_data.get('Contract', '')
    internet = customer_data.get('InternetService', '')
    tech_sup = customer_data.get('TechSupport', '')
    payment = customer_data.get('PaymentMethod', '')
    monthly = float(customer_data.get('MonthlyCharges', 0.0))
    tenure = int(customer_data.get('tenure', 0))

    if churn_prob >= RISK_THRESHOLD_HIGH:
        if contract == "Month-to-month":
            strategies.append({
                "priority": "HIGH",
                "action": "Contract Stabilization Offer",
                "detail": "Customer is on Month-to-Month billing with high churn risk. Propose a 1-Year commitment bundled with an immediate $15/month promotional rate credit for the first 3 billing cycles."
            })
        if internet == "Fiber optic" and tech_sup != "Yes":
            strategies.append({
                "priority": "HIGH",
                "action": "Proactive Tech Support Onboarding",
                "detail": "Fiber optic subscribers without dedicated tech support exhibit 2.4x higher attrition. Grant complimentary 6-month Premium Technical Support."
            })
        if payment == "Electronic check":
            strategies.append({
                "priority": "MEDIUM",
                "action": "Automated Billing Incentive",
                "detail": "Electronic check payments suffer from higher manual friction and payment failure rates. Incentivize Auto-Pay (Credit Card/ACH) with a one-time $20 statement rebate."
            })
        if tenure <= 6:
            strategies.append({
                "priority": "URGENT",
                "action": "Early-Tenure Concierge Check-in",
                "detail": "Customer is in the critical 90-day retention window. Dispatch a Customer Success outreach within 24 hours to address onboarding dissatisfaction."
            })
    elif churn_prob >= RISK_THRESHOLD_LOW:
        strategies.append({
            "priority": "MEDIUM",
            "action": "Mid-Tier Loyalty Engagement",
            "detail": "Account displays moderate stability. Offer loyalty milestone rewards, quarterly feature newsletters, and proactive customer satisfaction survey."
        })
    else:
        strategies.append({
            "priority": "LOW",
            "action": "Retention Maintenance & Advocacy",
            "detail": "Account demonstrates high loyalty and low attrition probability. Candidate for annual plan upgrade or referral program incentive."
        })

    return strategies
