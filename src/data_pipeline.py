"""
Enterprise Data Ingestion, Cleaning & Feature Engineering Pipeline
Includes Customer Lifetime Value (CLV) estimation, Charge Ratio engineering,
and stratified train-test splitting.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from config import ALL_FEATURE_COLUMNS, RANDOM_STATE, TEST_SPLIT_SIZE, MODELS_DIR
except ImportError:
    from src.config import ALL_FEATURE_COLUMNS, RANDOM_STATE, TEST_SPLIT_SIZE, MODELS_DIR

def clean_and_feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw customer data and computes advanced engineered features:
    - Imputes total charges using tenure * monthly charges
    - Calculates ChargeRatio (ratio of billed charges to expected charges)
    - Estimates Customer Lifetime Value (CLV)
    - Encodes binary and one-hot categorical features
    """
    data = df.copy()

    # 1. Total Charges Cleaning & Imputation
    if 'TotalCharges' in data.columns:
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        # Impute missing values with monthly charge * max(1, tenure)
        tenure_safe = data['tenure'].replace(0, 1) if 'tenure' in data.columns else 1
        data['TotalCharges'] = data['TotalCharges'].fillna(data['MonthlyCharges'] * tenure_safe)
    else:
        data['TotalCharges'] = data['MonthlyCharges'] * data['tenure']

    # 2. Advanced Feature Engineering
    # Feature A: ChargeRatio (detects unexpected billing variations or rate increases)
    expected_charges = data['tenure'] * data['MonthlyCharges'] + 1e-5
    data['ChargeRatio'] = (data['TotalCharges'] / expected_charges).clip(0.5, 2.0)

    # Feature B: Estimated Customer Lifetime Value (CLV)
    # Forward-looking discounted cash flow estimate assuming average 24-month retention horizon
    data['EstimatedCLV'] = (data['MonthlyCharges'] * (data['tenure'] + 24) * 0.75).round(2)

    # 3. Binary Encodings (Yes -> 1, No -> 0)
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in data.columns:
            data[col] = data[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    # Sub-Service Encodings (Yes -> 1, Any other / No service -> 0)
    service_cols = [
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in service_cols:
        if col in data.columns:
            data[col] = data[col].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

    # 4. Multi-Category Encoding
    # InternetService: DSL is baseline (0,0)
    if 'InternetService' in data.columns:
        data['InternetService_Fiber_optic'] = (data['InternetService'] == 'Fiber optic').astype(int)
        data['InternetService_No'] = (data['InternetService'] == 'No').astype(int)
    else:
        data['InternetService_Fiber_optic'] = 0
        data['InternetService_No'] = 0

    # Contract: Month-to-month is baseline (0,0)
    if 'Contract' in data.columns:
        data['Contract_One_year'] = (data['Contract'] == 'One year').astype(int)
        data['Contract_Two_year'] = (data['Contract'] == 'Two year').astype(int)
    else:
        data['Contract_One_year'] = 0
        data['Contract_Two_year'] = 0

    # PaymentMethod: Bank transfer is baseline (0,0,0)
    if 'PaymentMethod' in data.columns:
        data['PaymentMethod_Credit_card_automatic'] = (data['PaymentMethod'] == 'Credit card (automatic)').astype(int)
        data['PaymentMethod_Electronic_check'] = (data['PaymentMethod'] == 'Electronic check').astype(int)
        data['PaymentMethod_Mailed_check'] = (data['PaymentMethod'] == 'Mailed check').astype(int)
    else:
        data['PaymentMethod_Credit_card_automatic'] = 0
        data['PaymentMethod_Electronic_check'] = 0
        data['PaymentMethod_Mailed_check'] = 0

    # Guarantee all 24 required feature columns exist
    for col in ALL_FEATURE_COLUMNS:
        if col not in data.columns:
            data[col] = 0

    return data

def run_data_pipeline(data_path: str, models_dir: str = MODELS_DIR):
    """
    Ingests raw dataset, applies feature engineering, performs stratified 80/20 split,
    fits StandardScaler, and persists preprocessing artifacts.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print(f"[Pipeline] Ingesting dataset from: {data_path}")
    raw_df = pd.read_csv(data_path)

    # Target variable extraction
    y = raw_df['Churn'].map({'Yes': 1, 'No': 0}).values

    # Preprocessing
    processed_df = clean_and_feature_engineer(raw_df)
    X = processed_df[ALL_FEATURE_COLUMNS]

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SPLIT_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Persist artifacts
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(ALL_FEATURE_COLUMNS, os.path.join(models_dir, "features.joblib"))
    print(f"[Pipeline] Successfully processed {len(raw_df)} records. Features: {len(ALL_FEATURE_COLUMNS)} dimensions.")

    return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
