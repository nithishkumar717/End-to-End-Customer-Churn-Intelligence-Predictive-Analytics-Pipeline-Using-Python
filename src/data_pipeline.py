"""
Data Ingestion, Preprocessing & Feature Engineering Pipeline
Handles data cleaning, missing value imputation, categorical encoding, and feature scaling.
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
    'MonthlyCharges', 'TotalCharges', 'ChargeRatio',
    'InternetService_Fiber_optic', 'InternetService_No',
    'Contract_One_year', 'Contract_Two_year',
    'PaymentMethod_Credit_card_automatic',
    'PaymentMethod_Electronic_check',
    'PaymentMethod_Mailed_check'
]

def clean_and_feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    
    # 1. Clean TotalCharges (handle whitespace/nulls)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'] = data['TotalCharges'].fillna(data['MonthlyCharges'] * data['tenure'])
    
    # 2. Feature Engineering: ChargeRatio = TotalCharges / (tenure * MonthlyCharges + 1e-5)
    data['ChargeRatio'] = data['TotalCharges'] / (data['tenure'] * data['MonthlyCharges'] + 1e-5)
    data['ChargeRatio'] = data['ChargeRatio'].clip(0.5, 2.0)
    
    # 3. Binary encodings (Yes/No -> 1/0)
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in data.columns:
            data[col] = data[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
            
    # Sub-service columns (Yes -> 1, No/No service -> 0)
    service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        if col in data.columns:
            data[col] = data[col].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
            
    # 4. Multi-category columns via One-Hot Encoding
    # InternetService: reference is DSL
    if 'InternetService' in data.columns:
        data['InternetService_Fiber_optic'] = (data['InternetService'] == 'Fiber optic').astype(int)
        data['InternetService_No'] = (data['InternetService'] == 'No').astype(int)
    else:
        data['InternetService_Fiber_optic'] = 0
        data['InternetService_No'] = 0
        
    # Contract: reference is Month-to-month
    if 'Contract' in data.columns:
        data['Contract_One_year'] = (data['Contract'] == 'One year').astype(int)
        data['Contract_Two_year'] = (data['Contract'] == 'Two year').astype(int)
    else:
        data['Contract_One_year'] = 0
        data['Contract_Two_year'] = 0
        
    # PaymentMethod: reference is Bank transfer (automatic)
    if 'PaymentMethod' in data.columns:
        data['PaymentMethod_Credit_card_automatic'] = (data['PaymentMethod'] == 'Credit card (automatic)').astype(int)
        data['PaymentMethod_Electronic_check'] = (data['PaymentMethod'] == 'Electronic check').astype(int)
        data['PaymentMethod_Mailed_check'] = (data['PaymentMethod'] == 'Mailed check').astype(int)
    else:
        data['PaymentMethod_Credit_card_automatic'] = 0
        data['PaymentMethod_Electronic_check'] = 0
        data['PaymentMethod_Mailed_check'] = 0
        
    # Ensure all required feature columns exist
    for col in FEATURE_COLUMNS:
        if col not in data.columns:
            data[col] = 0
            
    return data

def run_data_pipeline(data_path: str, models_dir: str):
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Target variable
    y = df['Churn'].map({'Yes': 1, 'No': 0}).values
    
    # Process features
    processed_df = clean_and_feature_engineer(df)
    X = processed_df[FEATURE_COLUMNS]
    
    # Train / Test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Fit StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save artifacts
    os.makedirs(models_dir, exist_ok=True)
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    features_path = os.path.join(models_dir, "features.joblib")
    
    joblib.dump(scaler, scaler_path)
    joblib.dump(FEATURE_COLUMNS, features_path)
    print(f"Preprocessing artifacts saved to {models_dir}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
