"""
Central Configuration & Hyperparameter Registry
Defines feature sets, model hyperparameters, path constants, and business risk thresholds.
"""
import os

# Project Directory Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RAW_DATA_PATH = os.path.join(DATA_DIR, "customer_churn_data.csv")

# Risk Assessment Thresholds
RISK_THRESHOLD_LOW = 0.35      # < 35% probability = Low Risk / Loyal
RISK_THRESHOLD_HIGH = 0.65     # > 65% probability = High Risk / Critical

# Feature Schema Definitions
NUMERICAL_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges', 'ChargeRatio', 'EstimatedCLV']

CATEGORICAL_BINARY = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

SERVICE_FEATURES = [
    'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]

ONE_HOT_FEATURES = {
    'InternetService': ['Fiber optic', 'No'],     # DSL is baseline
    'Contract': ['One year', 'Two year'],          # Month-to-month is baseline
    'PaymentMethod': [
        'Credit card (automatic)',
        'Electronic check',
        'Mailed check'
    ]                                              # Bank transfer is baseline
}

ALL_FEATURE_COLUMNS = [
    'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
    'MonthlyCharges', 'TotalCharges', 'ChargeRatio', 'EstimatedCLV',
    'InternetService_Fiber_optic', 'InternetService_No',
    'Contract_One_year', 'Contract_Two_year',
    'PaymentMethod_Credit_card_automatic',
    'PaymentMethod_Electronic_check',
    'PaymentMethod_Mailed_check'
]

# Random State for Reproducibility
RANDOM_STATE = 42
TEST_SPLIT_SIZE = 0.20
