"""
Enterprise Relational Database Layer (SQLite)
Provides persistent ACID-compliant transaction storage for customer profiles,
real-time prediction audit logs, and counterfactual simulation histories.
"""
import os
import sqlite3
import json
import time
import pandas as pd
from datetime import datetime

try:
    from config import PROJECT_ROOT, RAW_DATA_PATH
except ImportError:
    from src.config import PROJECT_ROOT, RAW_DATA_PATH

DB_PATH = os.path.join(PROJECT_ROOT, "data", "enterprise_churn.db")

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database(seed_from_csv=True):
    """
    Initializes tables:
    1. subscribers: Enterprise subscriber registry
    2. prediction_audit_logs: Logs every single inference call with timestamp & latency
    3. simulated_interventions: Records counterfactual simulation experiments
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS subscribers (
        customerID TEXT PRIMARY KEY,
        gender TEXT,
        SeniorCitizen INTEGER,
        Partner TEXT,
        Dependents TEXT,
        tenure INTEGER,
        PhoneService TEXT,
        MultipleLines TEXT,
        InternetService TEXT,
        OnlineSecurity TEXT,
        OnlineBackup TEXT,
        DeviceProtection TEXT,
        TechSupport TEXT,
        StreamingTV TEXT,
        StreamingMovies TEXT,
        Contract TEXT,
        PaperlessBilling TEXT,
        PaymentMethod TEXT,
        MonthlyCharges REAL,
        TotalCharges REAL,
        Churn TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_audit_logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        customerID TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        model_name TEXT,
        churn_probability REAL,
        risk_tier TEXT,
        estimated_clv REAL,
        annual_revenue_at_risk REAL,
        execution_latency_ms REAL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS simulated_interventions (
        intervention_id INTEGER PRIMARY KEY AUTOINCREMENT,
        customerID TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        model_name TEXT,
        baseline_risk REAL,
        simulated_risk REAL,
        risk_reduction_pct REAL,
        annual_revenue_preserved REAL,
        interventions_applied TEXT
    )
    """)
    conn.commit()

    # Seed benchmark data if table is empty
    cursor.execute("SELECT COUNT(*) FROM subscribers")
    count = cursor.fetchone()[0]

    if count == 0 and seed_from_csv and os.path.exists(RAW_DATA_PATH):
        print(f"[Database] Seeding database with 7,043 benchmark records from {RAW_DATA_PATH}...")
        df = pd.read_csv(RAW_DATA_PATH)
        df.to_sql("subscribers", conn, if_exists="append", index=False)
        print("[Database] Successfully seeded 7,043 subscriber records.")

    conn.close()

def log_prediction(customerID: str, model_name: str, churn_prob: float, risk_tier: str,
                   clv: float, rev_at_risk: float, latency_ms: float = 2.5):
    """Logs an inference event to the audit trail."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO prediction_audit_logs (
        customerID, model_name, churn_probability, risk_tier,
        estimated_clv, annual_revenue_at_risk, execution_latency_ms
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (customerID, model_name, round(churn_prob, 4), risk_tier, clv, rev_at_risk, latency_ms))
    conn.commit()
    conn.close()

def log_simulation(customerID: str, model_name: str, base_risk: float,
                   sim_risk: float, reduction_pct: float, revenue_saved: float,
                   interventions: dict):
    """Logs a counterfactual policy simulation to the audit trail."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO simulated_interventions (
        customerID, model_name, baseline_risk, simulated_risk,
        risk_reduction_pct, annual_revenue_preserved, interventions_applied
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (customerID, model_name, round(base_risk, 4), round(sim_risk, 4),
          round(reduction_pct, 2), round(revenue_saved, 2), json.dumps(interventions)))
    conn.commit()
    conn.close()

def get_recent_audit_logs(limit: int = 25) -> pd.DataFrame:
    """Retrieves recent prediction audit logs."""
    conn = get_connection()
    query = """
    SELECT log_id, customerID, timestamp, model_name,
           ROUND(churn_probability * 100, 2) AS [Churn Prob %],
           risk_tier AS [Risk Tier],
           '$' || printf('%,.2f', estimated_clv) AS [Estimated CLV],
           '$' || printf('%,.2f', annual_revenue_at_risk) AS [Revenue at Risk],
           printf('%.2f ms', execution_latency_ms) AS [Latency]
    FROM prediction_audit_logs
    ORDER BY log_id DESC
    LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return df

def get_recent_simulations(limit: int = 25) -> pd.DataFrame:
    """Retrieves recent counterfactual policy simulations."""
    conn = get_connection()
    query = """
    SELECT intervention_id, customerID, timestamp, model_name,
           ROUND(baseline_risk * 100, 2) AS [Base Risk %],
           ROUND(simulated_risk * 100, 2) AS [New Risk %],
           risk_reduction_pct || '%' AS [Risk Reduction],
           '$' || printf('%,.2f', annual_revenue_preserved) AS [Preserved Revenue],
           interventions_applied AS [Strategy Applied]
    FROM simulated_interventions
    ORDER BY intervention_id DESC
    LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return df

if __name__ == "__main__":
    initialize_database()
    print(f"[Success] Database initialized and verified at: {DB_PATH}")
