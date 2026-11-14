"""
Dataset Generator for Customer Churn Intelligence & Predictive Analytics Platform
Generates a realistic benchmark dataset (7,043 records) following the Telco Churn standard
matching the feature specifications detailed in Chapter 6 (SRS) of the report.
"""
import os
import random
import csv
import math

def generate_telecom_churn_data(filepath, n_samples=7043, seed=42):
    random.seed(seed)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    headers = [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
    ]
    
    payment_methods = [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    
    records = []
    
    for i in range(1, n_samples + 1):
        cid = f"{random.randint(1000, 9999)}-{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=5))}"
        gender = random.choice(["Male", "Female"])
        senior = 1 if random.random() < 0.16 else 0
        partner = "Yes" if random.random() < 0.48 else "No"
        dependents = "Yes" if (partner == "Yes" and random.random() < 0.6) else ("Yes" if random.random() < 0.1 else "No")
        
        # Tenure distributions: bi-modal (many short tenure, many long tenure)
        if random.random() < 0.45:
            tenure = random.randint(1, 12)
        elif random.random() < 0.65:
            tenure = random.randint(13, 40)
        else:
            tenure = random.randint(41, 72)
            
        phone = "Yes" if random.random() < 0.90 else "No"
        if phone == "Yes":
            multiple_lines = "Yes" if random.random() < 0.42 else "No"
        else:
            multiple_lines = "No phone service"
            
        internet = random.choices(["Fiber optic", "DSL", "No"], weights=[0.44, 0.34, 0.22])[0]
        
        if internet != "No":
            online_sec = "Yes" if random.random() < 0.35 else "No"
            online_bak = "Yes" if random.random() < 0.40 else "No"
            dev_prot = "Yes" if random.random() < 0.39 else "No"
            tech_sup = "Yes" if random.random() < 0.35 else "No"
            stream_tv = "Yes" if random.random() < 0.45 else "No"
            stream_mov = "Yes" if random.random() < 0.45 else "No"
        else:
            online_sec = "No internet service"
            online_bak = "No internet service"
            dev_prot = "No internet service"
            tech_sup = "No internet service"
            stream_tv = "No internet service"
            stream_mov = "No internet service"
            
        if tenure < 12:
            contract = random.choices(["Month-to-month", "One year", "Two year"], weights=[0.85, 0.12, 0.03])[0]
        elif tenure < 36:
            contract = random.choices(["Month-to-month", "One year", "Two year"], weights=[0.45, 0.40, 0.15])[0]
        else:
            contract = random.choices(["Month-to-month", "One year", "Two year"], weights=[0.20, 0.35, 0.45])[0]
            
        paperless = "Yes" if random.random() < 0.59 else "No"
        payment = random.choice(payment_methods)
        
        # Monthly charges calculation based on services
        base_charge = 20.0
        if phone == "Yes":
            base_charge += 5.0
        if multiple_lines == "Yes":
            base_charge += 10.0
        if internet == "DSL":
            base_charge += 25.0
        elif internet == "Fiber optic":
            base_charge += 45.0
            
        for s in [online_sec, online_bak, dev_prot, tech_sup, stream_tv, stream_mov]:
            if s == "Yes":
                base_charge += 7.0
                
        # Add small random variation
        monthly = round(base_charge + random.uniform(-3.0, 4.0), 2)
        total = round(monthly * tenure + random.uniform(-10.0, 10.0), 2)
        if total < monthly:
            total = monthly
            
        # Realistic Churn Probability Model
        churn_score = -1.2  # baseline
        if contract == "Month-to-month":
            churn_score += 1.3
        elif contract == "Two year":
            churn_score -= 1.4
            
        if internet == "Fiber optic":
            churn_score += 0.6
        if tech_sup == "No" and internet != "No":
            churn_score += 0.5
        if online_sec == "No" and internet != "No":
            churn_score += 0.4
        if payment == "Electronic check":
            churn_score += 0.5
            
        if tenure <= 6:
            churn_score += 0.9
        elif tenure >= 48:
            churn_score -= 1.1
            
        if monthly > 85:
            churn_score += 0.4
            
        # Sigmoid probability
        prob = 1.0 / (1.0 + math.exp(-churn_score))
        churn = "Yes" if random.random() < prob else "No"
        
        records.append([
            cid, gender, senior, partner, dependents, tenure, phone,
            multiple_lines, internet, online_sec, online_bak, dev_prot,
            tech_sup, stream_tv, stream_mov, contract, paperless,
            payment, monthly, total, churn
        ])
        
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(records)
        
    print(f"Successfully generated {len(records)} customer records at {filepath}")

if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "customer_churn_data.csv")
    generate_telecom_churn_data(out_path)
