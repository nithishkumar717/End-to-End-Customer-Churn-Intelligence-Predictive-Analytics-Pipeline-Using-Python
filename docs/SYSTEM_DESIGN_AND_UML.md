# System Architecture, DFD & UML Design Specifications

**Project:** End-to-End Customer Churn Intelligence & Predictive Analytics Pipeline Using Python  
**Candidate:** Nithishkumar A S (Register Number: P25CA0044)  
**Degree:** Master of Computer Applications (MCA), BIHER

---

## 1. Data Flow Diagrams (DFD)

### 1.1 DFD Level 0 (Context Diagram)

```mermaid
graph TD
    User["Customer Retention Manager / Analyst"] -->|"1. Input Subscriber Profile or Batch CSV"| System["Customer Churn Intelligence Platform"]
    System -->|"2. Return Churn Probability (%), Risk Badges, XAI Drivers & Prescriptive Plan"| User
    Admin["System Administrator / Data Scientist"] -->|"3. Ingest Raw Datasets & Trigger Retraining"| System
    System -->|"4. Return Diagnostic Benchmark Curves & Metrics Summary"| Admin
```

---

### 1.2 DFD Level 1 (Functional Decomposition)

```mermaid
flowchart TD
    D1[("Raw Telecom Dataset / CSV")] --> P1["1.0 Data Ingestion & Imputation Engine"]
    P1 --> P2["2.0 Feature Engineering & Normalization\n(ChargeRatio, Estimated CLV, Encoding)"]
    P2 --> D2[("Feature Vector Storage (24 Dimensions)")]
    
    D2 --> P3["3.0 Multi-Model Inference Core\n(XGBoost, Random Forest, DNN, Logistic)"]
    P3 --> P4["4.0 Explainable AI (XAI) Attribution Engine"]
    P3 --> P5["5.0 Counterfactual 'What-If' Policy Simulator"]
    
    P4 --> P6["6.0 Presentation & Microservice Gateway\n(Streamlit Dashboard + FastAPI REST)"]
    P5 --> P6
    P6 --> Report["Scored Customer Risk Reports & Interactive Visuals"]
```

---

## 2. UML Use Case Diagram

```mermaid
flowchart LR
    subgraph SystemBoundary ["Customer Churn Intelligence Platform"]
        UC1(["Authenticate & Select Inference Engine"])
        UC2(["Input Single Customer Profile"])
        UC3(["View Churn Probability & Risk Tier"])
        UC4(["Inspect Local XAI Churn Drivers"])
        UC5(["Execute What-If Counterfactual Simulation"])
        UC6(["Upload Enterprise Batch CSV"])
        UC7(["Download Scored Risk Audit Report"])
        UC8(["Retrain ML & Deep Learning Models"])
        UC9(["Inspect Learning Curves & Confusion Matrices"])
    end

    CRM["Retention Manager / Account Executive"] --> UC1
    CRM --> UC2
    CRM --> UC3
    CRM --> UC4
    CRM --> UC5
    CRM --> UC6
    CRM --> UC7

    DS["Lead Data Scientist / Administrator"] --> UC1
    DS --> UC8
    DS --> UC9
    DS --> UC7
```

---

## 3. UML Sequence Diagram (Prediction & Simulation Flow)

```mermaid
sequenceDiagram
    autonumber
    actor User as Retention Specialist
    participant UI as Streamlit Web Dashboard
    participant API as Feature Pipeline
    participant Model as XGBoost / DNN Model
    participant XAI as Explainability Engine

    User->>UI: Selects parameters & clicks 'Run Diagnostic'
    UI->>API: Sends raw customer JSON dictionary
    API->>API: Imputes missing values, computes ChargeRatio & CLV
    API->>API: Standardizes 24-dimensional feature vector
    API->>Model: Passes scaled feature array
    Model-->>API: Returns class probability score P(Churn)
    API->>XAI: Requests local feature attribution
    XAI-->>API: Returns Top Churn Drivers & Loyalty Anchors
    API-->>UI: Compiles Probability, Risk Badge & Prescriptions
    UI-->>User: Displays live Risk Gauge, XAI Pills & Revenue Exposure

    opt What-If Policy Simulation
        User->>UI: Modifies contract terms & discount offers
        UI->>API: Passes base profile + counterfactual interventions
        API->>Model: Re-scores intervened profile
        Model-->>API: Returns new P(Churn_simulated)
        API-->>UI: Returns Delta P & Annual Revenue Preserved ($)
        UI-->>User: Displays real-time risk reduction metric
    end
```

---

## 4. Entity-Relationship (ER) Schema Specifications

```mermaid
erDiagram
    CUSTOMER ||--o{ SUBSCRIPTION_SERVICES : has
    CUSTOMER ||--|| BILLING_CONTRACT : possesses
    CUSTOMER ||--o{ CHURN_PREDICTION_LOG : generates
    CHURN_PREDICTION_LOG ||--o{ RETENTION_INTERVENTION : triggers

    CUSTOMER {
        string customerID PK
        string gender
        int SeniorCitizen
        string Partner
        string Dependents
        int tenure
    }

    SUBSCRIPTION_SERVICES {
        string serviceID PK
        string customerID FK
        string PhoneService
        string MultipleLines
        string InternetService
        string OnlineSecurity
        string OnlineBackup
        string DeviceProtection
        string TechSupport
        string StreamingTV
        string StreamingMovies
    }

    BILLING_CONTRACT {
        string contractID PK
        string customerID FK
        string ContractType
        string PaperlessBilling
        string PaymentMethod
        float MonthlyCharges
        float TotalCharges
        float ChargeRatio
        float EstimatedCLV
    }

    CHURN_PREDICTION_LOG {
        string predictionID PK
        string customerID FK
        float ChurnProbability
        string RiskCategory
        string ModelUsed
        datetime Timestamp
    }

    RETENTION_INTERVENTION {
        string interventionID PK
        string predictionID FK
        string RecommendedOffer
        float RevenuePreserved
        string Status
    }
```
