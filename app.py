"""
========================================================================================
Enterprise Customer Churn Intelligence & Predictive Analytics Platform
Master of Computer Applications (MCA) Final Year Project
Department of Computer Science & Applications, BIHER
========================================================================================
"""
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import time

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Enterprise Churn Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Professional CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .header-box {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
        padding: 1.8rem 2rem;
        border-radius: 12px;
        color: #FFFFFF;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .header-box h1 {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        color: #F8FAFC;
    }
    .header-box p {
        font-size: 1.05rem;
        color: #94A3B8;
        margin: 0.4rem 0 0 0;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #1E3A8A;
    }
    .kpi-lbl {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 800;
        display: inline-block;
        border: 1px solid #FCA5A5;
    }
    .badge-moderate {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 800;
        display: inline-block;
        border: 1px solid #FCD34D;
    }
    .badge-loyal {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 800;
        display: inline-block;
        border: 1px solid #86EFAC;
    }
    .xai-pill {
        background-color: #EFF6FF;
        border-left: 4px solid #2563EB;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0 6px 6px 0;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

@st.cache_resource
def load_all_artifacts():
    try:
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
        features = joblib.load(os.path.join(MODELS_DIR, "features.joblib"))
        lr_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression_model.joblib"))
        rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.joblib"))
        xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.joblib"))
        mlp_model = joblib.load(os.path.join(MODELS_DIR, "mlp_neural_network_model.joblib"))
        summary = joblib.load(os.path.join(MODELS_DIR, "metrics_summary.joblib"))
        stat_summary = joblib.load(os.path.join(MODELS_DIR, "statistical_summary.joblib")) if os.path.exists(os.path.join(MODELS_DIR, "statistical_summary.joblib")) else None
        return scaler, features, lr_model, rf_model, xgb_model, mlp_model, summary, stat_summary
    except Exception as e:
        return None, None, None, None, None, None, None, None

scaler, features, lr_model, rf_model, xgb_model, mlp_model, summary, stat_summary = load_all_artifacts()

# Sidebar Navigation
st.sidebar.title("⚡ Navigation")
page = st.sidebar.radio(
    "Select Enterprise View:",
    [
        "🏠 Executive KPI Overview",
        "🎯 Customer Risk Profiler & XAI",
        "🔄 'What-If' Policy Simulator",
        "📂 Enterprise Batch Auditing",
        "📊 Model Benchmarks & Statistics",
        "🗄️ Database Audit & Governance Logs"
    ]
)

st.sidebar.markdown("---")
active_algo_choice = st.sidebar.selectbox(
    "Active Inference Engine:",
    [
        "XGBoost Classifier (Champion)",
        "Random Forest Classifier",
        "Deep Neural Network (MLP)",
        "Logistic Regression (Baseline)"
    ]
)

def get_selected_model(choice_str):
    if "XGBoost" in choice_str:
        return xgb_model, "XGBoost"
    elif "Random Forest" in choice_str:
        return rf_model, "Random Forest"
    elif "Deep Neural" in choice_str:
        return mlp_model, "Deep Neural Network"
    else:
        return lr_model, "Logistic Regression"

active_model, active_model_name = get_selected_model(active_algo_choice)

# ======================================================================================
# VIEW 1: EXECUTIVE KPI OVERVIEW
# ======================================================================================
if page == "🏠 Executive KPI Overview":
    st.markdown("""
    <div class="header-box">
        <h1>Customer Churn Intelligence & Predictive Analytics Platform</h1>
        <p>Enterprise Customer Attrition Monitoring, Deep Learning Architectures & Prescriptive Decision Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="kpi-card"><div class="kpi-lbl">Total Customer Profiles</div><div class="kpi-val">7,043</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="kpi-card"><div class="kpi-lbl">Champion ROC-AUC Score</div><div class="kpi-val">0.8181</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="kpi-card"><div class="kpi-lbl">Champion Recall (Sensitivity)</div><div class="kpi-val">81.77%</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="kpi-card"><div class="kpi-lbl">Est. Revenue at Risk</div><div class="kpi-val">$2.84M</div></div>', unsafe_allow_html=True)

    st.markdown("### 📋 Executive Project Abstract")
    st.write("""
    In modern recurring-revenue and telecommunications platforms, customer attrition directly impacts EBITDA and market valuation. 
    This platform delivers an **End-to-End Enterprise Predictive Analytics System** developed for Master's thesis standards. 
    It incorporates **Stratified Feature Engineering**, **Cost-Sensitive Multi-Model Benchmarking**, **Explainable AI (XAI)**, 
    and an interactive **Counterfactual What-If Simulator** that empowers retention managers to test strategic interventions before execution.
    """)

    st.markdown("### 🏗️ Academic Pipeline Architecture")
    c_arch1, c_arch2 = st.columns([1, 1])
    with c_arch1:
        st.info("""
        **1. Ingestion & Preprocessing**:
        - Automated imputation of non-numeric billing strings.
        - Dynamic computation of **ChargeRatio** and **Estimated Customer Lifetime Value (CLV)**.
        - Robust categorical encoding across contract types, payment channels, and service combinations.
        
        **2. Multi-Model Inference Core**:
        - **Linear Baseline**: Regularized Logistic Regression.
        - **Ensemble Core**: Balanced Random Forest & Tuned XGBoost.
        - **Deep Neural Network**: 3-layer MLP with Adam adaptive optimizer.
        """)
    with c_arch2:
        st.info("""
        **3. Explainability (XAI)**:
        - Local feature attribution identifying individual risk amplifiers vs. loyalty anchors.
        - Global feature importance verification via Gini index and impurity reduction.
        
        **4. Prescriptive Decision Support**:
        - Automated synthesis of tailored retention offers.
        - Real-time simulation of revenue preserved through counterfactual adjustments.
        """)

    chart_bench = os.path.join(MODELS_DIR, "model_benchmark_comparison.png")
    if os.path.exists(chart_bench):
        st.markdown("### 📈 Multi-Algorithm Benchmark Comparison")
        st.image(chart_bench, use_container_width=True)

# ======================================================================================
# VIEW 2: REAL-TIME CUSTOMER RISK PROFILER & XAI
# ======================================================================================
elif page == "🎯 Customer Risk Profiler & XAI":
    st.markdown("""
    <div class="header-box">
        <h1>Real-Time Customer Risk Profiling & Explainability (XAI)</h1>
        <p>Interactive single-account diagnostic scoring, local feature attributions, and personalized retention actions.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("customer_input_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 👤 Account Demographics")
            cust_id = st.text_input("Subscriber Identifier", value="CUST-7749-DEMO")
            gender = st.selectbox("Customer Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen (Age 65+)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Has Partner / Spouse", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])
            tenure = st.slider("Account Tenure (Months)", min_value=1, max_value=72, value=6)

        with col2:
            st.markdown("#### 🌐 Service Ecosystem")
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            mult_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet = st.selectbox("Internet Service Architecture", ["Fiber optic", "DSL", "No"])
            online_sec = st.selectbox("Online Cyber-Security", ["Yes", "No", "No internet service"], index=1)
            online_bak = st.selectbox("Cloud Backup", ["Yes", "No", "No internet service"], index=1)
            dev_prot = st.selectbox("Device Protection Warranty", ["Yes", "No", "No internet service"], index=1)
            tech_sup = st.selectbox("Dedicated Technical Support", ["Yes", "No", "No internet service"], index=1)
            stream_tv = st.selectbox("Streaming TV Service", ["Yes", "No", "No internet service"])
            stream_mov = st.selectbox("Streaming Movies Service", ["Yes", "No", "No internet service"])

        with col3:
            st.markdown("#### 💳 Financials & Contract")
            contract = st.selectbox("Contract Terms", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Invoicing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )
            monthly_charges = st.number_input("Monthly Subscription ($)", min_value=18.0, max_value=130.0, value=89.0, step=1.0)
            total_charges = st.number_input("Cumulative Total Billed ($)", min_value=18.0, max_value=9000.0, value=float(monthly_charges * tenure), step=10.0)

        eval_btn = st.form_submit_button("⚡ Run Predictive Diagnostic & Explainability Analysis", use_container_width=True)

    if eval_btn:
        start_t = time.time()
        from src.data_pipeline import clean_and_feature_engineer
        from src.explainability import explain_prediction, generate_prescriptive_strategy
        from src.database import log_prediction

        cust_dict = {
            'customerID': cust_id,
            'gender': gender,
            'SeniorCitizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone,
            'MultipleLines': mult_lines,
            'InternetService': internet,
            'OnlineSecurity': online_sec,
            'OnlineBackup': online_bak,
            'DeviceProtection': dev_prot,
            'TechSupport': tech_sup,
            'StreamingTV': stream_tv,
            'StreamingMovies': stream_mov,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }

        explanation = explain_prediction(cust_dict, active_model, scaler, features)
        prob = explanation["churn_probability"]
        latency_ms = (time.time() - start_t) * 1000

        tier = "Critical Risk" if prob >= 0.65 else ("Moderate Risk" if prob >= 0.35 else "Low Risk / Loyal")
        clv = round(monthly_charges * (tenure + 24) * 0.75, 2)
        rev_risk = round(monthly_charges * 12, 2)

        # Log into SQLite Database
        log_prediction(cust_id, active_model_name, prob, tier, clv, rev_risk, latency_ms)

        st.markdown("---")
        st.markdown("### 📊 Predictive Diagnosis Results")
        res1, res2 = st.columns([1.1, 1.9])

        with res1:
            # Interactive Plotly Gauge Meter
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Probability (%)", 'font': {'size': 18, 'color': '#1E3A8A'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#DC2626" if prob >= 0.65 else ("#D97706" if prob >= 0.35 else "#059669")},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#CBD5E1",
                    'steps': [
                        {'range': [0, 35], 'color': '#DCFCE7'},
                        {'range': [35, 65], 'color': '#FEF3C7'},
                        {'range': [65, 100], 'color': '#FEE2E2'}
                    ]
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            if prob >= 0.65:
                st.markdown('<div class="badge-critical">🚨 CRITICAL ATTRITION RISK</div>', unsafe_allow_html=True)
            elif prob >= 0.35:
                st.markdown('<div class="badge-moderate">⚠️ MODERATE RETENTION RISK</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-loyal">✅ SECURE / HIGH LOYALTY ACCOUNT</div>', unsafe_allow_html=True)

            st.write(f"**Customer Lifetime Value (CLV)**: `${clv:,.2f}`")
            st.write(f"**Annual Revenue Exposure**: `${rev_risk:,.2f}`")
            st.caption(f"⚡ Inferred in `{latency_ms:.2f} ms` via SQLite transaction logging.")

        with res2:
            st.markdown("#### 🔍 Explainable AI (XAI) Attribution Breakdown")
            
            # Interactive Plotly Feature Attribution Bar Chart
            drivers = explanation["top_churn_drivers"]
            anchors = explanation["top_loyalty_anchors"]

            plot_items = []
            for d in drivers[:4]:
                plot_items.append({"Feature": d['feature'], "Impact": d['impact'], "Type": "Risk Amplifier (+)"})
            for a in anchors[:4]:
                plot_items.append({"Feature": a['feature'], "Impact": -a['impact'], "Type": "Loyalty Anchor (-)"})

            if plot_items:
                plot_df = pd.DataFrame(plot_items)
                fig_xai = px.bar(
                    plot_df, x="Impact", y="Feature", color="Type", orientation="h",
                    color_discrete_map={"Risk Amplifier (+)": "#DC2626", "Loyalty Anchor (-)": "#059669"},
                    title="Local Feature Contribution Scores"
                )
                fig_xai.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10))
                st.plotly_chart(fig_xai, use_container_width=True)

        st.markdown("---")
        st.markdown("### 💡 Prescriptive Retention Recommendations")
        prescriptions = generate_prescriptive_strategy(cust_dict, prob)
        for p in prescriptions:
            p_badge = "🔴 URGENT" if p['priority'] == "URGENT" else ("🟠 HIGH" if p['priority'] == "HIGH" else "🟡 MEDIUM")
            st.info(f"**[{p_badge}] {p['action']}**\n\n{p['detail']}")

# ======================================================================================
# VIEW 3: "WHAT-IF" POLICY SIMULATOR
# ======================================================================================
elif page == "🔄 'What-If' Policy Simulator":
    st.markdown("""
    <div class="header-box">
        <h1>Counterfactual 'What-If' Policy Intervention Simulator</h1>
        <p>Simulate strategic retention interventions (contract extensions, support waivers, discounts) and measure live risk reduction.</p>
    </div>
    """, unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns([1, 1])
    with sim_col1:
        st.markdown("#### 1. Baseline At-Risk Customer Profile")
        sim_id = st.text_input("Subscriber Identifier", value="CUST-SIM-044")
        base_tenure = st.slider("Baseline Account Tenure (Months)", 1, 36, 6)
        base_monthly = st.slider("Baseline Monthly Fee ($)", 40.0, 120.0, 89.0)
        base_contract = st.selectbox("Baseline Contract", ["Month-to-month", "One year", "Two year"], index=0)
        base_internet = st.selectbox("Baseline Internet", ["Fiber optic", "DSL", "No"], index=0)
        base_tech_sup = st.selectbox("Baseline Tech Support", ["No", "Yes"], index=0)
        base_payment = st.selectbox("Baseline Payment Method", ["Electronic check", "Bank transfer (automatic)", "Credit card (automatic)"], index=0)

    with sim_col2:
        st.markdown("#### 2. Strategic Interventions to Simulate")
        new_contract = st.selectbox("Offer Upgraded Contract", ["Keep Baseline", "One year", "Two year"], index=1)
        new_tech_sup = st.selectbox("Provide Complimentary Dedicated Support", ["Keep Baseline", "Yes"], index=1)
        apply_discount = st.slider("Promotional Monthly Discount ($)", 0.0, 30.0, 15.0, step=5.0)
        new_payment = st.selectbox("Migrate to Automated Payment", ["Keep Baseline", "Credit card (automatic)", "Bank transfer (automatic)"], index=1)

    if st.button("🚀 Run Counterfactual Policy Simulation", use_container_width=True):
        from src.explainability import simulate_what_if_counterfactual
        from src.database import log_simulation

        base_input = {
            'customerID': sim_id,
            'gender': 'Female',
            'SeniorCitizen': 0,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': base_tenure,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': base_internet,
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': base_tech_sup,
            'StreamingTV': 'Yes',
            'StreamingMovies': 'Yes',
            'Contract': base_contract,
            'PaperlessBilling': 'Yes',
            'PaymentMethod': base_payment,
            'MonthlyCharges': base_monthly,
            'TotalCharges': base_monthly * base_tenure
        }

        interventions = {}
        if new_contract != "Keep Baseline":
            interventions['Contract'] = new_contract
        if new_tech_sup != "Keep Baseline":
            interventions['TechSupport'] = new_tech_sup
        if apply_discount > 0:
            interventions['MonthlyCharges'] = max(20.0, base_monthly - apply_discount)
        if new_payment != "Keep Baseline":
            interventions['PaymentMethod'] = new_payment

        sim_result = simulate_what_if_counterfactual(base_input, interventions, active_model, scaler)

        # Log into SQLite Database
        log_simulation(
            sim_id, active_model_name,
            sim_result['base_churn_prob'], sim_result['simulated_churn_prob'],
            sim_result['percentage_risk_reduction'], sim_result['annual_revenue_preserved'],
            interventions
        )

        st.markdown("---")
        st.markdown("### 🎯 Simulation Outcomes")

        kpi_s1, kpi_s2, kpi_s3, kpi_s4 = st.columns(4)
        with kpi_s1:
            st.metric("Baseline Churn Risk", f"{sim_result['base_churn_prob']*100:.1f}%")
        with kpi_s2:
            st.metric("Simulated New Risk", f"{sim_result['simulated_churn_prob']*100:.1f}%", delta=f"-{sim_result['absolute_reduction']*100:.1f}%", delta_color="inverse")
        with kpi_s3:
            st.metric("Risk Reduction Impact", f"{sim_result['percentage_risk_reduction']:.1f}%")
        with kpi_s4:
            st.metric("Est. Annual Revenue Preserved", f"${sim_result['annual_revenue_preserved']:,.2f}")

        # Plotly comparison bar chart
        fig_sim = go.Figure(data=[
            go.Bar(name='Baseline Risk', x=['Attrition Probability'], y=[sim_result['base_churn_prob']*100], marker_color='#DC2626'),
            go.Bar(name='After Strategic Intervention', x=['Attrition Probability'], y=[sim_result['simulated_churn_prob']*100], marker_color='#059669')
        ])
        fig_sim.update_layout(barmode='group', height=260, yaxis=dict(title='Churn Risk (%)', range=[0, 100]))
        st.plotly_chart(fig_sim, use_container_width=True)

        st.success(f"""
        **Executive Conclusion:** Implementing these retention interventions successfully lowers customer churn probability 
        from **{sim_result['base_churn_prob']*100:.1f}%** down to **{sim_result['simulated_churn_prob']*100:.1f}%**, 
        yielding a **{sim_result['percentage_risk_reduction']:.1f}% relative reduction in attrition risk** 
        and safeguarding approximately **${sim_result['annual_revenue_preserved']:,.2f}** in annual recurring subscription revenue.
        """)

# ======================================================================================
# VIEW 4: ENTERPRISE BATCH AUDITING
# ======================================================================================
elif page == "📂 Enterprise Batch Auditing":
    st.markdown("""
    <div class="header-box">
        <h1>Enterprise Batch Churn Auditing & Risk Scoring</h1>
        <p>High-throughput automated scoring of enterprise subscriber rosters with financial risk categorization.</p>
    </div>
    """, unsafe_allow_html=True)

    up_file = st.file_uploader("Upload Customer Dataset (CSV)", type=["csv"])

    if up_file is not None:
        batch_df = pd.read_csv(up_file)
        st.write(f"Loaded **{len(batch_df)}** subscriber records:")
        st.dataframe(batch_df.head(5))

        if st.button("🚀 Process Batch Subscriber Roster"):
            from src.data_pipeline import clean_and_feature_engineer
            from src.config import ALL_FEATURE_COLUMNS

            with st.spinner("Processing records through inference pipeline..."):
                proc_df = clean_and_feature_engineer(batch_df)[ALL_FEATURE_COLUMNS]
                scaled_df = scaler.transform(proc_df)
                probs = active_model.predict_proba(scaled_df)[:, 1]

                out_df = batch_df.copy()
                out_df['Churn_Probability_%'] = np.round(probs * 100, 2)
                out_df['Risk_Tier'] = pd.cut(
                    probs, bins=[-1, 0.35, 0.65, 1.0],
                    labels=['Low Risk / Loyal', 'Moderate Risk', 'Critical Risk']
                )

                crit_cnt = (out_df['Risk_Tier'] == 'Critical Risk').sum()
                mod_cnt = (out_df['Risk_Tier'] == 'Moderate Risk').sum()
                low_cnt = (out_df['Risk_Tier'] == 'Low Risk / Loyal').sum()

                col_b1, col_b2, col_b3 = st.columns(3)
                with col_b1:
                    st.metric("Critical Risk Accounts", f"{crit_cnt} ({crit_cnt/len(out_df)*100:.1f}%)")
                with col_b2:
                    st.metric("Moderate Risk Accounts", f"{mod_cnt} ({mod_cnt/len(out_df)*100:.1f}%)")
                with col_b3:
                    st.metric("Low Risk / Loyal Accounts", f"{low_cnt} ({low_cnt/len(out_df)*100:.1f}%)")

                st.dataframe(out_df[['customerID', 'tenure', 'Contract', 'MonthlyCharges', 'Churn_Probability_%', 'Risk_Tier']].head(25))

                csv_data = out_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Scored Enterprise Risk Audit (CSV)",
                    data=csv_data,
                    file_name="enterprise_churn_audit_scored.csv",
                    mime="text/csv"
                )
    else:
        st.info("💡 You can upload your own customer roster CSV or click below to audit the standard benchmark dataset (7,043 accounts).")
        sample_path = os.path.join(DATA_DIR, "customer_churn_data.csv")
        if os.path.exists(sample_path) and st.button("Load & Audit Benchmark Dataset (7,043 Accounts)"):
            sample_df = pd.read_csv(sample_path)
            from src.data_pipeline import clean_and_feature_engineer
            from src.config import ALL_FEATURE_COLUMNS

            proc_df = clean_and_feature_engineer(sample_df)[ALL_FEATURE_COLUMNS]
            scaled_df = scaler.transform(proc_df)
            probs = active_model.predict_proba(scaled_df)[:, 1]

            out_df = sample_df.copy()
            out_df['Churn_Probability_%'] = np.round(probs * 100, 2)
            out_df['Risk_Tier'] = pd.cut(
                probs, bins=[-1, 0.35, 0.65, 1.0],
                labels=['Low Risk / Loyal', 'Moderate Risk', 'Critical Risk']
            )

            crit_cnt = (out_df['Risk_Tier'] == 'Critical Risk').sum()
            mod_cnt = (out_df['Risk_Tier'] == 'Moderate Risk').sum()
            low_cnt = (out_df['Risk_Tier'] == 'Low Risk / Loyal').sum()

            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                st.metric("Critical Risk Accounts", f"{crit_cnt} ({crit_cnt/len(out_df)*100:.1f}%)")
            with col_b2:
                st.metric("Moderate Risk Accounts", f"{mod_cnt} ({mod_cnt/len(out_df)*100:.1f}%)")
            with col_b3:
                st.metric("Low Risk / Loyal Accounts", f"{low_cnt} ({low_cnt/len(out_df)*100:.1f}%)")

            st.dataframe(out_df[['customerID', 'tenure', 'Contract', 'MonthlyCharges', 'Churn_Probability_%', 'Risk_Tier']].head(25))

# ======================================================================================
# VIEW 5: MODEL BENCHMARKS & STATISTICAL TESTS
# ======================================================================================
elif page == "📊 Model Benchmarks & Statistics":
    st.markdown("""
    <div class="header-box">
        <h1>Academic Model Benchmarking & Statistical Inference Suite</h1>
        <p>Comprehensive comparative performance metrics, loss curves, and Pearson's Chi-Square / Welch's t-Test findings.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🏆 Multi-Model Benchmark", "🔬 Statistical Hypothesis Tests", "📈 Convergence & Diagnostic Charts"])

    with tab1:
        if summary is not None:
            res = summary["results"]
            table_rows = []
            for k, v in res.items():
                table_rows.append({
                    "Algorithm Architecture": v["name"],
                    "Accuracy (%)": f"{v['accuracy']*100:.2f}%",
                    "ROC-AUC Score": f"{v['auc']:.4f}",
                    "Recall (Sensitivity)": f"{v['recall']*100:.2f}%",
                    "Precision (%)": f"{v['precision']*100:.2f}%",
                    "F1-Score (%)": f"{v['f1']*100:.2f}%",
                    "Brier Calibration Loss": f"{v['brier_score']:.4f}"
                })
            st.table(pd.DataFrame(table_rows))
        else:
            st.warning("Model summary not loaded.")

    with tab2:
        st.markdown("### 1. Pearson's Chi-Square Test of Independence (Categorical Features vs. Churn)")
        st.write("Null Hypothesis ($H_0$): *Customer subscription choices are independent of churn behavior.*")
        if stat_summary and "chi2_results" in stat_summary:
            st.dataframe(pd.DataFrame(stat_summary["chi2_results"]))

        st.markdown("### 2. Two-Sample Welch's t-Test & Effect Sizes (Continuous Features)")
        st.write("Null Hypothesis ($H_0$): *Mean feature values are identical between Retained and Churned cohorts.*")
        if stat_summary and "ttest_results" in stat_summary:
            st.dataframe(pd.DataFrame(stat_summary["ttest_results"]))

    with tab3:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("#### 1. Multi-Model Performance Comparison")
            c1 = os.path.join(MODELS_DIR, "model_benchmark_comparison.png")
            if os.path.exists(c1):
                st.image(c1, use_container_width=True)

        with col_d2:
            st.markdown("#### 2. Deep Neural Network (MLP) Loss Curve")
            c3 = os.path.join(MODELS_DIR, "neural_network_learning_curve.png")
            if os.path.exists(c3):
                st.image(c3, use_container_width=True)

        col_d3, col_d4 = st.columns(2)
        with col_d3:
            st.markdown("#### 3. Top-10 Churn Predictors (XGBoost Gini)")
            c2 = os.path.join(MODELS_DIR, "top_feature_importance.png")
            if os.path.exists(c2):
                st.image(c2, use_container_width=True)

        with col_d4:
            st.markdown("#### 4. Confusion Matrices Grid Across Models")
            c4 = os.path.join(MODELS_DIR, "confusion_matrices_grid.png")
            if os.path.exists(c4):
                st.image(c4, use_container_width=True)

# ======================================================================================
# VIEW 6: DATABASE AUDIT & GOVERNANCE LOGS
# ======================================================================================
elif page == "🗄️ Database Audit & Governance Logs":
    st.markdown("""
    <div class="header-box">
        <h1>Enterprise Relational Database Audit & Governance Logs</h1>
        <p>ACID-compliant transaction logging from SQLite: persistent subscriber records, real-time prediction latencies, and simulation histories.</p>
    </div>
    """, unsafe_allow_html=True)

    from src.database import get_recent_audit_logs, get_recent_simulations

    db_tab1, db_tab2 = st.tabs(["📋 Real-Time Prediction Audit Trail", "🔄 Counterfactual Policy Simulation Logs"])

    with db_tab1:
        st.markdown("#### Recent Single-Customer Inference Logs (from `prediction_audit_logs`)")
        audit_df = get_recent_audit_logs(limit=30)
        if not audit_df.empty:
            st.dataframe(audit_df, use_container_width=True)
        else:
            st.info("No inference events recorded yet. Run a prediction in the 'Customer Risk Profiler' tab to generate logs!")

    with db_tab2:
        st.markdown("#### Logged What-If Counterfactual Experiments (from `simulated_interventions`)")
        sim_df = get_recent_simulations(limit=30)
        if not sim_df.empty:
            st.dataframe(sim_df, use_container_width=True)
        else:
            st.info("No simulations recorded yet. Run an experiment in the 'What-If Policy Simulator' tab to generate logs!")
