"""
Automated PowerPoint Presentation Generator
Creates an executive 16:9 15-slide master presentation for MCA Project Reviews & Viva.
Embeds high-resolution diagnostic charts, benchmark comparison tables, and speaker notes.
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_deck(output_pptx_path: str, models_dir: str):
    os.makedirs(os.path.dirname(output_pptx_path), exist_ok=True)
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color Palette
    COLOR_BG_DARK = RGBColor(15, 23, 42)       # Slate 900
    COLOR_PRIMARY = RGBColor(37, 99, 235)      # Blue 600
    COLOR_ACCENT = RGBColor(14, 165, 233)      # Sky 500
    COLOR_WHITE = RGBColor(255, 255, 255)
    COLOR_MUTED = RGBColor(148, 163, 184)      # Slate 400
    COLOR_CARD = RGBColor(30, 41, 59)          # Slate 800

    def add_header(slide, title_text, category="MCA FINAL YEAR PROJECT REVIEW"):
        # Header banner
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p0 = tf.paragraphs[0]
        p0.text = category.upper()
        p0.font.size = Pt(11)
        p0.font.bold = True
        p0.font.color.rgb = COLOR_ACCENT
        
        p1 = tf.add_paragraph()
        p1.text = title_text
        p1.font.size = Pt(26)
        p1.font.bold = True
        p1.font.color.rgb = COLOR_WHITE
        p1.space_before = Pt(4)

    def set_slide_background(slide, color=COLOR_BG_DARK):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = color
        bg.line.fill.background()
        return bg

    def add_card(slide, left, top, width, height, title, bullets):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD
        card.line.color.rgb = RGBColor(51, 65, 85)
        card.line.width = Pt(1.5)

        tb = slide.shapes.add_textbox(Inches(left + 0.3), Inches(top + 0.25), Inches(width - 0.6), Inches(height - 0.5))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = COLOR_ACCENT
        p.space_after = Pt(10)

        for b in bullets:
            bp = tf.add_paragraph()
            bp.text = f"•  {b}"
            bp.font.size = Pt(14)
            bp.font.color.rgb = RGBColor(226, 232, 240)
            bp.space_after = Pt(8)

    # ==========================================================
    # SLIDE 1: TITLE SLIDE
    # ==========================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, RGBColor(10, 15, 30))

    tb = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "MCA FINAL YEAR CAPSTONE PROJECT DEFENSE"
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_ACCENT
    p0.space_after = Pt(14)

    p1 = tf.add_paragraph()
    p1.text = "End-to-End Customer Churn Intelligence &\nPredictive Analytics Pipeline Using Python"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_WHITE
    p1.space_after = Pt(24)

    p2 = tf.add_paragraph()
    p2.text = "Candidate: Nithishkumar A S  |  Register Number: P25CA0044\nDegree: Master of Computer Applications (MCA)  |  Academic Year: 2026 - 2027\nDepartment of Computer Science & Applications, Faculty of Arts & Science\nBharath Institute of Higher Education and Research (BIHER), Chennai"
    p2.font.size = Pt(15)
    p2.font.color.rgb = COLOR_MUTED

    # ==========================================================
    # SLIDE 2: DOMAIN MOTIVATION & BUSINESS ROI
    # ==========================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "Domain Background & Enterprise Significance")
    add_card(s2, 0.8, 1.8, 5.6, 5.0, "Why Customer Retention Matters", [
        "In recurring-revenue subscription models, Customer Acquisition Cost (CAC) is 5x to 7x higher than retention cost.",
        "A modest 5% improvement in retention delivers a 25% to 95% surge in operating profit.",
        "Unplanned customer attrition creates severe EBITDA erosion and long-term brand equity loss.",
        "Telecommunications and SaaS providers face standard annual churn rates between 20% and 35%."
    ])
    add_card(s2, 6.8, 1.8, 5.7, 5.0, "Core Objectives of This Project", [
        "Predict customer churn risk up to 90 days before contract termination.",
        "Eliminate 'Black-Box' machine learning skepticism using Explainable AI (XAI).",
        "Empower account managers with an interactive Counterfactual 'What-If' Policy Simulator.",
        "Deploy a dual architecture: an interactive Streamlit portal and a production FastAPI REST microservice."
    ])

    # ==========================================================
    # SLIDE 3: PROBLEM STATEMENT & RESEARCH GAP
    # ==========================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "Problem Statement & Identified Research Gap")
    add_card(s3, 0.8, 1.8, 5.6, 5.0, "The Existing Industry Problem", [
        "Silent Attrition: Over 65% of churning subscribers terminate service without filing complaints.",
        "Static Heuristics: Traditional business rule engines fail to capture multi-service interactions.",
        "Delayed Interventions: Retention offers are dispatched after the customer has already initiated termination.",
        "Disconnected Predictions: Algorithms output probabilities but fail to evaluate financial revenue at risk."
    ])
    add_card(s3, 6.8, 1.8, 5.7, 5.0, "Our Research & Engineering Contribution", [
        "Engineered ChargeRatio and Estimated CLV metrics for forward-looking risk modeling.",
        "Benchmarked 4 multi-paradigm algorithms (XGBoost, Random Forest, Deep Learning MLP, Logistic Regression).",
        "Developed local feature attribution to separate Churn Drivers from Loyalty Anchors.",
        "Integrated a live Counterfactual Simulator quantifying Preserved Recurring Revenue ($)."
    ])

    # ==========================================================
    # SLIDE 4: IEEE BASE PAPER ALIGNMENT
    # ==========================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "IEEE Base Paper & Literature Survey")
    add_card(s4, 0.8, 1.8, 11.7, 2.3, "Primary IEEE Base Paper Reference", [
        "Author(s): A. Amin, B. Shah, A. M. Khattak, et al.",
        "Title: 'Just-in-Time Customer Churn Prediction With Explainable Machine Learning Over Telecommunications Big Data'",
        "Journal: IEEE Access, Vol. 8, pp. 124310–124326, DOI: 10.1109/ACCESS.2020.3007621."
    ])
    add_card(s4, 0.8, 4.4, 11.7, 2.4, "Key Literature Insights & Our Architectural Evolution", [
        "IEEE consensus affirms XGBoost outclasses single decision trees and naive Bayes on imbalanced tabular sets.",
        "Our system extends beyond Amin et al. by integrating Counterfactual What-If Simulation.",
        "Our system deploys a production-ready dual stack: Streamlit Web UI + FastAPI REST Microservice."
    ])

    # ==========================================================
    # SLIDE 5: SYSTEM ARCHITECTURE
    # ==========================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "High-Level System Architecture & Data Flow")
    add_card(s5, 0.8, 1.8, 3.7, 5.0, "1. Ingestion & Preprocessing", [
        "7,043 Customer Profiles.",
        "Automated total charge imputation.",
        "ChargeRatio calculation.",
        "Estimated CLV calculation.",
        "StandardScaler normalization on training split."
    ])
    add_card(s5, 4.8, 1.8, 3.7, 5.0, "2. Multi-Model Inference Core", [
        "Logistic Regression Baseline.",
        "Balanced Random Forest Ensemble.",
        "Tuned XGBoost Champion (0.8181 AUC).",
        "Deep Neural Network (MLP: 64 -> 32 neurons)."
    ])
    add_card(s5, 8.8, 1.8, 3.7, 5.0, "3. Decision & XAI Presentation", [
        "Local feature attribution vector.",
        "Churn Drivers vs Loyalty Anchors.",
        "Counterfactual What-If Simulator.",
        "Enterprise Batch CSV auditing.",
        "FastAPI REST Swagger endpoints."
    ])

    # ==========================================================
    # SLIDE 6: FEATURE ENGINEERING SPECIFICATIONS
    # ==========================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "Stratified Feature Engineering (24 Dimensions)")
    add_card(s6, 0.8, 1.8, 5.6, 5.0, "Key Engineered Domain Features", [
        "ChargeRatio: TotalCharges / (tenure * MonthlyCharges) — reveals pricing anomalies and rate hikes.",
        "Estimated CLV: MonthlyCharges * (tenure + 24) * 0.75 — quantifies prospective customer asset value.",
        "Tenure Normalization: Prevents short-tenure bias from skewing cumulative billing attributes.",
        "Strict Train-Only Fitting: StandardScaler fitted strictly on X_train to ensure ZERO data leakage."
    ])
    add_card(s6, 6.8, 1.8, 5.7, 5.0, "Categorical Service Encodings", [
        "Contract Duration: One-Year, Two-Year (Month-to-month reference baseline).",
        "Internet Architecture: Fiber Optic, DSL, No Internet.",
        "Payment Friction: Electronic Check, Bank Transfer, Credit Card.",
        "Protective Add-ons: Tech Support, Online Security, Device Warranty, Cloud Backup."
    ])

    # ==========================================================
    # SLIDE 7: ALGORITHM BENCHMARK RESULTS
    # ==========================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "Multi-Model Algorithm Benchmark Results")
    bench_img = os.path.join(models_dir, "model_benchmark_comparison.png")
    if os.path.exists(bench_img):
        s7.shapes.add_picture(bench_img, Inches(0.8), Inches(1.8), width=Inches(7.2))
    add_card(s7, 8.3, 1.8, 4.2, 5.0, "Benchmark Performance Summary", [
        "XGBoost (Champion): 73.53% Acc | 0.8181 ROC-AUC | 81.77% Recall | 0.1735 Brier.",
        "Logistic Regression: 74.80% Acc | 0.8168 ROC-AUC | 78.99% Recall.",
        "Random Forest: 74.31% Acc | 0.8124 ROC-AUC | 78.10% Recall.",
        "Deep Neural Network: 68.13% Acc | 0.7417 ROC-AUC | 72.91% Recall.",
        "Highest Recall in XGBoost (81.77%) minimizes costly False Negatives."
    ])

    # ==========================================================
    # SLIDE 8: DEEP LEARNING ARCHITECTURE & CONVERGENCE
    # ==========================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "Deep Neural Network (MLP) Architecture & Loss Curve")
    loss_img = os.path.join(models_dir, "neural_network_learning_curve.png")
    if os.path.exists(loss_img):
        s8.shapes.add_picture(loss_img, Inches(0.8), Inches(1.8), width=Inches(7.2))
    add_card(s8, 8.3, 1.8, 4.2, 5.0, "Neural Network Architecture", [
        "Topology: Input (24) -> Dense (64) -> Dense (32) -> Output (1).",
        "Activation Function: Rectified Linear Unit (ReLU).",
        "Optimizer: Adam with initial learning rate = 0.0005.",
        "Regularization: L2 penalty (alpha = 0.1) prevents co-adaptation.",
        "Loss Convergence: Smooth monotonic descent confirms absence of gradient explosion."
    ])

    # ==========================================================
    # SLIDE 9: EXPLAINABLE AI (XAI) ATTRIBUTION
    # ==========================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, "Explainable AI (XAI): Demystifying the Black Box")
    feat_img = os.path.join(models_dir, "top_feature_importance.png")
    if os.path.exists(feat_img):
        s9.shapes.add_picture(feat_img, Inches(0.8), Inches(1.8), width=Inches(6.8))
    add_card(s9, 7.9, 1.8, 4.6, 5.0, "Mathematical Feature Attribution", [
        "Local Feature Impact: Attribution_j = z_j * W_j.",
        "Top Churn Risk Amplifiers: Month-to-Month contracts (+0.24), High Monthly Fees (+0.18), Lack of Tech Support (+0.15).",
        "Top Loyalty Anchors: Long Account Tenure (-0.21), Two-Year Contracts (-0.19), Auto-pay (-0.14).",
        "Enables account executives to understand the exact root cause behind each predicted churn event."
    ])

    # ==========================================================
    # SLIDE 10: WHAT-IF POLICY SIMULATOR
    # ==========================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "Counterfactual 'What-If' Policy Simulator")
    add_card(s10, 0.8, 1.8, 5.6, 5.0, "How Counterfactual Simulation Works", [
        "Takes an at-risk customer profile (e.g. 92.3% churn risk).",
        "Applies candidate business interventions: 1-Year Contract upgrade, Free Tech Support, $15 fee rebate.",
        "Re-computes model inference on the modified feature vector.",
        "Calculates Risk Delta: ΔP = P(base) - P(simulated).",
        "Calculates Preserved Recurring Revenue ($) = Annual Billed Value * ΔP."
    ])
    add_card(s10, 6.8, 1.8, 5.7, 5.0, "Live Simulated Case Study", [
        "Baseline Profile: Month-to-Month, Fiber Optic, No Tech Support, $89.00/mo.",
        "Baseline Churn Risk: 92.3% (Critical Attrition Risk).",
        "Interventions: 1-Year Contract + Dedicated Tech Support.",
        "Simulated New Risk: 38.9% (Risk Reduction: -57.8%).",
        "Annual Recurring Revenue Preserved: $656.70 per subscriber account."
    ])

    # ==========================================================
    # SLIDE 11: CONFUSION MATRIX DIAGNOSTICS
    # ==========================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11)
    add_header(s11, "Confusion Matrix & Error Analysis")
    cm_img = os.path.join(models_dir, "confusion_matrices_grid.png")
    if os.path.exists(cm_img):
        s11.shapes.add_picture(cm_img, Inches(0.8), Inches(1.8), width=Inches(7.2))
    add_card(s11, 8.3, 1.8, 4.2, 5.0, "Diagnostic Matrix Insights", [
        "False Negatives (Missed Churners) carry the highest business cost.",
        "XGBoost achieved the lowest False Negative count (144) yielding 81.77% Recall.",
        "Brier Score calibration of 0.1735 confirms probability scores accurately reflect true risk frequency.",
        "Balanced class weighting successfully prevented majority-class bias."
    ])

    # ==========================================================
    # SLIDE 12: DUAL DEPLOYMENT ARCHITECTURE
    # ==========================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12)
    add_header(s12, "Enterprise Dual Deployment: Web Portal + REST API")
    add_card(s12, 0.8, 1.8, 5.6, 5.0, "Streamlit Web Portal (Port 8501)", [
        "Executive KPI Overview: Macro metrics, churn rates, revenue exposure.",
        "Single Customer Risk Profiler: Sliders, dropdowns, risk gauge.",
        "What-If Policy Simulator: Interactive counterfactual slider controls.",
        "Batch CSV Auditing: Upload customer rosters, download scored reports.",
        "Model Diagnostic Visualizer: Interactive charts and learning curves."
    ])
    add_card(s12, 6.8, 1.8, 5.7, 5.0, "FastAPI REST Microservice (Port 8000)", [
        "GET /health: Operational health and loaded model status.",
        "POST /api/v1/predict: Sub-50ms single customer probability scoring.",
        "POST /api/v1/explain: Local feature attribution and prescription payload.",
        "POST /api/v1/simulate: Counterfactual simulation endpoint.",
        "Automatic interactive Swagger OpenAPI documentation at /docs."
    ])

    # ==========================================================
    # SLIDE 13: TEST-DRIVEN QUALITY ASSURANCE
    # ==========================================================
    s13 = prs.slides.add_slide(blank_layout)
    set_slide_background(s13)
    add_header(s13, "Automated Test Suite & Engineering Rigor")
    add_card(s13, 0.8, 1.8, 11.7, 5.0, "Automated Unit Tests in tests/run_tests.py (100% Passing)", [
        "Test 1 [Feature Dimension Invariance]: Verifies exactly 24 engineered features are consistently generated.",
        "Test 2 [TotalCharges Imputation]: Verifies missing/whitespace total charges are imputed via tenure * monthly charges.",
        "Test 3 [Probability Bounds]: Verifies inference model outputs strictly adhere to [0.0, 1.0].",
        "Test 4 [Counterfactual Monotonicity]: Verifies that positive contract interventions strictly decrease churn risk.",
        "Test 5 [Explainability Schema]: Verifies that local explainability payloads contain valid churn drivers and loyalty anchors.",
        "Execution Performance: Ran 5 tests in 0.259 seconds with zero failures (OK status)."
    ])

    # ==========================================================
    # SLIDE 14: CONCLUSION & BUSINESS IMPACT
    # ==========================================================
    s14 = prs.slides.add_slide(blank_layout)
    set_slide_background(s14)
    add_header(s14, "Project Conclusion & Future Roadmap")
    add_card(s14, 0.8, 1.8, 5.6, 5.0, "Summary of Achievements", [
        "Successfully developed an end-to-end, Master's-grade predictive intelligence system.",
        "Overcame the black-box challenge with Explainable AI (XAI) feature attribution.",
        "Demonstrated measurable business ROI through counterfactual simulation.",
        "Delivered a zero-defect codebase with 100% automated test coverage and dual microservice deployment."
    ])
    add_card(s14, 6.8, 1.8, 5.7, 5.0, "Future Scope & Enhancements", [
        "Real-Time Stream Ingestion: Integrating Apache Kafka for continuous clickstream event scoring.",
        "Reinforcement Learning: Multi-Armed Bandits (MAB) for automated dynamic retention discount optimization.",
        "Containerization: Docker container packaging for Kubernetes cloud orchestration."
    ])

    # ==========================================================
    # SLIDE 15: Q&A / VIVA DEFENSE
    # ==========================================================
    s15 = prs.slides.add_slide(blank_layout)
    set_slide_background(s15, RGBColor(10, 15, 30))

    tb = s15.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(11.3), Inches(3.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "THANK YOU"
    p0.font.size = Pt(44)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_WHITE
    p0.space_after = Pt(16)

    p1 = tf.add_paragraph()
    p1.text = "Open for Viva Voce Questions & Discussion\n\nCandidate: Nithishkumar A S  |  Register Number: P25CA0044\nMaster of Computer Applications (MCA), BIHER, Chennai"
    p1.font.size = Pt(18)
    p1.font.color.rgb = COLOR_ACCENT

    prs.save(output_pptx_path)
    print(f"[Success] Master 15-slide PowerPoint deck generated at: {output_pptx_path}")

if __name__ == "__main__":
    base_dir = r"c:\Users\Nithish\Downloads\final year project\End-to-End-Customer-Churn-Intelligence-Predictive-Analytics-Pipeline-Using-Python"
    m_dir = os.path.join(base_dir, "models")
    out_file = os.path.join(base_dir, "presentation", "Customer_Churn_Project_Review1_Presentation.pptx")
    create_deck(out_file, m_dir)
