import streamlit as st
import joblib
import pandas as pd
import os

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Intelligent Analytics",
    page_icon="📊",
    layout="wide"
)

# ─── Custom CSS (Slider labels fix + Premium UI) ────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* ── SLIDER LABEL FIX — yahi main issue tha ── */
label[data-testid="stWidgetLabel"] p,
label[data-testid="stWidgetLabel"],
.stSlider label,
div[data-baseweb="slider"] ~ div label,
[data-testid="stSlider"] label {
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}

/* Selectbox label */
.stSelectbox label,
.stSelectbox label p {
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}

/* ── Slider Track & Thumb ── */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: #6c63ff !important;
    border: 2px solid #fff !important;
    width: 18px !important;
    height: 18px !important;
}
.stSlider [data-baseweb="slider"] div[class*="Track"] {
    background: rgba(108, 99, 255, 0.35) !important;
}
.stSlider [data-baseweb="slider"] div[class*="Track"]:first-child {
    background: #6c63ff !important;
}

/* ── Slider value number ── */
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"],
[data-testid="stSlider"] div {
    color: #a0aec0 !important;
}

/* ── Cards / Containers ── */
div[data-testid="stVerticalBlock"] > div {
    border-radius: 12px;
}

/* ── Section Headers ── */
h1, h2, h3, h4 {
    color: #ffffff !important;
}

/* ── Markdown text ── */
.stMarkdown p {
    color: #cbd5e0;
}

/* ── Hero Header ── */
.hero-box {
    background: linear-gradient(90deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid rgba(108, 99, 255, 0.4);
    border-left: 5px solid #6c63ff;
    border-radius: 16px;
    padding: 30px 36px;
    margin-bottom: 30px;
    box-shadow: 0 8px 32px rgba(108, 99, 255, 0.15);
}
.hero-box h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff !important;
    margin-bottom: 8px;
}
.hero-box p {
    color: #94a3b8;
    font-size: 1rem;
    margin: 0;
}

/* ── Section Card ── */
.section-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
}
.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #c4b5fd !important;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Predict Button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6c63ff, #a855f7) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 48px !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 20px rgba(108,99,255,0.4) !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(108,99,255,0.6) !important;
}

/* ── Metric Cards ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(108,99,255,0.3) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(108,99,255,0.4) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
}

/* ── Alert boxes ── */
.stAlert {
    border-radius: 12px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95) !important;
    border-right: 1px solid rgba(108,99,255,0.3) !important;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(255,255,255,0.1) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Model ────────────────────────────────────────────────────────────────
model_path = os.path.join(os.path.dirname(__file__), "employee_attrition_model.pkl")
model = joblib.load(model_path)

features = [
    'Age', 'MonthlyIncome', 'DistanceFromHome',
    'YearsAtCompany', 'TotalWorkingYears', 'OverTime',
    'JobSatisfaction', 'EnvironmentSatisfaction',
    'JobInvolvement', 'WorkLifeBalance'
]

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖")
    st.title("HR Analytics")
    st.write("Employee Attrition Prediction Dashboard")
    st.markdown("---")
    st.markdown("### Machine Learning Project")
    st.info("""
    **Algorithm:** Random Forest Classifier  
    **Framework:** Streamlit  
    **Library:** Scikit-Learn • Pandas
    """)
    st.markdown("---")
    st.markdown("**Features Used:**")
    for f in ["Age", "Monthly Income", "Distance From Home", "Years At Company",
              "Total Working Years", "OverTime", "Job Satisfaction",
              "Environment Satisfaction", "Job Involvement", "Work Life Balance"]:
        st.markdown(f"• {f}")

# ─── HERO HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
    <h1>📊 HR Intelligent Analytics</h1>
    <p>Predict employee attrition risks in real-time using advanced machine learning models</p>
</div>
""", unsafe_allow_html=True)

# ─── INPUT SECTION ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Personal & Financial Attributes</div>', unsafe_allow_html=True)
    age = st.slider("Age", 18, 60, 30)
    monthly_income = st.slider("Monthly Income (₹)", 1000, 50000, 5000, step=500)
    distance = st.slider("Distance From Home (km)", 1, 30, 5)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💼 Experience & Contractual Status</div>', unsafe_allow_html=True)
    years_at_company = st.slider("Years At Company", 0, 40, 5)
    total_working_years = st.slider("Total Working Years", 0, 40, 10)
    overtime = st.selectbox("OverTime", ["No", "Yes"])
    st.markdown('</div>', unsafe_allow_html=True)

# ─── SURVEY SCORES ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">⭐ Survey Scores (1 = Low → 4 = High)</div>', unsafe_allow_html=True)

score_col1, score_col2 = st.columns(2, gap="large")
with score_col1:
    job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
with score_col2:
    job_involvement = st.slider("Job Involvement", 1, 4, 3)
    work_life_balance = st.slider("Work Life Balance", 1, 4, 3)

st.markdown('</div>', unsafe_allow_html=True)

# ─── PREDICT BUTTON ────────────────────────────────────────────────────────────
overtime_numeric = 1 if overtime == "Yes" else 0
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Predict Attrition Risk", type="primary"):

    input_data = pd.DataFrame([[
        age, monthly_income, distance,
        years_at_company, total_working_years, overtime_numeric,
        job_satisfaction, environment_satisfaction,
        job_involvement, work_life_balance
    ]], columns=features)

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    confidence = max(probability[0]) * 100

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction[0] == 1:
        st.error(f"🔴 **High Attrition Risk** — Employee is likely to Leave ({confidence:.1f}% confidence)")
    else:
        st.success(f"🟢 **Low Attrition Risk** — Employee is likely to Stay ({confidence:.1f}% confidence)")

    st.markdown("---")
    st.markdown("### 📈 Prediction Summary")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Age", f"{age} yrs")
    with c2:
        st.metric("Monthly Income", f"₹{monthly_income:,}")
    with c3:
        st.metric("Company Experience", f"{years_at_company} yrs")
    with c4:
        st.metric("OverTime", overtime)

# ─── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<br><br>
<p style='color: #4a5568; font-size: 12px; text-align: center;'>
HR Intelligent Analytics | Powered by Random Forest ML | Built with Streamlit
</p>
""", unsafe_allow_html=True)