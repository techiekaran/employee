import streamlit as st
import joblib
import pandas as pd
import os

# 1. Page Configuration (Browser tab settings)
st.set_page_config(
    page_title="HR Analytics - Attrition Predictor",
    page_icon="👥",
    layout="wide"
)

# Custom CSS Injection for Modern Responsive UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Main body styling */
    .stApp {
        background-color: #F8FAFC;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Title styling */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Hero Header Panel */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
        border-left: 6px solid #0D9488;
        position: relative;
        overflow: hidden;
    }
    
    .hero-banner h1 {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        color: #F8FAFC !important;
        letter-spacing: -0.025em;
    }
    
    .hero-banner p {
        color: #94A3B8 !important;
        font-size: 1rem !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0 !important;
    }
    
    /* Target Streamlit's native st.container(border=True) cards */
    div[data-testid="stVerticalBlockBorder"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.02) !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Custom button behavior */
    .stButton>button {
        background: linear-gradient(135deg, #0D9488 0%, #0F766E 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        border-radius: 8px !important;
        box-shadow: 0 8px 12px -3px rgba(13, 148, 136, 0.25) !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        margin-top: 1rem;
        cursor: pointer;
    }
    
    .stButton>button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 12px 16px -3px rgba(13, 148, 136, 0.35) !important;
    }
    
    /* Custom Results Cards */
    .result-box {
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
    
    .result-stay {
        background-color: #F0FDF4;
        border: 1px solid #DCFCE7;
        border-left: 6px solid #16A34A;
    }
    
    .result-leave {
        background-color: #FEF2F2;
        border: 1px solid #FEE2E2;
        border-left: 6px solid #DC2626;
    }
    
    /* Dynamic Metric cards */
    .custom-metric {
        background: white;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
        transition: all 0.2s ease;
    }
    
    .custom-metric:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.04);
    }
    
    .custom-metric-label {
        font-size: 0.8rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    
    .custom-metric-value {
        font-size: 1.5rem;
        color: #0F172A;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Load trained model (path fix for Streamlit Cloud)
model_path = os.path.join(os.path.dirname(__file__), "employee_attrition_model.pkl")
model = joblib.load(model_path)

features = [
    'Age',
    'MonthlyIncome',
    'DistanceFromHome',
    'YearsAtCompany',
    'TotalWorkingYears',
    'OverTime',
    'JobSatisfaction',
    'EnvironmentSatisfaction',
    'JobInvolvement',
    'WorkLifeBalance'
]

# 2. SIDEBAR (Left Panel) - Branding & Project Details
with st.sidebar:
    st.markdown("<h2 style='color: #0F172A; font-weight: 700; margin-bottom: 0;'>👥 HR Portal</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B; font-size: 0.95rem; margin-top: 0;'>Control Center</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 📋 Predictive System")
    st.write("This engine uses machine learning to compute the probability of an employee seeking external opportunities based on their professional profile.")
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Engine Specs")
    st.info("""
    **Classifier:** Random Forest  
    **Version:** MLv1.2.0  
    **Data Scope:** Demographics, Engagement, Finance, & Career History.
    """)
    st.markdown("<br><p style='color: #94A3B8; font-size: 0.8rem; text-align: center;'>Engine Online • Secure Connection</p>", unsafe_allow_html=True)

# 3. HERO HEADER
st.markdown("""
<div class="hero-banner">
    <h1>HR Intelligent Analytics</h1>
    <p>Predict employee attrition risks in real-time using advanced machine learning models</p>
</div>
""", unsafe_allow_html=True)

# Layout: Split into two columns for inputs using native container cards
col_left, col_right = st.columns(2)

with col_left:
    with st.container(border=True):
        st.markdown('<h3 style="margin-top:0; color:#0F172A; font-size:1.15rem; font-weight:600;">📊 Personal & Financial Attributes</h3>', unsafe_allow_html=True)
        age = st.slider("Employee Age (Years)", 18, 60, 32)
        monthly_income = st.slider("Monthly Income (₹)", 1000, 50000, 6800, step=500)
        distance = st.slider("Distance From Home (km)", 1, 30, 8)

with col_right:
    with st.container(border=True):
        st.markdown('<h3 style="margin-top:0; color:#0F172A; font-size:1.15rem; font-weight:600;">💼 Experience & Contractual Status</h3>', unsafe_allow_html=True)
        years_at_company = st.slider("Years At Company", 0, 40, 4)
        total_working_years = st.slider("Total Working Years", 0, 40, 8)
        overtime = st.selectbox("Mandatory Overtime Status", ["No", "Yes"])

# Survey Scores in wide card
with st.container(border=True):
    st.markdown('<h3 style="margin-top:0; color:#0F172A; font-size:1.15rem; font-weight:600;">📋 Engagement & Satisfaction Scores</h3>', unsafe_allow_html=True)
    
    survey_col1, survey_col2, survey_col3, survey_col4 = st.columns(4)
    
    with survey_col1:
        job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
    with survey_col2:
        environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
    with survey_col3:
        job_involvement = st.slider("Job Involvement (1-4)", 1, 4, 3)
    with survey_col4:
        work_life_balance = st.slider("Work Life Balance (1-4)", 1, 4, 3)

# Convert OverTime to numeric
overtime_numeric = 1 if overtime == "Yes" else 0

# Predict triggers
if st.button("Evaluate Attrition Risk"):
    
    # Structure input
    input_data = pd.DataFrame([[
        age,
        monthly_income,
        distance,
        years_at_company,
        total_working_years,
        overtime_numeric,
        job_satisfaction,
        environment_satisfaction,
        job_involvement,
        work_life_balance
    ]], columns=features)
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    confidence = max(probability[0]) * 100
    
    # Custom Alert Box
    if prediction[0] == 1:
        st.markdown(f"""
        <div class="result-box result-leave">
            <h3 style='color: #991B1B; margin:0; font-weight: 700;'>🚨 High Attrition Risk Detected</h3>
            <p style='color: #7F1D1D; font-size: 1.05rem; margin-top: 0.5rem; margin-bottom: 0;'>
                The system predicts the employee is <b>likely to leave</b>.
                Confidence Score: <b>{confidence:.1f}%</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box result-stay">
            <h3 style='color: #166534; margin:0; font-weight: 700;'>✅ Stable Retention Status</h3>
            <p style='color: #14532D; font-size: 1.05rem; margin-top: 0.5rem; margin-bottom: 0;'>
                The system predicts the employee is <b>likely to stay</b> with the company.
                Confidence Score: <b>{confidence:.1f}%</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<h4 style='color: #0F172A; font-weight: 600; margin-bottom: 1rem;'>📈 Subject Evaluation Summary</h4>", unsafe_allow_html=True)
    
    # Metrics display grid
    metric_grid_col1, metric_grid_col2, metric_grid_col3, metric_grid_col4 = st.columns(4)
    
    with metric_grid_col1:
        st.markdown(f"""
        <div class="custom-metric">
            <div class="custom-metric-label">Employee Age</div>
            <div class="custom-metric-value">{age} Yrs</div>
        </div>
        """, unsafe_allow_html=True)
        
    with metric_grid_col2:
        st.markdown(f"""
        <div class="custom-metric">
            <div class="custom-metric-label">Monthly Salary</div>
            <div class="custom-metric-value">₹{monthly_income:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with metric_grid_col3:
        st.markdown(f"""
        <div class="custom-metric">
            <div class="custom-metric-label">Tenure (Company)</div>
            <div class="custom-metric-value">{years_at_company} Yrs</div>
        </div>
        """, unsafe_allow_html=True)
        
    with metric_grid_col4:
        st.markdown(f"""
        <div class="custom-metric">
            <div class="custom-metric-label">Overtime</div>
            <div class="custom-metric-value">{overtime}</div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br><p style='color: #94A3B8; font-size: 0.8rem; text-align: center;'>HR Intelligent Systems | Version MLv1.2.0 • Data-Secured</p>", unsafe_allow_html=True)