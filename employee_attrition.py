import streamlit as st
import joblib
import pandas as pd
import os

# 1. Page Configuration (Browser tab settings)
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="📊",
    layout="wide"  # Isse margins wide ho jayengi aur clean look aayega
)

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
    st.markdown("### 👤")
    st.title("HR Analytics")
    st.write("Employee Attrition Prediction Dashboard")

    st.markdown("---")

    st.markdown("### Machine Learning Project")
    # Yahan aap apna naam ya details change kar sakte hain
    st.info("""
    **Algorithm:** Random Forest Classifier  
    **Developed using:** Streamlit • Scikit-Learn • Pandas
    """)

# 3. MAIN PAGE (Right Panel)
st.markdown("## 📊 Input Parameters")

# Inputs ko grid structure dene ke liye 2 bade columns banate hain
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Personal & Financials")
    age = st.slider("Age", 18, 60, 30)
    monthly_income = st.slider("Monthly Income (₹)", 1000, 50000, 5000, step=500)
    distance = st.slider("Distance From Home (km)", 1, 30, 5)

with col_right:
    st.markdown("#### Experience & Satisfaction")
    years_at_company = st.slider("Years At Company", 0, 40, 5)
    total_working_years = st.slider("Total Working Years", 0, 40, 10)
    overtime = st.selectbox("OverTime", ["No", "Yes"])

# Satisfaction Sliders ko thoda clean dikhane ke liye side-by-side columns mein kiya hai
st.markdown("---")
st.markdown("#### Survey Scores (1 to 4)")
score_col1, score_col2 = st.columns(2)

with score_col1:
    job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)

with score_col2:
    job_involvement = st.slider("Job Involvement", 1, 4, 3)
    work_life_balance = st.slider("Work Life Balance", 1, 4, 3)

# Convert OverTime to numeric
overtime_numeric = 1 if overtime == "Yes" else 0

st.markdown("<br>", unsafe_allow_html=True)

# 4. PREDICTION LOGIC & SUMMARY LAYOUT
# Primary button blue color mein aayega
if st.button("Predict", type="primary"):

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

    # Prediction Alert Box (Green/Red)
    if prediction[0] == 1:
        st.error(f"🔴 Employee is likely to Leave ({confidence:.2f}% confidence)")
    else:
        st.success(f"🟢 Employee is likely to Stay ({confidence:.2f}% confidence)")

    st.markdown("---")

    # Prediction Summary Section (Horizontal metrics cards)
    st.markdown("### 📈 Prediction Summary")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric(label="Age", value=age)

    with metric_col2:
        st.metric(label="Monthly Income", value=f"₹{monthly_income:,}")

    with metric_col3:
        st.metric(label="Experience (Company)", value=f"{years_at_company} Years")

# Page footer
st.markdown("<br><br><p style='color: gray; font-size: 12px; text-align: center;'>HR Analytics Dashboard | Streamlit - Machine Learning</p>", unsafe_allow_html=True)