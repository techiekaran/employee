import streamlit as st 
import joblib
import pandas as pd

# 1. Page Configuration (Browser tab settings)
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="📊",
    layout="wide"  # Isse margins wide ho jayengi aur clean look aayega
)

# Load trained model
model = joblib.load("employee_attrition_model.pkl")

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
    # Number input aur uske turant baad slider (jaisa image mein experience ke liye tha)
    age = st.number_input("Age", 18, 60, 30)
    age_slider = st.slider("Adjust Age", 18, 60, int(age), key="age_s")
    
    monthly_income = st.number_input("Monthly Income (₹)", 1000, 50000, 5000)
    income_slider = st.slider("Adjust Monthly Income", 1000, 50000, int(monthly_income), key="income_s")
    
    distance = st.number_input("Distance From Home (km)", 1, 30, 5)
    distance_slider = st.slider("Adjust Distance", 1, 30, int(distance), key="distance_s")

with col_right:
    st.markdown("#### Experience & Satisfaction")
    years_at_company = st.number_input("Years At Company", 0, 40, 5)
    yac_slider = st.slider("Adjust Years at Company", 0, 40, int(years_at_company), key="yac_s")
    
    total_working_years = st.number_input("Total Working Years", 0, 40, 10)
    twy_slider = st.slider("Adjust Total Working Years", 0, 40, int(total_working_years), key="twy_s")
    
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
        age_slider,               # Slider values ko model ko pass kar rahe hain kyuki wo sync hain
        income_slider,
        distance_slider,
        yac_slider,
        twy_slider,
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
        st.markdown("<p style='color: gray; font-size: 16px; margin-bottom: 2px;'>Age</p>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='margin-top: 0px;'>{age_slider}</h2>", unsafe_allow_html=True)
        
    with metric_col2:
        st.markdown("<p style='color: gray; font-size: 16px; margin-bottom: 2px;'>Monthly Income</p>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='margin-top: 0px;'>₹{income_slider:,}</h2>", unsafe_allow_html=True)
        
    with metric_col3:
        st.markdown("<p style='color: gray; font-size: 16px; margin-bottom: 2px;'>Experience (Company)</p>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='margin-top: 0px;'>{yac_slider} Years</h2>", unsafe_allow_html=True)

# Page footer
st.markdown("<br><br><p style='color: gray; font-size: 12px; text-align: center;'>HR Analytics Dashboard | Streamlit - Machine Learning</p>", unsafe_allow_html=True)