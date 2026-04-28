import streamlit as st
import pickle
import numpy as np
import time

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="CardioVision | Health Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Custom Dark Theme & UI CSS ---
st.markdown("""
    <style>
    /* Global Dark Theme */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161A25;
        border-right: 1px solid #2B3040;
    }
    
    /* Styled Cards/Divs for Information */
    .info-card {
        background-color: #1E2333;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #E63946;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Enhanced Primary Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #E63946 0%, #9B2226 100%);
        color: white;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        padding: 12px 24px;
        width: 100%;
        border: none;
        box-shadow: 0px 4px 10px rgba(230, 57, 70, 0.4);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0px 6px 15px rgba(230, 57, 70, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. Load Model and Scaler ---
# --- 3. Load Model and Scaler ---
@st.cache_resource
def load_models():
    try:
        model = pickle.load(open("heart_model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        return model, scaler
    except Exception as e:
        
        # This will force the actual system error onto your screen
        st.error(f"🚨 ACTUAL SYSTEM ERROR: {e}")
        st.stop() 
        return None, None
    

model, scaler = load_models()

# --- 4. Sidebar Panel & Logo ---
st.sidebar.markdown("<h1 style='color: #E63946; margin-top: 0; font-size: 36px;'>🫀 CardioVision</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Navigation Panel
menu = st.sidebar.radio("📌 Navigation Panel", ["🎛️ Interactive Predictor", "🔬 Deep Dive: Prevention"])
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Utkarsh Verma. Built with Streamlit.")

# --- 5. Main Content: Interactive Predictor ---
if menu == "🎛️ Interactive Predictor":
    
    st.markdown("""
        <div style='display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #161A25 0%, #1E2333 100%); padding: 30px; border-radius: 12px; margin-bottom: 30px; border: 1px solid #2B3040; box-shadow: 0 4px 6px rgba(0,0,0,0.2);'>
            <div style='font-size: 70px; margin-right: 25px; filter: drop-shadow(0px 4px 6px rgba(230,57,70,0.4));'>🩸</div>
            <div>
                <h1 style='color: #FAFAFA; margin: 0; padding-bottom: 5px;'>Diagnostic Risk Engine</h1>
                <p style='color: #A0AEC0; margin: 0; font-size: 18px;'>Translate raw clinical data into actionable health insights using machine learning.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 👤 Vitals")
            age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)
            sex_display = st.selectbox("Biological Sex", ["Female", "Male"], index=1)
            sex = 1 if sex_display == "Male" else 0
            trestbps = st.number_input("Resting BP (mm Hg)", min_value=50, max_value=250, value=120)
            chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)

        with col2:
            st.markdown("### 🩺 Symptoms")
            cp_options = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
            cp_display = st.selectbox("Chest Pain Type", list(cp_options.keys()))
            cp = cp_options[cp_display]
            fbs_display = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
            fbs = 1 if fbs_display == "Yes" else 0
            exang_display = st.selectbox("Exercise Angina", ["No", "Yes"])
            exang = 1 if exang_display == "Yes" else 0
            thalach = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)

        with col3:
            st.markdown("### 📈 ECG & Scans")
            restecg_options = {"Normal": 0, "ST-T Abnormality": 1, "LV Hypertrophy": 2}
            restecg_display = st.selectbox("Resting ECG", list(restecg_options.keys()))
            restecg = restecg_options[restecg_display]
            oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope_options = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
            slope_display = st.selectbox("ST Slope", list(slope_options.keys()))
            slope = slope_options[slope_display]
            ca = st.selectbox("Vessels Colored", [0, 1, 2, 3])
            thal_options = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
            thal_display = st.selectbox("Thalassemia", list(thal_options.keys()), index=1)
            thal = thal_options[thal_display]

    st.markdown("<br>", unsafe_allow_html=True)
    
    _, center_btn, _ = st.columns([1, 2, 1])
    with center_btn:
        if st.button("Run Diagnostic Algorithm"):
            if model and scaler:
                with st.spinner('Compiling data and running risk assessment...'):
                    time.sleep(1.5)
                    sample = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                        thalach, exang, oldpeak, slope, ca, thal]])
                    sample_scaled = scaler.transform(sample)
                    prediction = model.predict(sample_scaled)

                st.divider()
                if prediction[0] == 1:
                    st.error("### 🚨 Elevated Risk Profile Detected\nThe diagnostic model flags a high probability of cardiovascular complications. Please forward these insights to a medical professional immediately.")
                else:
                    st.success("### ✅ Optimal Health Profile\nThe diagnostic model indicates a low cardiovascular risk. Keep up the great preventative work!")
                    st.balloons()
            else:
                st.warning("⚠️ Backend model offline. UI rendering in preview mode.")

# --- 6. Main Content: Prevention & Articles ---
elif menu == "🔬 Deep Dive: Prevention":
    st.markdown("""
        <div style='background-color: #1E2333; padding: 25px; border-radius: 12px; border-bottom: 4px solid #E63946; margin-bottom: 25px;'>
            <h1 style='margin: 0; color: #FAFAFA;'>Strategies to Prevent Heart Disease</h1>
            <p style='margin: 5px 0 0 0; color: #A0AEC0; font-size: 18px;'>Proactive health management is the key to longevity.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Restored the image of the man wearing a watch/fitness tracker here!
    st.image("https://images.unsplash.com/photo-1434494878577-86c23bcb06b9?auto=format&fit=crop&q=80&w=1200&h=400", caption="Monitoring heart rate and daily activity is a cornerstone of cardiovascular prevention.", use_container_width=True)
    
    st.markdown("""
    <div class="info-card">
        <h3>How to Avoid and Prevent Heart Disease</h3>
        <p>Heart disease is highly preventable through dedicated lifestyle modifications. The foundation of prevention is known as the "ABCS": taking Aspirin if eligible, controlling Blood pressure, managing Cholesterol, and Smoking cessation. Quitting tobacco is one of the most immediate ways to improve heart health; the risk of heart disease drops significantly within just one year of quitting.</p>
        <p>Additionally, treating obesity as a medical condition rather than a simple lifestyle flaw is crucial. Maintaining a healthy Body Mass Index (BMI) drastically reduces the strain on your cardiovascular system. Pairing weight management with regular physical activity—aiming for 150 minutes of moderate aerobic activity weekly—strengthens the heart muscle and lowers blood sugar.</p>
    </div>
    
    <div class="info-card">
        <h3>The Power of Diet and Sleep</h3>
        <p>Dietary choices act as preventative medicine. The DASH (Dietary Approaches to Stop Hypertension) eating plan is heavily endorsed by cardiologists. It emphasizes dark leafy greens, whole grains, and lean proteins, while strictly limiting sodium, refined sugars, and saturated fats. Beyond diet, chronic sleep deprivation places devastating stress on the heart. Prioritizing at least 8 hours of sleep per night is an absolute necessity for cellular repair and cardiovascular resilience.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### 📚 Authoritative Medical Articles & Resources")
    st.write("Click the buttons below to explore peer-reviewed articles and official guidelines directly from top medical authorities.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.link_button("Mayo Clinic: Strategies to Prevent Heart Disease", "https://www.mayoclinic.org/diseases-conditions/heart-disease/in-depth/heart-disease-prevention/art-20046502", use_container_width=True)
        st.link_button("AHA: New Heart Disease & Stroke Guidelines", "https://www.heart.org/en/news/2021/08/04/new-heart-disease-and-stroke-prevention-guidelines-released", use_container_width=True)
    with col_b:
        st.link_button("CDC: Preventing Heart Disease Fact Sheet", "https://www.cdc.gov/heart-disease/prevention/index.html", use_container_width=True)
        st.link_button("Mayo Clinic: Unique Risk Factors in Women", "https://newsnetwork.mayoclinic.org/discussion/womens-wellness-understand-heart-disease-symptoms-and-risk-factors-unique-to-women/", use_container_width=True)
