# 🩺 Smart Medical Assistant
import pickle
import numpy as np
import streamlit as st

# -------------------- PAGE SETTINGS --------------------
st.set_page_config(page_title="Smart Medical Assistant", page_icon="🩺", layout="wide")

# -------------------- STYLE --------------------
st.markdown("""
<style>
.stApp {
    background-color: #f9fafb;
    color: #111;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
h1 {
    text-align: center;
    color: #d32f2f;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.subtitle {
    text-align: center;
    color: #555;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}
.section-title {
    color: #1565c0;
    font-size: 1.3rem;
    font-weight: 700;
    margin-top: 2rem;
}
.stButton>button {
    background-color: #d32f2f;
    color: white;
    font-weight: 700;
    font-size: 1rem;
    padding: 10px 25px;
    border-radius: 8px;
    border: none;
    transition: 0.3s ease;
    width: 100%;
}
.stButton>button:hover {
    background-color: #b71c1c;
    transform: scale(1.03);
}
.stMultiSelect div[data-baseweb="select"] {
    background-color: #fff !important;
    color: #111 !important;
    border-radius: 8px;
    border: 1px solid #ccc;
}
.stMultiSelect span[data-baseweb="tag"] {
    background-color: #d32f2f !important;
    color: white !important;
    font-weight: 600;
}
.result-box {
    background-color: #e8f5e9;
    border-left: 8px solid #43a047;
    padding: 15px;
    border-radius: 10px;
    margin-top: 20px;
    font-size: 1.1rem;
    color: #1b5e20;
    font-weight: 700;
}
.doctor-box {
    background-color: #e3f2fd;
    border-left: 8px solid #1976d2;
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
    font-size: 1.1rem;
    color: #0d47a1;
    font-weight: 700;
}

/* About Section */
.about-expander {
    border: 2px solid #90caf9;
    background-color: #f1f8ff;
    border-radius: 10px;
    padding: 10px 15px;
    margin-top: 20px;
}
.about-text {
    color: #333;
    font-weight: 600;
    font-size: 0.95rem;
    line-height: 1.4;
    border: 2px solid #90caf9;
    border-radius: 8px;
    padding: 10px;
    background-color: #f8fbff;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("<h1>🩺 Smart Medical Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-powered Symptom-to-Disease Predictor</p>", unsafe_allow_html=True)
# st.markdown("-")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_bundle():
    with open("medical_model.pkl", "rb") as f:
        b = pickle.load(f)
    return b["model"], b["encoder"], b["scaler"], b["feature_names"]

model, encoder, scaler, feature_names = load_bundle()

# -------------------- DOCTOR MAPPING --------------------
doctor_mapping = {
    "Fungal infection": "Dermatologist (Skin Specialist)",
    "Allergy": "Immunologist / General Physician",
    "GERD": "Gastroenterologist",
    "Chronic cholestasis": "Hepatologist / Gastroenterologist",
    "Drug Reaction": "Dermatologist / Allergist",
    "Peptic ulcer disease": "Gastroenterologist",
    "AIDS": "Infectious Disease Specialist",
    "Diabetes": "Endocrinologist",
    "Gastroenteritis": "Gastroenterologist",
    "Bronchial Asthma": "Pulmonologist",
    "Hypertension": "Cardiologist",
    "Migraine": "Neurologist",
    "Cervical spondylosis": "Orthopedic / Neurologist",
    "Paralysis (brain hemorrhage)": "Neurologist",
    "Jaundice": "Hepatologist",
    "Malaria": "Infectious Disease Specialist / General Physician",
    "Chicken pox": "Dermatologist / Infectious Disease Specialist",
    "Dengue": "Infectious Disease Specialist",
    "Typhoid": "Infectious Disease Specialist",
    "Hepatitis A": "Hepatologist",
    "Hepatitis B": "Hepatologist",
    "Hepatitis C": "Hepatologist",
    "Hepatitis D": "Hepatologist",
    "Hepatitis E": "Hepatologist",
    "Alcoholic hepatitis": "Hepatologist",
    "Tuberculosis": "Pulmonologist / Infectious Disease Specialist",
    "Common Cold": "General Physician",
    "Pneumonia": "Pulmonologist",
    "Dimorphic hemorrhoids(piles)": "Proctologist / Gastroenterologist",
    "Heart attack": "Cardiologist",
    "Varicose veins": "Vascular Surgeon",
    "Hypothyroidism": "Endocrinologist",
    "Hyperthyroidism": "Endocrinologist",
    "Hypoglycemia": "Endocrinologist",
    "Osteoarthristis": "Orthopedic",
    "Arthritis": "Rheumatologist",
    "(vertigo) Paroymsal  Positional Vertigo": "ENT Specialist / Neurologist",
    "Acne": "Dermatologist",
    "Urinary tract infection": "Urologist",
    "Psoriasis": "Dermatologist",
    "Impetigo": "Dermatologist"
}

# -------------------- INPUT SECTION --------------------
st.markdown("<div class='section-title'>🧩 Select Your Symptoms</div>", unsafe_allow_html=True)
selected_symptoms = st.multiselect("Choose symptoms you are experiencing:", feature_names)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_btn = st.button("🔮 Predict Disease")

# -------------------- PREDICTION SECTION --------------------
if predict_btn:
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        x = np.array([1 if s in selected_symptoms else 0 for s in feature_names], dtype=float).reshape(1, -1)
        x_scaled = scaler.transform(x)
        y_enc = model.predict(x_scaled)[0]
        disease = encoder.inverse_transform([y_enc])[0]
        doctor = doctor_mapping.get(disease, "General Physician")

        st.markdown(f"<div class='result-box'>🧾 Predicted Disease: <b>{disease}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='doctor-box'>👨‍⚕️ Recommended Doctor: <b>{doctor}</b></div>", unsafe_allow_html=True)

        # 🩺 ABOUT SECTION - Clickable
        with st.expander("ℹ️ About This Project"):
            st.markdown("""
            <div class='about-text'>
                📘 This is a <b>research-based academic project</b> developed for disease prediction using curated medical data.<br><br>
                ⚠️ <b>Disclaimer:</b> This tool is for educational and demonstration purposes only. Always consult a licensed physician for real medical advice.
            </div>
            """, unsafe_allow_html=True)
