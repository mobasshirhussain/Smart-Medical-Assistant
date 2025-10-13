# Smart Medical Assistant – Streamlit App
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# PAGE CONFIG & STYLING
st.set_page_config(page_title="Smart Medical Assistant", page_icon="🩺", layout="wide")
st.title("🩺 Smart Medical Assistant")
st.markdown("#### A research-based project for disease prediction with doctor recommendations")

#  LOAD MODEL BUNDLE
@st.cache_resource
def load_bundle():
    with open("medical_model.pkl", "rb") as f:
        b = pickle.load(f)
    return b["model"], b["encoder"], b["scaler"], b["feature_names"]

model, encoder, scaler, feature_names = load_bundle()

#  DOCTOR MAPPING
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

#  TABS: PREDICT | EVALUATION
tabs = st.tabs(["🔍 Predict Disease", "📊 Model Evaluation"])

# PREDICT TAB 
with tabs[0]:
    st.subheader("Select your symptoms")
    selected_symptoms = st.multiselect("Choose symptoms:", feature_names)

    if st.button("Predict", type="primary"):
        if not selected_symptoms:
            st.warning(" Please select at least one symptom.")
        else:
            x = np.array([1 if s in selected_symptoms else 0 for s in feature_names], dtype=float).reshape(1, -1)
            x_scaled = scaler.transform(x)
            y_enc = model.predict(x_scaled)[0]
            disease = encoder.inverse_transform([y_enc])[0]

            doctor = doctor_mapping.get(disease, "General Physician")

            st.success(f"🧾 **Predicted Disease:** {disease}")
            st.info(f"👨‍⚕️ **Recommended Doctor:** {doctor}")

            with st.expander("About this project"):
                st.caption("Research-based project with a curated dataset (knowledge base, not real patient data).")
                # st.caption("100% accuracy is expected because:")
                # st.caption("• Each disease has a fixed set of symptom mappings.")
                # st.caption("• The ML model efficiently learns these mappings.")
                # st.caption("• That's the nature of a knowledge-driven dataset.")
                st.caption("This tool is for educational purposes only. Consult a licensed physician for medical advice.")
                # st.caption("Additional contributions:")
                # st.caption(" Cleaning & preprocessing pipeline")
                # st.caption(" Accuracy, precision, recall, F1 + confusion matrix")
                # st.caption(" Feature importance analysis")
                # st.caption("Streamlit UI with doctor recommendation mapping")

#  EVALUATION TAB 
with tabs[1]:
    st.subheader("Model Performance Metrics")
    # Load test metrics from training or recompute quickly
    # (Optional: store y_test/y_pred in bundle if you want exact values)
    st.markdown("""
    -  Accuracy, Precision, Recall, F1-score  
    -  Confusion Matrix  
    -  Top 10 most important symptoms  
    """)

    # Confusion matrix on training data (quick view)
    st.write("### Confusion Matrix")
    # We cannot recompute without y_test, but show placeholder
    st.caption("Load metrics from training script for full report.")
    st.warning("For detailed matrix, save `y_test` and `y_pred` in the bundle during training.")

    st.write("### Top 10 Important Symptoms")
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8,5))
    importances.iloc[::-1].plot.barh(ax=ax, color="teal")
    ax.set_title("Top 10 Important Symptoms")
    ax.set_xlabel("Importance")
    st.pyplot(fig)
