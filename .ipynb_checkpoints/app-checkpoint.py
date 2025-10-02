import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import time

# -------------------------------
# Load Model and Dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("waterDataset.csv")

@st.cache_resource
def load_model():
    return joblib.load("water_quality_rf_model.pkl")

df = load_data()
model = load_model()

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("💧 Water Quality Classification")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Model Explanation", "Comparison Mode", "Live Monitoring"])

# -------------------------------
# Home
# -------------------------------
if page == "Home":
    st.title("💧 Water Quality Classification Dashboard")
    st.write("""
    Welcome to the **Water Quality Classification Dashboard**!  

    - 🔹 Predict water quality based on 6 key features: **EC, TDS, Na, TH, Cl, pH**  
    - 🔹 Understand *why* the model makes its predictions using **SHAP explainability**  
    - 🔹 Compare multiple water samples side-by-side with **Comparison Mode**  
    - 🔹 Monitor **real-time sensor data (simulated)** in **Live Monitoring**  
    """)

# -------------------------------
# Prediction
# -------------------------------
elif page == "Prediction":
    st.title("🤖 Water Quality Prediction")
    st.write("Enter values for the features below:")

    ec = st.number_input("Enter EC (µS/cm)", value=float(df["EC"].median()))
    tds = st.number_input("Enter TDS (mg/L)", value=float(df["TDS"].median()))
    na = st.number_input("Enter Na (mg/L)", value=float(df["Na"].median()))
    th = st.number_input("Enter TH (mg/L)", value=float(df["TH"].median()))
    cl = st.number_input("Enter Cl (mg/L)", value=float(df["Cl"].median()))
    ph = st.number_input("Enter pH", value=float(df["pH"].median()))

    if st.button("Predict"):
        input_data = pd.DataFrame(
            [[ec, tds, na, th, cl, ph]], 
            columns=["EC", "TDS", "Na", "TH", "Cl", "pH"]
        )

        # 🔹 Prediction
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]   # <-- FIXED: define proba here

        # Save for SHAP use
        st.session_state["last_input"] = input_data  
        st.session_state["last_prediction"] = prediction  
        st.session_state["last_proba"] = proba  

        st.success(f"💡 Predicted Water Quality: **{prediction}**")

        # 🔹 Confidence Table
        proba_df = pd.DataFrame({
            "Class": model.classes_, 
            "Confidence (%)": proba * 100
        }).sort_values(by="Confidence (%)", ascending=False)

        st.subheader("📊 Prediction Confidence")
        st.dataframe(proba_df.style.format({"Confidence (%)": "{:.2f}"}))

        # 🔹 Confidence Chart
        colors = ["green" if c == "Excellent" else "gold" if c == "Good" else "red" for c in proba_df["Class"]]
        fig, ax = plt.subplots()
        bars = ax.barh(proba_df["Class"], proba_df["Confidence (%)"], color=colors)
        ax.set_xlabel("Confidence (%)")
        ax.set_title("Prediction Confidence by Class")

        for bar in bars:
            width = bar.get_width()
            ax.annotate(f'{width:.2f}%', 
                        xy=(width, bar.get_y() + bar.get_height()/2), 
                        xytext=(3, 0), 
                        textcoords="offset points", 
                        ha='left', va='center')

        st.pyplot(fig)

# -------------------------------
# Model Explanation (SHAP)
# -------------------------------
elif page == "Model Explanation":
    st.title("📖 Model Explanation with SHAP")

    explainer = shap.TreeExplainer(model)

    # Use last input if available, otherwise fallback
    if "last_input" in st.session_state:
        input_data = st.session_state["last_input"]
        predicted_class = st.session_state.get("last_prediction", None)
    else:
        input_data = pd.DataFrame(
            [[df["EC"].median(), df["TDS"].median(), df["Na"].median(), 
              df["TH"].median(), df["Cl"].median(), df["pH"].median()]],
            columns=["EC", "TDS", "Na", "TH", "Cl", "pH"]
        )
        predicted_class = None

    shap_values = explainer(input_data)
    class_names = model.classes_

    # If user has a prediction, default to that class
    if predicted_class:
        selected_class = st.selectbox("Choose class to explain", class_names, 
                                      index=list(class_names).index(predicted_class))
    else:
        selected_class = st.selectbox("Choose class to explain", class_names)

    class_index = list(class_names).index(selected_class)

    shap_single = shap.Explanation(
        values=shap_values.values[0, class_index],
        base_values=shap_values.base_values[0, class_index],
        data=shap_values.data[0],
        feature_names=list(input_data.columns)
    )

    st.subheader(f"1. SHAP Waterfall Plot for class: {selected_class}")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_single, show=False)
    st.pyplot(fig)

    st.subheader("2. SHAP Global Feature Importance")
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values[..., class_index], show=False)
    st.pyplot(fig)

    st.subheader("3. Explanation in Simple Terms")
    st.write(f"""
    👉 The model is showing you **why the prediction leaned towards '{selected_class}'**.  

    - **Red bars (positive values)** = pushed prediction *towards this class*.  
    - **Blue bars (negative values)** = pushed prediction *away from this class*.  
    """)
# -------------------------------
# Comparison Mode
# -------------------------------
elif page == "Comparison Mode":
    st.title("📊 Comparison Mode")

    st.write("""
    Upload a CSV file with multiple samples to compare predictions.  
    - File must have columns: **EC, TDS, Na, TH, Cl, pH**  
    """)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        compare_df = pd.read_csv(uploaded_file)

        required_cols = ["EC", "TDS", "Na", "TH", "Cl", "pH"]
        if all(col in compare_df.columns for col in required_cols):
            st.success("✅ File uploaded successfully!")

            predictions = model.predict(compare_df[required_cols])
            probas = model.predict_proba(compare_df[required_cols])

            compare_df["Prediction"] = predictions
            compare_df["Top Confidence (%)"] = probas.max(axis=1) * 100

            st.subheader("📋 Predictions Table")
            st.dataframe(compare_df.style.format({"Top Confidence (%)": "{:.2f}"}))

            st.subheader("📊 Confidence Chart (first 5 samples)")
            for i in range(min(5, len(compare_df))):
                sample_proba = probas[i]
                sample_classes = model.classes_
                colors = ["green" if c == "Excellent" else "gold" if c == "Good" else "red" for c in sample_classes]

                fig, ax = plt.subplots()
                ax.barh(sample_classes, sample_proba * 100, color=colors)
                ax.set_xlabel("Confidence (%)")
                ax.set_title(f"Sample {i+1} Prediction Confidence")

                for j, v in enumerate(sample_proba * 100):
                    ax.text(v + 1, j, f"{v:.2f}%", va='center')

                st.pyplot(fig)
        else:
            st.error(f"❌ CSV must contain columns: {required_cols}")

# -------------------------------
# Live Monitoring (Simulated)
# -------------------------------
elif page == "Live Monitoring":
    st.title("📡 Live Monitoring Dashboard (Simulated)")
    st.write("This simulates sensor data streaming every few seconds...")

    if "live_data" not in st.session_state:
        st.session_state["live_data"] = pd.DataFrame(columns=["EC", "TDS", "Na", "TH", "Cl", "pH", "Prediction"])

    # Generate new random sample
    new_sample = {
        "EC": float(np.random.uniform(100, 1500)),
        "TDS": float(np.random.uniform(50, 1200)),
        "Na": float(np.random.uniform(5, 200)),
        "TH": float(np.random.uniform(10, 400)),
        "Cl": float(np.random.uniform(5, 250)),
        "pH": float(np.random.uniform(6.0, 9.0)),
    }
    input_data = pd.DataFrame([new_sample])
    prediction = model.predict(input_data)[0]
    new_sample["Prediction"] = prediction

    # Append to session and keep last 20 samples
    st.session_state["live_data"] = pd.concat([st.session_state["live_data"], pd.DataFrame([new_sample])]).tail(20)

    st.subheader("🔹 Latest Reading")
    st.json(new_sample)

    st.subheader("📈 Recent Trends")
    st.line_chart(st.session_state["live_data"].drop(columns="Prediction"))

    st.subheader("📊 Recent Predictions Count")
    st.bar_chart(st.session_state["live_data"]["Prediction"].value_counts())

    # Refresh every 2 seconds
    time.sleep(2)
    st.rerun()