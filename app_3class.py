import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ============================================
# Config
# ============================================
DATA_FILE = "waterDataset.csv"
MODEL_FILE = "water_quality_xgb_3class.pkl"
FEATURES = ["EC", "TDS", "Na", "TH", "Cl", "pH"]
# Label order used during training (LabelEncoder sorts alphabetically)
CLASS_NAMES = ["Good", "Poor", "Unsuitable"]
COLOR_MAP = {"Good": "green", "Poor": "gold", "Unsuitable": "red"}

st.set_page_config(page_title="Water Quality (3-Class)", page_icon="💧", layout="centered")

# ============================================
# Loaders (cached)
# ============================================
@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

df = load_data()
model = load_model()

# ============================================
# Sidebar
# ============================================
st.sidebar.title("💧 Water Quality (3-Class)")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Model Explanation", "Comparison Mode"])

# ============================================
# Home
# ============================================
if page == "Home":
    st.title("💧 Water Quality Classification.")
    st.write("""
This version uses your **best model** from cross-validation (**XGBoost**) and reduces classes to:
- **Good**  (was: Excellent + Good)
- **Poor**  (was: Poor + Very Poor yet Drinkable)
- **Unsuitable**

Why? It **reduces bias** from extremely imbalanced tails and improved macro-F1 to **0.984** in 5-fold CV.
""")

# ============================================
# Prediction
# ============================================
elif page == "Prediction":
    st.title("Predict Water Quality (3-Class)")

    # Inputs (use dataset medians as defaults)
    col1, col2 = st.columns(2)
    with col1:
        ec  = st.number_input("EC (µS/cm)",  value=float(df["EC"].median()))
        na  = st.number_input("Na (mg/L)",   value=float(df["Na"].median()))
        cl  = st.number_input("Cl (mg/L)",   value=float(df["Cl"].median()))
    with col2:
        tds = st.number_input("TDS (mg/L)",  value=float(df["TDS"].median()))
        th  = st.number_input("TH (mg/L)",   value=float(df["TH"].median()))
        ph  = st.number_input("pH",          value=float(df["pH"].median()))

    if st.button("Predict"):
        # Build input row in the exact training order
        X_row = pd.DataFrame([[ec, tds, na, th, cl, ph]], columns=FEATURES)

        # XGB was trained with integer labels [0,1,2] that correspond to CLASS_NAMES
        y_pred_int = model.predict(X_row)[0]
        y_proba = model.predict_proba(X_row)[0]   # order corresponds to model.classes_ = [0,1,2]

        # Map probas to our class names using the same order
        proba_df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Confidence (%)": y_proba * 100.0
        }).sort_values(by="Confidence (%)", ascending=False)

        pred_label = CLASS_NAMES[int(y_pred_int)]
        st.success(f"💡 Predicted Water Quality: **{pred_label}**")

        # Save for SHAP page
        st.session_state["last_input"] = X_row
        st.session_state["last_prediction"] = pred_label
        st.session_state["last_proba_df"] = proba_df

        # Confidence as table
        st.subheader("📊 Prediction Confidence")
        st.dataframe(proba_df.style.format({"Confidence (%)": "{:.2f}"}))

        # Confidence as horizontal bars with labels
        colors = [COLOR_MAP.get(c, "blue") for c in proba_df["Class"]]
        fig, ax = plt.subplots()
        bars = ax.barh(proba_df["Class"], proba_df["Confidence (%)"], color=colors)
        ax.set_xlabel("Confidence (%)")
        ax.set_title("Prediction Confidence by Class")
        for bar in bars:
            width = bar.get_width()
            ax.annotate(f"{width:.2f}%",
                        xy=(width, bar.get_y() + bar.get_height()/2),
                        xytext=(4, 0), textcoords="offset points",
                        va="center", ha="left")
        st.pyplot(fig)

# ============================================
# Model Explanation (SHAP)
# ============================================
elif page == "Model Explanation":
    st.title("📖 Why did the model predict that? (SHAP)")

    # Use last prediction if available; else median sample
    if "last_input" in st.session_state:
        X_row = st.session_state["last_input"]
        default_class = st.session_state.get("last_prediction", CLASS_NAMES[0])
    else:
        X_row = pd.DataFrame([[df[c].median() for c in FEATURES]], columns=FEATURES)
        default_class = CLASS_NAMES[0]

    # TreeExplainer works natively with XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_row)     # shape: (1, n_features, n_classes)

    # Choose a class to explain
    selected_class = st.selectbox("Class to explain", CLASS_NAMES,
                                  index=CLASS_NAMES.index(default_class))
    class_index = CLASS_NAMES.index(selected_class)

    # Build a single-sample Explanation object for the selected class
    # shap_values.values: (1, n_features, n_classes)
    shap_single = shap.Explanation(
        values=shap_values.values[0, :, class_index],
        base_values=shap_values.base_values[0, class_index],
        data=X_row.values[0, :],
        feature_names=FEATURES
    )

    view = st.radio("Explanation view", ["Waterfall (local)", "Bar (global-ish on this sample)", "Plain-language"])
    if view == "Waterfall (local)":
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_single, show=False)
        st.pyplot(fig)
    elif view == "Bar (global-ish on this sample)":
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[..., class_index], show=False)
        st.pyplot(fig)
    else:
        st.write(f"""
**Interpretation tips for class: _{selected_class}_**

- **Positive bars** push the prediction **towards** _{selected_class}_.  
- **Negative bars** push the prediction **away** from _{selected_class}_.  
- The **longer** the bar, the **stronger** the effect.  
- Read it like a doctor’s note: “Because EC was high and TH moderate, the sample leaned toward _{selected_class}_.”
""")

# ============================================
# Comparison Mode
# ============================================
elif page == "Comparison Mode":
    st.title("📊 Compare Multiple Samples")
    st.write("Upload a CSV with columns: **EC, TDS, Na, TH, Cl, pH**")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df_up = pd.read_csv(uploaded_file)
        missing = [c for c in FEATURES if c not in df_up.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            y_pred_int = model.predict(df_up[FEATURES])
            y_proba = model.predict_proba(df_up[FEATURES])
            labels = [CLASS_NAMES[int(i)] for i in y_pred_int]
            df_up["Prediction"] = labels
            df_up["Top Confidence (%)"] = (y_proba.max(axis=1) * 100).round(2)
            st.dataframe(df_up)

            st.subheader("Confidence (first 5)")
            for i in range(min(5, len(df_up))):
                p = y_proba[i]
                chart_df = pd.DataFrame({"Class": CLASS_NAMES, "Confidence (%)": p*100})
                colors = [COLOR_MAP.get(c, "blue") for c in chart_df["Class"]]
                fig, ax = plt.subplots()
                bars = ax.barh(chart_df["Class"], chart_df["Confidence (%)"], color=colors)
                ax.set_xlabel("Confidence (%)")
                ax.set_title(f"Sample {i+1}")
                for bar in bars:
                    width = bar.get_width()
                    ax.annotate(f"{width:.2f}%",
                                xy=(width, bar.get_y() + bar.get_height()/2),
                                xytext=(4, 0), textcoords="offset points",
                                va="center", ha="left")
                st.pyplot(fig)