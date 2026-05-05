import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

# -------------------------------
# Page configuration
st.set_page_config(page_title="Diabetes Prediction System", layout="wide")
st.title("🩺 Diabetes Prediction System")
st.markdown("Predict whether a patient has diabetes based on medical parameters.")

# -------------------------------
# Helper functions
@st.cache_resource
def load_or_train_model():
    """Load pre-trained model or train & save if not exists."""
    model_path = "models/logistic_reg.sav"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    else:
        # Load dataset
        df = pd.read_csv("diabetes.csv")
        # Separate features and target
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Create pipeline with scaling and logistic regression
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        # Evaluate
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ Model trained on the fly. Test accuracy: {acc:.2f}")
        # Save model
        os.makedirs("models", exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)
        return pipeline

# -------------------------------
# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Single Prediction", "Batch Prediction", "About & Dataset Info"])

# Load model (cached)
model = load_or_train_model()

# -------------------------------
# Single Prediction Page
if option == "Single Prediction":
    st.header("🔍 Single Patient Prediction")
    st.write("Enter the patient's health parameters below:")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    with col2:
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
        age = st.number_input("Age", min_value=0, max_value=120, value=30)

    # Input data as DataFrame
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]],
                              columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                                       "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])

    if st.button("🔮 Predict Diabetes", type="primary"):
        # Predict
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]  # Probability of class 1

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ **Diabetic** (Probability: {proba:.2f})")
            st.markdown("The model indicates a **high risk** of diabetes. Please consult a doctor.")
        else:
            st.success(f"✅ **Not Diabetic** (Probability: {proba:.2f})")
            st.markdown("The model indicates **low risk**. Maintain a healthy lifestyle.")

# -------------------------------
# Batch Prediction Page
elif option == "Batch Prediction":
    st.header("📂 Batch Prediction from CSV File")
    st.markdown("Upload a CSV file with the same columns as the training dataset (without the `Outcome` column).")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview")
        st.dataframe(df_upload.head())

        # Check required columns
        required_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                         "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
        if all(col in df_upload.columns for col in required_cols):
            # Make predictions
            predictions = model.predict(df_upload[required_cols])
            probabilities = model.predict_proba(df_upload[required_cols])[:, 1]

            # Add results to dataframe
            df_upload["Prediction"] = predictions
            df_upload["Probability (Diabetic)"] = probabilities
            df_upload["Result"] = df_upload["Prediction"].apply(lambda x: "Diabetic" if x == 1 else "Not Diabetic")

            st.subheader("📊 Prediction Results")
            st.dataframe(df_upload)

            # Download button
            csv_output = df_upload.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Predictions as CSV", csv_output, "predictions.csv", "text/csv")

            # Summary statistics
            st.subheader("Summary")
            diabetic_count = df_upload["Prediction"].sum()
            total = len(df_upload)
            st.metric("Total Patients", total)
            st.metric("Predicted Diabetic", diabetic_count, delta=f"{diabetic_count/total*100:.1f}%")
        else:
            st.error(f"CSV must contain columns: {required_cols}")

# -------------------------------
# About & Dataset Info Page
else:
    st.header("📖 About the Diabetes Prediction System")
    st.markdown("""
    ### **Overview**
    This application uses a **Logistic Regression** model to predict whether a patient has diabetes based on diagnostic measurements.
    The model was trained on the **PIMA Indian Diabetes Database** from the National Institute of Diabetes and Digestive and Kidney Diseases.

    ### **Features (Input Variables)**
    - **Pregnancies** – Number of times pregnant  
    - **Glucose** – Plasma glucose concentration after 2 hours in an oral glucose tolerance test  
    - **BloodPressure** – Diastolic blood pressure (mm Hg)  
    - **SkinThickness** – Triceps skin fold thickness (mm)  
    - **Insulin** – 2‑Hour serum insulin (mu U/ml)  
    - **BMI** – Body mass index (weight in kg/(height in m)²)  
    - **DiabetesPedigreeFunction** – A function that scores likelihood of diabetes based on family history  
    - **Age** – Age in years  

    ### **Model Performance**
    """)

    # Show model evaluation on original dataset (optional)
    df_full = pd.read_csv("diabetes.csv")
    X_full = df_full.drop("Outcome", axis=1)
    y_full = df_full["Outcome"]
    _, X_test, _, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    st.metric("Test Accuracy", f"{acc:.2%}")

    # Confusion matrix heatmap
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    st.markdown("""
    ### **How to Use**
    1. **Single Prediction** – Enter patient data manually and click *Predict Diabetes*.  
    2. **Batch Prediction** – Upload a CSV file with the same features (except `Outcome`) and download predictions.  
    3. **About** – Learn about the model and dataset.

    ### **Technical Notes**
    - The model is trained automatically on first run and saved locally (`models/logistic_reg.sav`).
    - Features are **standardized** (zero mean, unit variance) before training for better performance.
    - The app uses **caching** to avoid reloading the model repeatedly.

    **Built with** Streamlit, scikit-learn, pandas, and matplotlib.
    """)    