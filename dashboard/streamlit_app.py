# Importing required libraries for file handling, model loading, UI, and visualization
import os
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


# Load Model

# Getting base directory path to correctly locate model file
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Constructing full path to saved model
model_path = os.path.join(BASE_DIR, "models", "fraud_model.pkl")

# Loading trained model using pickle
model = pickle.load(open(model_path, "rb"))


# Setting Streamlit page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# Main dashboard title
st.title("💳 Fraud Detection System Dashboard")


# Sidebar Navigation

# Creating sidebar menu for navigation between different sections
menu = st.sidebar.selectbox("Menu", [
    "📊 Overview",
    "🔍 Predict Transaction",
    "📁 Batch Prediction",
    "📈 Model Performance",
    "📡 Monitoring"
])


# Load Sample Data

# Caching data to improve performance (avoids reloading on every interaction)
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(BASE_DIR, "data", "creditcard.csv")

    df = pd.read_csv(data_path)

    # Applying same feature engineering as used during model training
    df['hour'] = df['Time'] // 3600
    df['amount_log'] = np.log1p(df['Amount'])

    return df

# Loading dataset
df = load_data()


# 📊 OVERVIEW

if menu == "📊 Overview":
    st.title("💳 Fraud Detection Dashboard")

    # Creating KPI cards
    col1, col2, col3 = st.columns(3)

    total = len(df)
    fraud = df['Class'].sum()
    non_fraud = total - fraud

    # Displaying key metrics
    col1.metric("Total Transactions", total)
    col2.metric("Fraud Cases", fraud)
    col3.metric("Non-Fraud Cases", non_fraud)

    # Visualizing class distribution
    st.subheader("Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Class', data=df, ax=ax)
    st.pyplot(fig)


# 🔍 PREDICTION

elif menu == "🔍 Predict Transaction":
    st.subheader("Predict Fraud")

    input_data = []

    # Taking user input for each feature
    for col in df.columns[:-1]:
        val = st.number_input(f"{col}", value=0.0)
        input_data.append(val)

    if st.button("Predict"):
        # Creating DataFrame from user input
        input_df = pd.DataFrame([input_data], columns=df.columns[:-1])

        # Applying same feature engineering as training
        input_df['hour'] = input_df['Time'] // 3600
        input_df['amount_log'] = np.log1p(input_df['Amount'])

        # Ensuring correct column order for model
        input_df = input_df[model.get_booster().feature_names]

        # Making prediction
        prediction = model.predict(input_df)[0]

        # Showing result
        if prediction == 1:
            st.error("🚨 Fraud Detected!")
            st.toast("High risk transaction detected!", icon="⚠️")
        else:
            st.success("✅ Legit Transaction")


# 📈 MODEL PERFORMANCE

elif menu == "📈 Model Performance":
    st.subheader("📈 Model Evaluation")

    # Preparing data
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Generating predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Calculating ROC-AUC score
    roc_auc = roc_auc_score(y, y_prob)

    # Displaying metrics
    col1, col2 = st.columns(2)
    col1.metric("ROC-AUC Score", f"{roc_auc:.4f}")
    col2.metric("Model Type", "XGBoost")

    st.markdown("---")

    # Confusion Matrix visualization
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

    # ROC Curve visualization
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y, y_prob)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax2.plot([0, 1], [0, 1], linestyle='--')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()

    st.pyplot(fig2)

    # Business insights section
    st.markdown("### 💡 Insights")
    st.info("""
    - High ROC-AUC indicates strong fraud detection capability  
    - Confusion matrix shows balance between fraud detection and false positives  
    - Model performs well in identifying rare fraud cases  
    """)



# 📡 MONITORING (PSI)

elif menu == "📡 Monitoring":
    st.subheader("Model Monitoring (PSI)")

    st.write("Simulating distribution shift...")

    # Getting original predictions
    probs_original = model.predict_proba(df.drop('Class', axis=1))[:,1]

    # Simulating shift in data
    shifted = df.copy()
    shifted['Amount'] *= 1.5

    # Predictions after shift
    probs_shifted = model.predict_proba(shifted.drop('Class', axis=1))[:,1]

    # PSI calculation function
    def calculate_psi(expected, actual, bins=10):
        breakpoints = np.linspace(0, 1, bins + 1)
        expected_bins = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        actual_bins = np.histogram(actual, bins=breakpoints)[0] / len(actual)
        psi = np.sum((actual_bins - expected_bins) * np.log((actual_bins + 1e-6) / (expected_bins + 1e-6)))
        return psi

    # Calculating PSI value
    psi_value = calculate_psi(probs_original, probs_shifted)

    st.write(f"PSI Value: {psi_value:.4f}")

    # Interpreting PSI result
    if psi_value > 0.2:
        st.error("⚠️ Significant drift detected! Retraining required.")
    else:
        st.success("✅ Model is stable.")


# 📁 BATCH PREDICTION

elif menu == "📁 Batch Prediction":
    st.subheader("📁 Upload CSV for Fraud Detection")

    # Upload file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Showing preview of uploaded data
        st.write("📊 Uploaded Data Preview:")
        st.dataframe(data.head())

        try:
            # Applying same feature engineering
            data['hour'] = data['Time'] // 3600
            data['amount_log'] = np.log1p(data['Amount'])

            # Removing target column if present
            if 'Class' in data.columns:
                data = data.drop('Class', axis=1)

            # Ensuring correct column order
            data = data[model.get_booster().feature_names]

            # Making predictions
            predictions = model.predict(data)
            probabilities = model.predict_proba(data)[:, 1]

            # Adding results to dataset
            data['Fraud_Prediction'] = predictions
            data['Fraud_Probability'] = probabilities

            st.success("✅ Prediction Completed!")
            st.dataframe(data.head())

            # Download results
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Results",
                csv,
                "fraud_predictions.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"Error: {e}")