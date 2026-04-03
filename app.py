import streamlit as st
import pickle
import pandas as pd

# Load model, PCA, encoder
model = pickle.load(open("model/model.pkl", "rb"))
pca = pickle.load(open("model/pca.pkl", "rb"))
encoder = pickle.load(open("model/encoder.pkl", "rb"))

st.title("🧠 Brain Tumor Prediction System")

st.write("Upload a CSV file with gene expression data:")

uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("📊 Input Data Preview:")
    st.write(df.head())

    # Clean data
    df = df.drop(columns=["samples", "type"], errors="ignore")
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)

    # ✅ BUTTON MUST BE HERE
    if st.button("Predict"):
        try:
            data_pca = pca.transform(df)
            predictions = model.predict(data_pca)
            tumor_types = encoder.inverse_transform(predictions)

            df["Prediction"] = tumor_types

            st.success("✅ Prediction Completed")
            st.write(df)

        except Exception as e:
            st.error(f"❌ Error: {e}")