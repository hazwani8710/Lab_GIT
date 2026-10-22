# eBantuan Eligibility Prediction System
# Purpose: Predict assistance eligibility using Streamlit.

import streamlit as st
import pandas as pd
import joblib

# Configure page
st.set_page_config(
    page_title="eBantuan Eligibility Prediction",
    page_icon="📋"
)

st.title("Sistem Ramalan Kelayakan Bantuan")
st.write("Logistic Regression Classification")

# Load saved model and preprocessing objects
@st.cache_resource
def load_model():
    model = joblib.load("ebantuan_logistic_model.joblib")
    encoder = joblib.load("ebantuan_encoder.joblib")
    scaler = joblib.load("ebantuan_scaler.joblib")
    features = joblib.load("ebantuan_features.joblib")

    return model, encoder, scaler, features


try:
    model, encoder, scaler, features = load_model()
except Exception as e:
    st.error(f"Failed to load model files: {e}")
    st.stop()

# Load feature information
numerical_columns = features["numerical_features"]
categorical_columns = features["categorical_features"]
model_features = features["model_features"]

# Validate saved model configuration
required_columns = [
    "pendapatan_bulanan",
    "pendapatan_per_kapita",
    "umur",
    "bil_tanggungan"
]

if (
    not all(col in numerical_columns for col in required_columns)
    or "kategori_pendapatan" in categorical_columns
):
    st.error("Saved features do not match the expected model.")
    st.stop()

# Input form
st.subheader("Maklumat Pemohon")

with st.form("eligibility_form"):

    pendapatan = st.number_input(
        "Pendapatan Bulanan (RM)",
        min_value=0.0,
        value=2500.0,
        step=100.0
    )

    umur = st.number_input(
        "Umur (tahun)",
        min_value=1,
        max_value=120,
        value=40,
        step=1
    )

    tanggungan = st.number_input(
        "Bilangan Tanggungan",
        min_value=0,
        value=3,
        step=1
    )

    # Collect categorical inputs using training categories
    categorical_input = {}

    labels = {
        "kategori_penerima": "Kategori Penerima",
        "status_pekerjaan": "Status Pekerjaan"
    }

    for index, column in enumerate(categorical_columns):

        categories = encoder.categories_[index].tolist()

        categorical_input[column] = st.selectbox(
            labels.get(column, column),
            options=categories
        )

    submitted = st.form_submit_button(
        "Ramalkan Kelayakan",
        type="primary"
    )

# Predict when button is clicked
if submitted:

    # Calculate income per capita automatically
    per_capita = pendapatan / (tanggungan + 1)

    # Prepare numerical and categorical inputs
    user_input = {
        "pendapatan_bulanan": pendapatan,
        "pendapatan_per_kapita": per_capita,
        "umur": umur,
        "bil_tanggungan": tanggungan
    }

    user_input.update(categorical_input)

    input_df = pd.DataFrame([user_input])

    # Apply saved One-Hot Encoder
    encoded_data = encoder.transform(
        input_df[categorical_columns]
    )

    # Handle sparse and dense encoder outputs
    if hasattr(encoded_data, "toarray"):
        encoded_data = encoded_data.toarray()

    encoded_columns = encoder.get_feature_names_out(
        categorical_columns
    )

    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoded_columns
    )

    # Combine numerical and categorical features
    input_final = pd.concat(
        [
            input_df[numerical_columns].reset_index(drop=True),
            encoded_df
        ],
        axis=1
    )

    # Match training feature order
    input_final = input_final.reindex(
        columns=model_features,
        fill_value=0
    )

    # Apply saved MinMaxScaler
    input_final[numerical_columns] = scaler.transform(
        input_final[numerical_columns]
    )

    # Predict assistance eligibility
    prediction = model.predict(input_final)[0]

    # Display result
    st.subheader("Keputusan Ramalan")

    st.write(
        f"Pendapatan Per Kapita: RM {per_capita:,.2f}"
    )

    if prediction == 1:
        st.success("11Ramalan: LAYAK (YA)")
    else:
        st.warning("Ramalan: TIDAK LAYAK (TIDAK)")

    st.info(
        "Ramalan ini adalah untuk tujuan latihan sahaja. "
        "Keputusan kelayakan sebenar memerlukan semakan pegawai."
    )
