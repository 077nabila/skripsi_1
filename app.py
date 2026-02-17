# =========================
# IMPORT LIBRARY
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model


# =========================
# TITLE
# =========================

st.title("PREDIKSI CURAH HUJAN MENGGUNAKAN LSTM")

st.sidebar.title("Main Menu")

menu = st.sidebar.radio(
    "Go to",
    [
        "Dataset",
        "Interpolasi Linear",
        "Normalisasi Data",
        "Model LSTM",
        "Prediksi LSTM",
        "Implementasi"
    ]
)


# =========================
# SESSION STATE INIT
# =========================

for key in [
    "df",
    "scaler",
    "scaled_data",
    "model",
    "x_test",
    "y_test",
    "test_predictions"
]:
    if key not in st.session_state:
        st.session_state[key] = None


# =========================
# DATASET
# =========================

if menu == "Dataset":

    df = pd.read_excel("Dataset_Curah_Hujan.xlsx")

    df["Tanggal"] = pd.to_datetime(df["Tanggal"])

    df = df.reset_index(drop=True)

    st.session_state.df = df

    st.write(df)



# =========================
# INTERPOLASI LINEAR (3 FITUR)
# =========================

elif menu == "Interpolasi Linear":

    df = st.session_state.df

    if df is None:
        st.warning("Load dataset dulu")
        st.stop()

    # GANTI SESUAI FITUR KAMU
    fitur = ["RR", "Tn", "Tx"]

    df[fitur] = df[fitur].interpolate(method="linear")

    df[fitur] = df[fitur].bfill()
    df[fitur] = df[fitur].ffill()

    st.session_state.df = df

    st.write(df)

    st.success("Inte
