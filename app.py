# ============================================================
# IMPORT LIBRARY
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# ============================================================
# KONFIGURASI
# ============================================================

st.set_page_config(
    page_title="Prediksi Curah Hujan LSTM",
    layout="wide"
)

st.title("Prediksi Curah Hujan Menggunakan LSTM")

# FITUR DAN TIMESTEP (HARUS SESUAI MODEL)
FITUR = ["TAVG", "RH_AVG", "RR"]
TIMESTEP = 50

# ============================================================
# SESSION STATE
# ============================================================

if "model" not in st.session_state:
    st.session_state.model = None

if "scaler" not in st.session_state:
    st.session_state.scaler = None

if "data" not in st.session_state:
    st.session_state.data = None


# ============================================================
# MENU
# ============================================================

menu = st.sidebar.selectbox(
    "Menu",
    [
        "Upload Data",
        "Load Model",
        "Prediksi Test",
        "Prediksi Masa Depan"
    ]
)


# ============================================================
# FUNGSI BUAT DATASET
# ============================================================

def create_dataset(data, timestep):
    X, y = [], []

    for i in range(len(data) - timestep):
        X.append(data[i:(i+timestep)])
        y.append(data[i+timestep, 2])  # RR

    return np.array(X), np.array(y)


# ============================================================
# MENU 1 — UPLOAD DATA
# ============================================================

if menu == "Upload Data":

    file = st.file_uploader("Upload file CSV", type=["csv"])

    if file is not None:

        data = pd.read_csv(file)

        st.write("Data Awal:")
        st.dataframe(data.head())

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data[FITUR])

        st.session_state.data = data
        st.session_state.scaler = scaler

        st.success("Data berhasil diupload dan discaling")


# ============================================================
# MENU 2 — LOAD MODEL
# ============================================================

elif menu == "Load Model":

    if st.button("Load Model"):

        try:

            BASE_DIR = os.path.dirname(os.path.abspath(__file__))

            model_path = os.path.join(
                BASE_DIR,
                "model_ts_50_ep_100_lr_0.01.h5"
            )

            model = load_model(
                model_path,
                compile=False
            )

            model.compile(
                optimizer="adam",
                loss="mse"
            )

            st.session_state.model = model

            st.success("Model berhasil di-load")

        except Exception as e:
            st.error(f"Gagal load model: {e}")


# ============================================================
# MENU 3 — PREDIKSI TEST
# ============================================================

elif menu == "Prediksi Test":

    if st.session_state.model is None:
        st.warning("Load model dulu")
        st.stop()

    if st.session_state.data is None:
        st.warning("Upload data dulu")
        st.stop()

    scaler = st.session_state.scaler
    model = st.session_state.model
    data = st.session_state.data

    data_scaled = scaler.transform(data[FITUR])

    X, y = create_dataset(data_scaled, TIMESTEP)

    pred = model.predict(X)

    # inverse scaling
    dummy = np.zeros((len(pred), len(FITUR)))
    dummy[:, 2] = pred[:, 0]

    pred_inv = scaler.inverse_transform(dummy)[:, 2]

    dummy2 = np.zeros((len(y), len(FITUR)))
    dummy2[:, 2] = y

    y_inv = scaler.inverse_transform(dummy2)[:, 2]

    # plot
    fig, ax = plt.subplots()

    ax.plot(y_inv, label="Aktual")
    ax.plot(pred_inv, label="Prediksi")

    ax.legend()
    ax.set_title("Prediksi vs Aktual")

    st.pyplot(fig)


# ============================================================
# MENU 4 — PREDIKSI MASA DEPAN
# ============================================================

elif menu == "Prediksi Masa Depan":

    if st.session_state.model is None:
        st.warning("Load model dulu")
        st.stop()

    if st.session_state.data is None:
        st.warning("Upload data dulu")
        st.stop()

    jumlah_hari = st.number_input(
        "Jumlah hari diprediksi",
        min_value=1,
        max_value=365,
        value=30
    )

    if st.button("Prediksi"):

        scaler = st.session_state.scaler
        model = st.session_state.model
        data = st.session_state.data

        data_scaled = scaler.transform(data[FITUR])

        last_data = data_scaled[-TIMESTEP:]

        future = []

        current = last_data.copy()

        for i in range(jumlah_hari):

            x = current.reshape(1, TIMESTEP, len(FITUR))

            pred = model.predict(x, verbose=0)

            new_row = current[-1].copy()
            new_row[2] = pred[0, 0]

            future.append(pred[0, 0])

            current = np.vstack([current[1:], new_row])

        # inverse scaling
        dummy = np.zeros((len(future), len(FITUR)))
        dummy[:, 2] = future

        future_inv = scaler.inverse_transform(dummy)[:, 2]

        st.write("Hasil prediksi:")
        st.write(future_inv)

        # plot
        fig, ax = plt.subplots()

        ax.plot(future_inv)
        ax.set_title("Prediksi Masa Depan")

        st.pyplot(fig)
