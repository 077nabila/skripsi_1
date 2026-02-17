# =========================
# IMPORT LIBRARY
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


# =========================
# CONFIG
# =========================

FITUR = ["TAVG", "RH_AVG", "RR"]
TIMESTEP = 25


# =========================
# TITLE
# =========================

st.title("PREDIKSI CURAH HUJAN MENGGUNAKAN LSTM")

menu = st.sidebar.radio(
    "Menu",
    [
        "Dataset",
        "Interpolasi Linear",
        "Normalisasi",
        "Load Model",
        "Prediksi Test",
        "Prediksi Masa Depan"
    ]
)


# =========================
# SESSION STATE
# =========================

if "df" not in st.session_state:
    st.session_state.df = None

if "scaler" not in st.session_state:
    st.session_state.scaler = None

if "scaled_data" not in st.session_state:
    st.session_state.scaled_data = None

if "model" not in st.session_state:
    st.session_state.model = None

if "x_test" not in st.session_state:
    st.session_state.x_test = None

if "y_test" not in st.session_state:
    st.session_state.y_test = None


# =========================
# MENU 1 — DATASET
# =========================

if menu == "Dataset":

    df = pd.read_excel("dataset_skripsi.xlsx")

    # bersihkan nama kolom
    df.columns = df.columns.str.strip()

    # convert tanggal aman
    df["Tanggal"] = pd.to_datetime(
        df["Tanggal"],
        errors="coerce"
    )

    df = df.dropna(subset=["Tanggal"])

    # convert fitur ke numerik
    df[FITUR] = df[FITUR].apply(
        pd.to_numeric,
        errors="coerce"
    )

    df = df.reset_index(drop=True)

    st.session_state.df = df

    st.write("Dataset:")
    st.dataframe(df)

    st.success("Dataset berhasil load")


# =========================
# MENU 2 — INTERPOLASI
# =========================

elif menu == "Interpolasi Linear":

    df = st.session_state.df

    if df is None:
        st.error("Load Dataset dulu")
        st.stop()

    df[FITUR] = df[FITUR].interpolate(
        method="linear"
    )

    df[FITUR] = df[FITUR].bfill()
    df[FITUR] = df[FITUR].ffill()

    st.session_state.df = df

    st.write(df)

    st.success("Interpolasi berhasil")


# =========================
# MENU 3 — NORMALISASI
# =========================

elif menu == "Normalisasi":

    df = st.session_state.df

    if df is None:
        st.error("Interpolasi dulu")
        st.stop()

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(
        df[FITUR]
    )

    st.session_state.scaler = scaler
    st.session_state.scaled_data = scaled

    df_scaled = df.copy()
    df_scaled[FITUR] = scaled

    st.session_state.df = df_scaled

    st.write(df_scaled)

    st.success("Normalisasi berhasil")


# =========================
# MENU 4 — LOAD MODEL
# =========================

elif menu == "Load Model":

    if st.button("Load Model"):

        model = load_model(
            "model_splitdata_0.9_epochs_100_lr_0.01_ts_25.h5"
        )

        x_test = pd.read_csv(
            "xtest_splitdata_0.9_epochs_100_lr_0.01_ts_25.csv"
        ).values

        y_test = pd.read_csv(
            "ytest_splitdata_0.9_epochs_100_lr_0.01_ts_25.csv"
        ).values

        x_test = x_test.reshape(
            x_test.shape[0],
            TIMESTEP,
            len(FITUR)
        )

        st.session_state.model = model
        st.session_state.x_test = x_test
        st.session_state.y_test = y_test

        st.success("Model berhasil di load")


# =========================
# MENU 5 — PREDIKSI TEST
# =========================

elif menu == "Prediksi Test":

    model = st.session_state.model
    scaler = st.session_state.scaler
    x_test = st.session_state.x_test
    y_test = st.session_state.y_test
    df = st.session_state.df

    if model is None:
        st.error("Load model dulu")
        st.stop()

    pred = model.predict(x_test)

    # inverse scaling
    dummy = np.zeros((len(pred), len(FITUR)))
    dummy[:, 2] = pred.flatten()

    pred_inverse = scaler.inverse_transform(dummy)[:, 2]

    # actual
    dummy_y = np.zeros((len(y_test), len(FITUR)))
    dummy_y[:, 2] = y_test.flatten()

    actual = scaler.inverse_transform(dummy_y)[:, 2]

    tanggal = df["Tanggal"].iloc[-len(pred):]

    hasil = pd.DataFrame({

        "Tanggal": tanggal.values,
        "Aktual RR": actual,
        "Prediksi RR": pred_inverse

    })

    st.dataframe(hasil)

    # RMSE
    rmse = np.sqrt(
        np.mean((actual - pred_inverse)**2)
    )

    st.write("RMSE:", rmse)

    # grafik
    fig, ax = plt.subplots()

    ax.plot(tanggal, actual, label="Aktual")
    ax.plot(tanggal, pred_inverse, label="Prediksi")

    ax.legend()

    st.pyplot(fig)


# =========================
# MENU 6 — IMPLEMENTASI
# =========================

elif menu == "Prediksi Masa Depan":

    model = st.session_state.model
    scaler = st.session_state.scaler
    x_test = st.session_state.x_test
    df = st.session_state.df

    if model is None:
        st.error("Load model dulu")
        st.stop()

    n = st.selectbox(
        "Jumlah hari prediksi",
        [1, 7, 14, 30, 90, 180, 365]
    )

    last = x_test[-1:]

    future = []

    for i in range(n):

        pred = model.predict(last)

        future.append(pred[0][0])

        new_row = last[:, -1, :].copy()

        new_row[0][2] = pred

        last = np.append(
            last[:, 1:, :],
            new_row.reshape(1, 1, len(FITUR)),
            axis=1
        )

    future = np.array(future)

    dummy = np.zeros((n, len(FITUR)))

    dummy[:, 2] = future

    future_inverse = scaler.inverse_transform(dummy)[:, 2]

    tanggal_future = pd.date_range(
        df["Tanggal"].iloc[-1],
        periods=n+1
    )[1:]

    hasil_future = pd.DataFrame({

        "Tanggal": tanggal_future,
        "Prediksi RR": future_inverse

    })

    st.dataframe(hasil_future)

    fig, ax = plt.subplots()

    ax.plot(
        tanggal_future,
        future_inverse
    )

    st.pyplot(fig)
