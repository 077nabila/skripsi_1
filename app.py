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

    uploaded_file = st.file_uploader(
        "Upload dataset Excel",
        type=["xlsx"]
    )

    if uploaded_file is not None:

        df = pd.read_excel(uploaded_file)

        # bersihkan nama kolom
        df.columns = df.columns.str.strip()

        # konversi tanggal (AMAN untuk BMKG)
        df["Tanggal"] = pd.to_datetime(
            df["Tanggal"],
            errors="coerce",
            dayfirst=True
        )

        df = df.sort_values("Tanggal")

        df = df.reset_index(drop=True)

        st.session_state.df = df

        st.success("Dataset berhasil di load")

        st.dataframe(df)

    else:

        st.warning("Upload dataset dulu")


# =========================
# INTERPOLASI LINEAR
# =========================

elif menu == "Interpolasi Linear":

    df = st.session_state.df

    if df is None:
        st.warning("Load dataset dulu")
        st.stop()

    fitur = ["RR", "Tn", "Tx"]

    df[fitur] = df[fitur].interpolate(method="linear")

    df[fitur] = df[fitur].bfill()

    df[fitur] = df[fitur].ffill()

    st.session_state.df = df

    st.success("Interpolasi selesai")

    st.dataframe(df)


# =========================
# NORMALISASI
# =========================

elif menu == "Normalisasi Data":

    df = st.session_state.df

    if df is None:
        st.warning("Interpolasi dulu")
        st.stop()

    fitur = ["TAVG", "RH_AVG", "RR"]

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(df[fitur])

    st.session_state.scaler = scaler

    st.session_state.scaled_data = scaled

    df_scaled = df.copy()

    df_scaled[fitur] = scaled

    st.session_state.df = df_scaled

    st.success("Normalisasi selesai")

    st.dataframe(df_scaled)


# =========================
# LOAD MODEL
# =========================

elif menu == "Model LSTM":

    if st.button("Load Model"):

        try:

            model = load_model(
                "model_splitdata_0.9_epochs_100_lr_0.01_ts_25.h5"
            )

            x_test = pd.read_csv(
                "xtest_splitdata_0.9_epochs_100_lr_0.01_ts_25.csv"
            ).values

            y_test = pd.read_csv(
                "ytest_splitdata_0.9_epochs_100_lr_0.01_ts_25.csv"
            ).values


            timestep = 25
            fitur = 3

            x_test = x_test.reshape(
                x_test.shape[0],
                timestep,
                fitur
            )

            st.session_state.model = model

            st.session_state.x_test = x_test

            st.session_state.y_test = y_test

            st.success("Model berhasil di load")

        except Exception as e:

            st.error(f"Gagal load model: {e}")


# =========================
# PREDIKSI
# =========================

elif menu == "Prediksi LSTM":

    model = st.session_state.model
    scaler = st.session_state.scaler
    x_test = st.session_state.x_test
    df = st.session_state.df

    if model is None or scaler is None:
        st.warning("Load model dan normalisasi dulu")
        st.stop()

    pred = model.predict(x_test)

    # inverse scaling prediksi
    dummy = np.zeros((len(pred), 3))

    dummy[:, 0] = pred.flatten()

    pred_inverse = scaler.inverse_transform(dummy)[:, 0]

    # inverse actual
    y_dummy = np.zeros((len(st.session_state.y_test), 3))

    y_dummy[:, 0] = st.session_state.y_test.flatten()

    actual = scaler.inverse_transform(y_dummy)[:, 0]

    tanggal = df["Tanggal"].iloc[-len(pred):].reset_index(drop=True)

    hasil = pd.DataFrame({

        "Tanggal": tanggal,
        "Aktual": actual,
        "Prediksi": pred_inverse

    })

    st.session_state.test_predictions = hasil

    st.subheader("Tabel Hasil Prediksi")

    st.dataframe(hasil)

    rmse = np.sqrt(np.mean((actual - pred_inverse)**2))

    st.write("RMSE:", rmse)

    fig, ax = plt.subplots()

    ax.plot(tanggal, actual, label="Aktual")

    ax.plot(tanggal, pred_inverse, label="Prediksi")

    ax.legend()

    ax.set_title("Perbandingan Aktual vs Prediksi")

    st.pyplot(fig)


# =========================
# IMPLEMENTASI
# =========================

elif menu == "Implementasi":

    model = st.session_state.model
    scaler = st.session_state.scaler
    x_test = st.session_state.x_test
    df = st.session_state.df

    if model is None or scaler is None:
        st.warning("Prediksi dulu")
        st.stop()

    n = st.selectbox(
        "Jumlah hari prediksi",
        [1, 7, 14, 30, 180, 365]
    )

    last = x_test[-1:]

    future = []

    for i in range(n):

        pred = model.predict(last)

        future.append(pred[0][0])

        new_row = last[:, -1, :].copy()

        new_row[0] = pred

        last = np.append(
            last[:, 1:, :],
            new_row.reshape(1, 1, 3),
            axis=1
        )

    future = np.array(future)

    dummy = np.zeros((n, 3))

    dummy[:, 0] = future

    future_inverse = scaler.inverse_transform(dummy)[:, 0]

    tanggal_future = pd.date_range(
        start=df["Tanggal"].iloc[-1],
        periods=n+1
    )[1:]

    hasil_future = pd.DataFrame({

        "Tanggal": tanggal_future,
        "Prediksi": future_inverse

    })

    st.subheader("Prediksi Masa Depan")

    st.dataframe(hasil_future)

    fig, ax = plt.subplots()

    ax.plot(tanggal_future, future_inverse)

    ax.set_title("Prediksi Curah Hujan Masa Depan")

    st.pyplot(fig)
