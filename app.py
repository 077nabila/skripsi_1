# ===================== IMPORT =====================
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# ===================== CONFIG =====================
st.set_page_config(page_title="Prediksi Curah Hujan LSTM", layout="wide")
st.title("PREDIKSI CURAH HUJAN MENGGUNAKAN LSTM")
st.sidebar.title("Main Menu")

menu = st.sidebar.radio(
    "Go to",
    ["Dataset", "Interpolasi Linear", "Normalisasi Data", "Model LSTM", "Prediksi LSTM"]
)

# ===================== SESSION STATE =====================
if "df" not in st.session_state:
    st.session_state.df = None

if "scaler" not in st.session_state:
    st.session_state.scaler = None

if "scaled_data" not in st.session_state:
    st.session_state.scaled_data = None

if "model" not in st.session_state:
    st.session_state.model = None

if "test_predictions" not in st.session_state:
    st.session_state.test_predictions = None

# ===================== PATH =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model_ts_50_ep_100_lr_0.01.h5")
CSV_PATH = os.path.join(BASE_DIR, "prediksi_ts_50_ep_100_lr_0.01.csv")

# ===================== MENU DATASET =====================
if menu == "Dataset":

    st.subheader("Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload file Excel (.xlsx)", type="xlsx"
    )

    if uploaded_file is not None:

        df = pd.read_excel(uploaded_file)

        # pastikan kolom tanggal benar
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")

        # urutkan tanggal
        df = df.sort_values("Tanggal").reset_index(drop=True)

        st.session_state.df = df

        st.success("Dataset berhasil diupload")

        st.dataframe(df, use_container_width=True)

    else:
        st.info("Silakan upload dataset")


# ===================== MENU INTERPOLASI =====================
elif menu == "Interpolasi Linear":

    df = st.session_state.df

    if df is not None:

        df_interp = df.copy()

        df_interp[['TAVG', 'RH_AVG', 'RR']] = (
            df_interp[['TAVG', 'RH_AVG', 'RR']]
            .replace('-', np.nan)
            .apply(pd.to_numeric, errors='coerce')
            .interpolate(method="linear")
            .fillna(method="bfill")
            .fillna(method="ffill")
        )

        st.session_state.df = df_interp

        st.success("Interpolasi berhasil")

        st.dataframe(df_interp, use_container_width=True)

    else:
        st.warning("Upload dataset terlebih dahulu")


# ===================== MENU NORMALISASI =====================
elif menu == "Normalisasi Data":

    df = st.session_state.df

    if df is not None:

        scaler = MinMaxScaler()

        scaled = scaler.fit_transform(df[['RR']])

        df["Normalisasi"] = scaled

        st.session_state.df = df
        st.session_state.scaler = scaler

        st.success("Normalisasi berhasil")

        st.dataframe(
            df[["Tanggal", "RR", "Normalisasi"]],
            use_container_width=True
        )

    else:
        st.warning("Lakukan interpolasi terlebih dahulu")


# ===================== MENU LOAD MODEL =====================
elif menu == "Model LSTM":

    st.subheader("Load Model")

    if st.button("Load Model"):

        try:

            model = load_model(MODEL_PATH)

            st.session_state.model = model

            st.success("Model berhasil di-load")

        except Exception as e:

            st.error(f"Gagal load model: {e}")


# ===================== MENU PREDIKSI =====================
elif menu == "Prediksi LSTM":

    df = st.session_state.df
    model = st.session_state.model

    if df is not None and model is not None:

        try:

            preds = pd.read_csv(CSV_PATH).values.flatten()

            st.session_state.test_predictions = preds

            # ambil data sesuai panjang prediksi
            tanggal = df["Tanggal"].iloc[-len(preds):].values
            aktual = df["RR"].iloc[-len(preds):].values

            # buat dataframe hasil
            hasil = pd.DataFrame({
                "Tanggal": tanggal,
                "Aktual": aktual,
                "Prediksi": preds
            })

            st.subheader("Tabel Hasil Prediksi")

            st.dataframe(hasil, use_container_width=True)

            # ===================== HITUNG RMSE =====================
            rmse = np.sqrt(np.mean((aktual - preds) ** 2))

            st.success(f"RMSE : {rmse:.4f}")

            # ===================== GRAFIK =====================
            st.subheader("Grafik Perbandingan Aktual vs Prediksi")

            fig, ax = plt.subplots(figsize=(12,5))

            ax.plot(
                hasil["Tanggal"],
                hasil["Aktual"],
                label="Aktual",
                marker="o"
            )

            ax.plot(
                hasil["Tanggal"],
                hasil["Prediksi"],
                label="Prediksi",
                marker="o"
            )

            ax.set_xlabel("Tanggal")
            ax.set_ylabel("Curah Hujan (mm)")
            ax.set_title("Perbandingan Curah Hujan Aktual vs Prediksi")
            ax.legend()
            ax.grid(True)

            st.pyplot(fig)

        except Exception as e:

            st.error(f"Gagal prediksi: {e}")

    else:

        st.warning("Pastikan dataset dan model sudah di-load")
