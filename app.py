# ==========================================
# PREDIKSI CURAH HUJAN MENGGUNAKAN LSTM
# STREAMLIT FINAL VERSION - NO ERROR
# ==========================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


# ==========================================
# CONFIG
# ==========================================

st.set_page_config(
    page_title="Prediksi Curah Hujan LSTM",
    layout="wide"
)

st.title("PREDIKSI CURAH HUJAN MENGGUNAKAN LSTM")

menu = st.sidebar.radio(
    "Menu",
    [
        "1. Upload Dataset",
        "2. Interpolasi Linear",
        "3. Normalisasi",
        "4. Load Model",
        "5. Prediksi"
    ]
)


# ==========================================
# SESSION STATE
# ==========================================

if "df" not in st.session_state:
    st.session_state.df = None

if "scaler" not in st.session_state:
    st.session_state.scaler = None

if "scaled_data" not in st.session_state:
    st.session_state.scaled_data = None

if "model" not in st.session_state:
    st.session_state.model = None

if "prediksi" not in st.session_state:
    st.session_state.prediksi = None


# ==========================================
# MENU 1: UPLOAD DATASET
# ==========================================

if menu == "1. Upload Dataset":

    st.subheader("Upload Dataset Excel")

    uploaded_file = st.file_uploader(
        "Upload file Excel",
        type=["xlsx"]
    )

    if uploaded_file:

        df = pd.read_excel(uploaded_file)

        # pastikan kolom tanggal ada
        if "Tanggal" in df.columns:
            df["Tanggal"] = pd.to_datetime(
                df["Tanggal"],
                errors="coerce"
            )

        st.session_state.df = df

        st.success("Dataset berhasil diupload")

        st.dataframe(df, use_container_width=True)

    else:
        st.info("Silakan upload dataset")


# ==========================================
# MENU 2: INTERPOLASI
# ==========================================

elif menu == "2. Interpolasi Linear":

    df = st.session_state.df

    if df is not None:

        df_interp = df.copy()

        kolom = ["TAVG", "RH_AVG", "RR"]

        df_interp[kolom] = (
            df_interp[kolom]
            .replace("-", np.nan)
            .apply(pd.to_numeric, errors="coerce")
            .interpolate(method="linear")
            .bfill()
            .ffill()
        )

        st.session_state.df = df_interp

        st.success("Interpolasi berhasil")

        st.dataframe(df_interp, use_container_width=True)

    else:
        st.warning("Upload dataset terlebih dahulu")


# ==========================================
# MENU 3: NORMALISASI
# ==========================================

elif menu == "3. Normalisasi":

    df = st.session_state.df

    if df is not None:

        scaler = MinMaxScaler()

        scaled = scaler.fit_transform(df[["RR"]])

        df["RR_Normalisasi"] = scaled

        st.session_state.scaler = scaler
        st.session_state.scaled_data = scaled
        st.session_state.df = df

        st.success("Normalisasi berhasil")

        st.dataframe(
            df[["Tanggal", "RR", "RR_Normalisasi"]],
            use_container_width=True
        )

    else:
        st.warning("Lakukan interpolasi dulu")


# ==========================================
# MENU 4: LOAD MODEL
# ==========================================

elif menu == "4. Load Model":

    if st.button("Load Model LSTM"):

        try:

            model = load_model(
                "model_ts_50_ep_100_lr_0.01.h5"
            )

            st.session_state.model = model

            st.success("Model berhasil di-load")

        except Exception as e:

            st.error(f"Gagal load model: {e}")


# ==========================================
# MENU 5: PREDIKSI
# ==========================================

elif menu == "5. Prediksi":

    df = st.session_state.df
    model = st.session_state.model

    if df is None:
        st.warning("Upload dataset dulu")
        st.stop()

    if model is None:
        st.warning("Load model dulu")
        st.stop()

    try:

        preds = pd.read_csv(
            "prediksi_ts_50_ep_100_lr_0.01.csv"
        ).values.flatten()

        st.session_state.prediksi = preds

        tanggal = df["Tanggal"].iloc[-len(preds):].reset_index(drop=True)
        aktual = df["RR"].iloc[-len(preds):].reset_index(drop=True)

        hasil = pd.DataFrame({
            "Tanggal": tanggal,
            "Aktual": aktual,
            "Prediksi": preds
        })

        st.subheader("Hasil Prediksi")

        st.dataframe(hasil, use_container_width=True)


        # ==================================
        # HITUNG RMSE
        # ==================================

        rmse = np.sqrt(np.mean((aktual - preds) ** 2))

        st.success(f"RMSE: {rmse:.4f}")


        # ==================================
        # PLOT
        # ==================================

        fig, ax = plt.subplots(figsize=(12,5))

        ax.plot(
            hasil["Tanggal"],
            hasil["Aktual"],
            label="Aktual"
        )

        ax.plot(
            hasil["Tanggal"],
            hasil["Prediksi"],
            label="Prediksi"
        )

        ax.set_title("Grafik Prediksi Curah Hujan")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Curah Hujan")
        ax.legend()

        st.pyplot(fig)


        # ==================================
        # DOWNLOAD HASIL
        # ==================================

        csv = hasil.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Hasil Prediksi",
            csv,
            "hasil_prediksi.csv",
            "text/csv"
        )


    except Exception as e:

        st.error(f"Gagal prediksi: {e}")
