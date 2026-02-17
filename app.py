# app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# ===================== UI =====================
st.set_page_config(page_title="Prediksi Curah Hujan LSTM", layout="wide")
st.title("PREDIKSI CURAH HUJAN MENGGUNAKAN LSTM")
st.sidebar.title("Main Menu")

menu = st.sidebar.radio(
    "Go to",
    ["Dataset", "Interpolasi Linear", "Normalisasi Data", "Model LSTM", "Prediksi LSTM"]
)

# ===================== SESSION STATE =====================
if 'df' not in st.session_state:
    st.session_state.df = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'scaled_data' not in st.session_state:
    st.session_state.scaled_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'test_predictions' not in st.session_state:
    st.session_state.test_predictions = None

# ===================== MENU =====================
if menu == "Dataset":
    st.subheader("Upload & Data Understanding")

    # Upload file Excel supaya aman di Streamlit Cloud
    uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type="xlsx")

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        # Parse tanggal aman, otomatis deteksi format
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors='coerce')
        st.session_state.df = df
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("Silakan upload file Excel terlebih dahulu.")

elif menu == "Interpolasi Linear":
    df = st.session_state.df
    if df is not None:
        df_interp = df[['TAVG', 'RH_AVG', 'RR']].copy()
        df_interp = df_interp.replace('-', np.nan)
        df_interp = df_interp.apply(pd.to_numeric, errors='coerce')
        df_interp = df_interp.interpolate(method='linear')
        df_interp = df_interp.fillna(method='bfill').fillna(method='ffill')
        st.session_state.df = df_interp
        st.dataframe(df_interp, use_container_width=True)
    else:
        st.warning("Upload dataset dulu.")

elif menu == "Normalisasi Data":
    df = st.session_state.df
    if df is not None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[['RR']])
        df['Normalisasi'] = scaled_data

        st.session_state.scaler = scaler
        st.session_state.scaled_data = scaled_data
        st.session_state.df = df

        st.dataframe(df[['RR', 'Normalisasi']], use_container_width=True)
    else:
        st.warning("Lakukan interpolasi dulu.")

elif menu == "Model LSTM":
    if st.button("Load Model"):
        try:
            model = load_model("model_ts_50_ep_100_lr_0.01.h5")
            st.session_state.model = model
            st.success("Model LSTM berhasil di-load.")
        except Exception as e:
            st.error(f"Gagal load model: {e}")

elif menu == "Prediksi LSTM":
    if st.session_state.model is not None and st.session_state.df is not None:
        try:
            preds = pd.read_csv("prediksi_ts_50_ep_100_lr_0.01.csv").values.flatten()
            st.session_state.test_predictions = preds

            df = st.session_state.df

            st.subheader("Hasil Prediksi")
            st.write(preds)

            # Hitung RMSE
            rmse = np.sqrt(np.mean((df['RR'].iloc[-len(preds):] - preds) ** 2))
            st.write("RMSE:", rmse)

            # Plot hasil prediksi vs asli
            plt.figure(figsize=(12, 5))
            plt.plot(df['Tanggal'].iloc[-len(preds):], df['RR'].iloc[-len(preds):], label='Asli')
            plt.plot(df['Tanggal'].iloc[-len(preds):], preds, label='Prediksi')
            plt.xlabel("Tanggal")
            plt.ylabel("Curah Hujan")
            plt.title("Prediksi Curah Hujan LSTM")
            plt.legend()
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Gagal prediksi: {e}")
    else:
        st.warning("Load model & dataset dulu.")




