# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# ===================== UI =====================
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
    st.subheader("Data Understanding")
    df = pd.read_excel("Dataset_Curah_Hujan.xlsx")
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], format="%d-%m-%Y")
    st.session_state.df = df
    st.dataframe(df)

elif menu == "Interpolasi Linear":
    df = st.session_state.df
    if df is not None:
        df = df[['TAVG', 'RH_AVG', 'RR']].copy()
        df = df.replace('-', np.nan)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.interpolate(method='linear')
        df = df.fillna(method='bfill').fillna(method='ffill')
        st.session_state.df = df
        st.dataframe(df)
    else:
        st.warning("Load dataset dulu.")

elif menu == "Normalisasi Data":
    df = st.session_state.df
    if df is not None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[['RR']])
        df['Normalisasi'] = scaled_data

        st.session_state.scaler = scaler
        st.session_state.scaled_data = scaled_data
        st.session_state.df = df

        st.dataframe(df[['RR', 'Normalisasi']])
    else:
        st.warning("Lakukan interpolasi dulu.")

elif menu == "Model LSTM":
    if st.button("Load Model"):
        model = load_model("model_ts_50_ep_100_Ir_0.01.h5")
        st.session_state.model = model
        st.success("Model berhasil di-load.")

elif menu == "Prediksi LSTM":
    if st.session_state.mode_
