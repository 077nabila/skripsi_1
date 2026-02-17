# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

# Set the title of the app
st.title("PREDIKSI CURAH HUJAN MENGGUNAKAN LSTM")
st.sidebar.title("Main Menu")

menu = st.sidebar.radio(
    "Go to",
    ["Dataset", "Interpolasi Linear", "Normalisasi Data", "Model LSTM", "Prediksi LSTM", "Implementasi"]
)

# Init session_state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'scaled_data' not in st.session_state:
    st.session_state.scaled_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'x_test' not in st.session_state:
    st.session_state.x_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'test_predictions' not in st.session_state:
    st.session_state.test_predictions = None

# ===================== MENU =====================

if menu == "Dataset":
    st.subheader("Data Understanding")

    st.write(
        "Dataset ini berisi data curah hujan dari BMKG Stasiun Meteorologi Maritim Tanjung Perak "
        "(1 Jan 2019 – 31 Agt 2023)."
    )

    df = pd.read_excel("Dataset_Curah_Hujan.xlsx")
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], format="%d-%m-%Y")
    st.session_state.df = df

    st.write("Dataset Curah Hujan:")
    st.dataframe(df)

elif menu == "Interpolasi Linear":
    df = st.session_state.df

    if df is not None:
        df = d
