import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.title("Prediksi Curah Hujan LSTM")

# load dataset
df = pd.read_excel("Dataset_Curah_Hujan.xlsx")

# tampilkan kolom
st.write("Kolom dataset:", df.columns.tolist())

# ubah tanggal
df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
df = df.dropna(subset=["Tanggal"])

# FITUR (SESUIKAN DENGAN EXCEL)
fitur = ["Curah Hujan", "Suhu", "Kelembaban"]

# UBAH KE NUMERIK (INI BAGIAN PALING PENTING)
df[fitur] = df[fitur].apply(pd.to_numeric, errors="coerce")

# interpolasi linear
df[fitur] = df[fitur].interpolate(method="linear")

# hapus jika masih ada NaN
df = df.dropna()

# scaling
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[fitur])

# load model
model = load_model("model_ts_50_ep_100_Ir_0.01.h5")

# fungsi sequence
def buat_sequence(data, timestep=50):
    X = []
    for i in range(len(data) - timestep):
        X.append(data[i:i+timestep])
    return np.array(X)

X = buat_sequence(data_scaled)

# prediksi
pred_scaled = model.predict(X)

# inverse scaling
dummy = np.zeros((len(pred_scaled), len(fitur)))
dummy[:,0] = pred_scaled[:,0]

prediksi = scaler.inverse_transform(dummy)[:,0]

# tampilkan hasil
hasil = pd.DataFrame({
    "Tanggal": df["Tanggal"].iloc[50:].values,
    "Prediksi Curah Hujan": prediksi
})

st.write(hasil)
