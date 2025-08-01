import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Judul
st.title("📊 Dashboard Analisis & Prediksi Risiko Stroke")

# Load data utama
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Sekilas data
st.subheader("🔍 Sekilas Data")
st.dataframe(df.head())

# Visualisasi faktor-faktor
st.subheader("🚬 Visualisasi Faktor Risiko")
# -> kode untuk bar chart smoking_status, pie usia, bar hipertensi, jantung, glukosa

# Heatmap (jika tetap ingin dipakai)
st.subheader("📈 Korelasi Antar Variabel")
# -> heatmap

# Data Training
st.subheader("🧠 Distribusi Data Training")
# -> tampilkan berapa yang stroke vs tidak (training set)

# Upload & Prediksi
st.subheader("📄 Hasil Prediksi Model")
uploaded_file = st.file_uploader("Upload file CSV hasil prediksi", type=["csv"])
if uploaded_file:
    pred_df = pd.read_csv(uploaded_file)
    st.dataframe(pred_df.head())
    st.subheader("📊 Distribusi Hasil Prediksi")
    # -> tampilkan bar chart prediksi

# Insight akhir
st.subheader("📝 Insight Penting")
st.markdown("""
- Mayoritas stroke terjadi pada usia di atas 60 tahun.
- Faktor-faktor seperti hipertensi, diabetes, dan penyakit jantung punya pengaruh meski korelasinya lemah.
- Model prediksi memiliki akurasi **94.33%**, cukup baik untuk kasus imbalance.
""")
