import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Ubah stroke jadi label
df["stroke"] = df["stroke"].replace({0: "Tidak", 1: "Ya"})

# Daftar fitur kategorikal yang ingin divisualisasikan
categorical_features = ["gender", "hypertension", "heart_disease", "ever_married", 
                        "work_type", "Residence_type", "smoking_status"]

# Judul
st.title("📊 Dashboard Prediksi Risiko Stroke")

# Visualisasi satu per satu
for feature in categorical_features:
    st.subheader(f"📌 Perbandingan Stroke berdasarkan {feature.replace('_', ' ').title()}")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x=feature, hue="stroke", palette="Set2", ax=ax)
    ax.set_ylabel("Jumlah")
    ax.set_xlabel(feature.replace("_", " ").title())
    ax.legend(title="Stroke")
    plt.xticks(rotation=30)
    st.pyplot(fig)
