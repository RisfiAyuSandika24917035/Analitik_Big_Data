import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Judul dashboard
st.title("📊 Dashboard Prediksi Risiko Stroke")

# Load data
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

st.subheader("🔎 Sekilas Data")
st.dataframe(df.head())

# Bar chart: stroke vs non-stroke per kategori smoking_status
st.subheader("🚬 Perbandingan Stroke berdasarkan Status Merokok")
fig1, ax1 = plt.subplots()
sns.countplot(data=df, x="smoking_status", hue="stroke", palette="Set2", ax=ax1)
ax1.set_ylabel("Jumlah")
ax1.set_xlabel("Status Merokok")
ax1.legend(title="Stroke", labels=["Tidak", "Ya"])
st.pyplot(fig1)

# Pie chart: stroke berdasarkan kelompok usia (>60 vs <=60)
st.subheader("🎂 Proporsi Stroke Berdasarkan Kelompok Usia")
df["age_group"] = df["age"].apply(lambda x: ">60" if x > 60 else "<=60")
age_stroke = df[df["stroke"] == 1]["age_group"].value_counts()
fig2, ax2 = plt.subplots()
ax2.pie(age_stroke, labels=age_stroke.index, autopct="%1.1f%%", startangle=90, colors=["#66c2a5", "#fc8d62"])
ax2.axis("equal")
st.pyplot(fig2)

# Heatmap korelasi
st.subheader("📈 Korelasi Antar Variabel")
correlation = df[["age", "hypertension", "heart_disease", "avg_glucose_level", "stroke"]].corr()
fig3, ax3 = plt.subplots()
sns.heatmap(correlation, annot=True, cmap="coolwarm", ax=ax3)
st.pyplot(fig3)

# Upload prediksi (jika punya)
st.subheader("📄 Hasil Prediksi Model")
uploaded_file = st.file_uploader("Upload file CSV hasil prediksi", type=["csv"])
if uploaded_file:
    pred_df = pd.read_csv(uploaded_file)
    st.write("Contoh hasil prediksi:")
    st.dataframe(pred_df.head())
