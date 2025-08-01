import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Judul dashboard
st.title("📊 Dashboard Prediksi Risiko Stroke")

# Load data
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Tampilan data awal
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

# Bar chart: stroke vs non-stroke per kategori hipertensi
st.subheader("💉 Perbandingan Stroke berdasarkan Riwayat Hipertensi")
df["hypertension_label"] = df["hypertension"].map({0: "Tidak", 1: "Ya"})
fig3, ax3 = plt.subplots()
sns.countplot(data=df, x="hypertension_label", hue="stroke", palette="Set2", ax=ax3)
ax3.set_ylabel("Jumlah")
ax3.set_xlabel("Hipertensi")
ax3.legend(title="Stroke", labels=["Tidak", "Ya"])
st.pyplot(fig3)

# Bar chart: stroke vs non-stroke per kategori penyakit jantung
st.subheader("❤️ Perbandingan Stroke berdasarkan Riwayat Penyakit Jantung")
df["heart_disease_label"] = df["heart_disease"].map({0: "Tidak", 1: "Ya"})
fig4, ax4 = plt.subplots()
sns.countplot(data=df, x="heart_disease_label", hue="stroke", palette="Set2", ax=ax4)
ax4.set_ylabel("Jumlah")
ax4.set_xlabel("Penyakit Jantung")
ax4.legend(title="Stroke", labels=["Tidak", "Ya"])
st.pyplot(fig4)

# Bar chart: stroke vs non-stroke per kategori glukosa
st.subheader("🩸 Perbandingan Stroke berdasarkan Level Glukosa")
# Kelompokkan glukosa: <=100 (Normal), 100-140 (Pre-diabetes), >140 (Diabetes)
def categorize_glucose(value):
    if value <= 100:
        return "Normal"
    elif value <= 140:
        return "Pre-diabetes"
    else:
        return "Diabetes"

df["glucose_category"] = df["avg_glucose_level"].apply(categorize_glucose)
fig5, ax5 = plt.subplots()
sns.countplot(data=df, x="glucose_category", hue="stroke", palette="Set2", ax=ax5,
              order=["Normal", "Pre-diabetes", "Diabetes"])
ax5.set_ylabel("Jumlah")
ax5.set_xlabel("Kategori Glukosa")
ax5.legend(title="Stroke", labels=["Tidak", "Ya"])
st.pyplot(fig5)

# Korelasi antar variabel
st.subheader("📈 Korelasi Antar Variabel")
correlation = df[["age", "hypertension", "heart_disease", "avg_glucose_level", "stroke"]].corr()
fig6, ax6 = plt.subplots()
sns.heatmap(correlation, annot=True, cmap="coolwarm", ax=ax6)
st.pyplot(fig6)

# Upload prediksi (jika ada)
st.subheader("📄 Hasil Prediksi Model")
uploaded_file = st.file_uploader("Upload file CSV hasil prediksi", type=["csv"])
if uploaded_file:
    pred_df = pd.read_csv(uploaded_file)
    st.write("Contoh hasil prediksi:")
    st.dataframe(pred_df.head())
