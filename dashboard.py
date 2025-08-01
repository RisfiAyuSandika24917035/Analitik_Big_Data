import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Evaluasi Model Random Forest", layout="centered")
st.title("📊 Evaluasi Model Random Forest untuk Prediksi Stroke")

st.markdown("""
Dashboard ini mengevaluasi performa model Random Forest untuk memprediksi risiko stroke.
Unggah hasil prediksi data training dan testing dalam format CSV.
""")

# Upload file
st.subheader("📁 Upload File")
file_train = st.file_uploader("Upload file hasil prediksi data training (CSV)", type=["csv"], key="train")
file_test = st.file_uploader("Upload file hasil prediksi data testing (CSV)", type=["csv"], key="test")

if file_train and file_test:
    df_train = pd.read_csv(file_train)
    df_test = pd.read_csv(file_test)

    for df in [df_train, df_test]:
        df['stroke'] = df['stroke'].astype(int)
        df['prediction'] = df['prediction'].astype(int)

    # Akurasi
    st.subheader("✅ Akurasi Model")
    col1, col2 = st.columns(2)
    acc_train = accuracy_score(df_train['stroke'], df_train['prediction'])
    acc_test = accuracy_score(df_test['stroke'], df_test['prediction'])

    col1.metric("🎓 Akurasi Training", f"{acc_train*100:.2f}%")
    col2.metric("🧪 Akurasi Testing", f"{acc_test*100:.2f}%")

    # Confusion Matrix Training
    st.subheader("📌 Confusion Matrix - Training")
    cm_train = confusion_matrix(df_train['stroke'], df_train['prediction'])
    fig1, ax1 = plt.subplots()
    sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", ax=ax1)
    ax1.set_xlabel("Prediksi")
    ax1.set_ylabel("Aktual")
    ax1.set_title("Confusion Matrix - Training")
    st.pyplot(fig1)

    # Confusion Matrix Testing
    st.subheader("📌 Confusion Matrix - Testing")
    cm_test = confusion_matrix(df_test['stroke'], df_test['prediction'])
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Greens", ax=ax2)
    ax2.set_xlabel("Prediksi")
    ax2.set_ylabel("Aktual")
    ax2.set_title("Confusion Matrix - Testing")
    st.pyplot(fig2)

    # Classification Report Testing
    st.subheader("🧾 Classification Report - Data Testing")
    report = classification_report(df_test['stroke'], df_test['prediction'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}'}))

    # Visualisasi Data Aktual vs Prediksi
    st.subheader("📈 Distribusi Data Aktual vs Prediksi - Testing Set")
    count_actual = df_test['stroke'].value_counts().sort_index()
    count_pred = df_test['prediction'].value_counts().sort_index()

    df_line = pd.DataFrame({
        'Label': ['Tidak Stroke', 'Stroke'],
        'Actual': count_actual.values,
        'Predicted': count_pred.values
    })

    fig3, ax3 = plt.subplots()
    ax3.plot(df_line['Label'], df_line['Actual'], marker='o', label='Actual', color='blue')
    ax3.plot(df_line['Label'], df_line['Predicted'], marker='o', label='Predicted', color='orange')
