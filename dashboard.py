import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.title("Dashboard Evaluasi Model Random Forest")

st.subheader("Upload hasil prediksi data training")
file_train = st.file_uploader("Upload file pred_train.csv", type=["csv"])

st.subheader("Upload hasil prediksi data testing")
file_test = st.file_uploader("Upload file pred_test.csv", type=["csv"])

if file_train and file_test:
    df_train = pd.read_csv(file_train)
    df_test = pd.read_csv(file_test)

    # Pastikan kolom bertipe integer
    for df in [df_train, df_test]:
        df['stroke'] = df['stroke'].astype(int)
        df['prediction'] = df['prediction'].astype(int)

    # Hitung akurasi
    acc_train = accuracy_score(df_train['stroke'], df_train['prediction'])
    acc_test = accuracy_score(df_test['stroke'], df_test['prediction'])

    st.metric("Accuracy Training", f"{acc_train*100:.2f}%")
    st.metric("Accuracy Testing", f"{acc_test*100:.2f}%")

    # Confusion Matrix Training
    st.subheader("Confusion Matrix Training")
    cm_train = confusion_matrix(df_train['stroke'], df_train['prediction'])
    fig, ax = plt.subplots()
    sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Training Set")
    st.pyplot(fig)

    # Confusion Matrix Testing
    st.subheader("Confusion Matrix Testing")
    cm_test = confusion_matrix(df_test['stroke'], df_test['prediction'])
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Greens", ax=ax2)
    ax2.set_title("Testing Set")
    st.pyplot(fig2)

    # Classification report Testing
    st.subheader("Classification Report Testing")
    report_test = classification_report(df_test['stroke'], df_test['prediction'], output_dict=True)
    st.dataframe(pd.DataFrame(report_test).transpose())
