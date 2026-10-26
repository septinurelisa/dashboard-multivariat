import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

st.title("Dashboard Eksplorasi Data Multivariat")

file = st.file_uploader("Upload file CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
    num = df.select_dtypes(include=np.number)

    st.subheader("1. Data")
    st.dataframe(df.head())
    st.write(df.describe())

    st.subheader("2. Korelasi")
    fig, ax = plt.subplots()
    sns.heatmap(num.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("3. Scatterplot Matrix")
    st.pyplot(sns.pairplot(num).figure)

    st.subheader("4. Uji Normalitas (Shapiro-Wilk)")
    hasil = []
    for c in num.columns:
        stat, p = stats.shapiro(num[c].dropna()[:5000])
        hasil.append([c, stat, p, "Normal" if p > 0.05 else "Tidak normal"])
    st.dataframe(pd.DataFrame(hasil, columns=["Variabel", "Statistik", "p-value", "Kesimpulan"]))

    st.subheader("5. Deteksi Outlier (IQR)")
    for c in num.columns:
        q1, q3 = num[c].quantile(0.25), num[c].quantile(0.75)
        iqr = q3 - q1
        out = num[(num[c] < q1 - 1.5 * iqr) | (num[c] > q3 + 1.5 * iqr)]
        st.write(f"{c}: {len(out)} outlier")
        fig, ax = plt.subplots()
        sns.boxplot(x=num[c], ax=ax)
        st.pyplot(fig)
