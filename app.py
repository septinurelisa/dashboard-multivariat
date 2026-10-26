import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

# KONFIGURASI HALAMAN

st.set_page_config(
    page_title="Iris Dashboard",
    page_icon="📊",
    layout="wide"
)

# JUDUL DASHBOARD

st.title("My Iris Dataset Dashboard 🌺")

st.write(
    """
    Dashboard ini digunakan untuk mengeksplorasi data multivariat
    pada dataset Iris, meliputi korelasi, uji normalitas, dan
    deteksi outlier melalui visualisasi data interaktif.
    """
)

st.divider()

# MEMUAT DATASET


@st.cache_data
def load_data():
    return px.data.iris().drop(columns="species_id")


df = load_data()

# Mendefinisikan atribut dan spesies
attributes = df.columns[:-1].tolist()
species = df["species"].unique().tolist()

# INFORMASI DATASET

with st.expander("About the Iris Dataset"):

    st.header("About the Iris Dataset")

    st.write(
        """
        The Iris dataset consists of 150 observations
        from three species of iris flowers: Setosa,
        Versicolor, and Virginica.

        Each observation contains four numerical
        attributes: sepal length, sepal width,
        petal length, and petal width.

        The dataset is commonly used to demonstrate
        exploratory data analysis, data visualization,
        and classification.
        """
    )

# SIDEBAR: PILIHAN VARIABEL DAN SPESIES

with st.sidebar:

    st.header("Filters")

    # Pilihan atribut
    selected_attribute = st.selectbox(
        "Select Attribute:",
        attributes,
        index=0
    )

    # Pilihan spesies
    selected_species = st.multiselect(
        "Select Species:",
        species,
        default=species,
        placeholder="Filter by species"
    )

# Memfilter data berdasarkan spesies yang dipilih
filtered_df = df[
    df["species"].isin(selected_species)
]

if filtered_df.empty:
    st.warning("Pilih minimal satu spesies pada sidebar.")
    st.stop()

num = filtered_df[attributes]

# 1. DATA

st.header("1. Data")

st.dataframe(filtered_df.head())

st.dataframe(num.describe().round(2))

# 2. KORELASI

st.header("2. Korelasi")

if len(filtered_df) >= 3:

    heatmap_fig = px.imshow(
        num.corr().round(2),
        text_auto=True,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap"
    )

    st.plotly_chart(heatmap_fig)

else:

    st.info("Pilih minimal tiga data untuk melihat korelasi.")

# 3. SCATTERPLOT MATRIX

st.header("3. Scatterplot Matrix")

pairplot_fig = px.scatter_matrix(
    filtered_df,
    dimensions=attributes,
    color="species",
    opacity=0.6,
    title="Pairwise Scatter Plot"
)

pairplot_fig.update_traces(
    diagonal_visible=False
)

pairplot_fig.update_layout(
    height=750
)

st.plotly_chart(pairplot_fig)

# 4. UJI NORMALITAS

st.header("4. Uji Normalitas (Shapiro-Wilk)")

st.write(
    """
    H0: data berdistribusi normal. H0 ditolak jika p-value
    kurang dari 0,05.
    """
)

if len(filtered_df) >= 3:

    hasil = []
    for c in num.columns:
        stat, p = stats.shapiro(num[c].dropna()[:5000])
        hasil.append([c, stat, p, "Normal" if p > 0.05 else "Tidak normal"])

    st.dataframe(
        pd.DataFrame(
            hasil,
            columns=["Variabel", "Statistik", "p-value", "Kesimpulan"]
        )
    )

else:

    st.info("Uji normalitas membutuhkan minimal tiga data.")

hist_fig = px.histogram(
    filtered_df,
    x=selected_attribute,
    color="species",
    marginal="box",
    barmode="overlay",
    opacity=0.7,
    title=f"Histogram of {selected_attribute}"
)

hist_fig.update_layout(
    hovermode="x unified"
)

st.plotly_chart(hist_fig)

# 5. DETEKSI OUTLIER

st.header("5. Deteksi Outlier (IQR)")

show_points = st.checkbox(
    "Show individual points",
    value=True
)

points = "all" if show_points else False

for c in num.columns:
    q1, q3 = num[c].quantile(0.25), num[c].quantile(0.75)
    iqr = q3 - q1
    out = num[(num[c] < q1 - 1.5 * iqr) | (num[c] > q3 + 1.5 * iqr)]
    st.write(f"{c}: {len(out)} outlier")

    violin_fig = px.violin(
        filtered_df,
        x="species",
        y=c,
        color="species",
        box=True,
        points=points,
        title=f"Violin Plot of {c}"
    )

    st.plotly_chart(violin_fig)

st.divider()

st.caption(
    "Praktikum Eksplorasi dan Visualisasi Data Multivariat"
)
