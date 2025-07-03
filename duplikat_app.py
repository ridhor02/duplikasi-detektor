import streamlit as st
import pandas as pd
from rapidfuzz import fuzz
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from io import StringIO, BytesIO
import base64

st.set_page_config(page_title="Deteksi Duplikat Lite", layout="wide")
st.title("🔍 Deteksi Duplikasi (Versi Cepat)")

st.markdown("Deteksi cepat duplikasi teks menggunakan **TF-IDF + DBSCAN** atau **RapidFuzz Ratio**, tanpa analisis tambahan.")

# Sidebar input
with st.sidebar:
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    method = st.radio("Metode Deteksi", ["TF-IDF + DBSCAN", "RapidFuzz Ratio"])
    threshold = st.slider("Ambang Kemiripan (%)", 30, 100, 50)
    limit = st.number_input("Batas jumlah baris untuk deteksi (0 = semua)", 0, 10000, 0)
    detect_button = st.button("🚀 Jalankan Deteksi")

if uploaded_file and detect_button:
    content = uploaded_file.read()
    decoded = content.decode("utf-8", errors="replace")
    df = pd.read_csv(StringIO(decoded), on_bad_lines='skip')

    column = st.selectbox("Pilih kolom untuk deteksi:", df.columns)
    df = df[[column]].copy()
    df.reset_index(inplace=True)  # Keep track of original row

    if limit > 0:
        df = df.head(limit)

    df['cluster'] = -1

    with st.spinner("Mendeteksi duplikat..."):
        if method == "TF-IDF + DBSCAN":
            tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
            matrix = tfidf.fit_transform(df[column].astype(str))
            model = DBSCAN(eps=(1 - threshold/100), min_samples=1, metric='cosine')
            df['cluster'] = model.fit_predict(matrix)

        else:  # RapidFuzz
            texts = df[column].astype(str).tolist()
            labels = [-1] * len(texts)
            cluster_id = 0
            for i in range(len(texts)):
                if labels[i] == -1:
                    labels[i] = cluster_id
                    for j in range(i + 1, len(texts)):
                        if labels[j] == -1 and fuzz.ratio(texts[i], texts[j]) >= threshold:
                            labels[j] = cluster_id
                    cluster_id += 1
            df['cluster'] = labels

        dupes = df.groupby("cluster").filter(lambda x: len(x) > 1)
        st.success(f"{len(dupes)} baris duplikat ditemukan dari {len(df)} total baris.")
        st.dataframe(dupes.sort_values(by='cluster'), use_container_width=True)

        def to_excel_bytes(df_out):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_out.to_excel(writer, index=False, sheet_name="Duplikat")
            return output.getvalue()

        excel_bytes = to_excel_bytes(dupes)
        b64 = base64.b64encode(excel_bytes).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="duplikat_cepat.xlsx">🔹 Download Hasil Excel</a>'
        st.markdown(href, unsafe_allow_html=True)

elif uploaded_file and not detect_button:
    st.info("Silakan tekan tombol 'Jalankan Deteksi' untuk memulai.")
