import streamlit as st
import pandas as pd
from rapidfuzz import fuzz
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from io import StringIO
import base64

st.set_page_config(page_title="Deteksi Duplikat", layout="wide")
st.title("🔎 Deteksi Duplikasi Data Teks")
st.markdown("Deteksi potensi data duplikat menggunakan metode **TF-IDF + DBSCAN** atau **RapidFuzz**.")

uploaded_file = st.file_uploader("📤 Upload file CSV", type=["csv"])

if uploaded_file is not None:
    content = uploaded_file.read()
    decoded = None
    for enc in ['utf-8', 'latin1', 'windows-1252']:
        try:
            decoded = content.decode(enc)
            st.success(f"File berhasil didekode dengan encoding: {enc}")
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        st.error("Gagal membaca file, encoding tidak dikenali.")
        st.stop()

    try:
        df = pd.read_csv(StringIO(decoded), on_bad_lines='skip')
        st.write("📄 **Contoh data**:", df.head())
    except Exception as e:
        st.error(f"Gagal parsing CSV: {e}")
        st.stop()

    column_to_check = st.selectbox("📌 Pilih kolom untuk dicek duplikasi:", df.columns)
    similarity_threshold = st.slider("🎯 Ambang kemiripan (persen):", 30, 100, 50)
    method = st.radio("🧠 Metode deteksi:", ["TF-IDF + DBSCAN", "RapidFuzz Ratio"])

    if st.button("🚀 Jalankan Deteksi Duplikasi"):
        try:
            df['cluster'] = -1
            if method == "TF-IDF + DBSCAN":
                vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
                tfidf_matrix = vectorizer.fit_transform(df[column_to_check].astype(str))

                model = DBSCAN(eps=(1 - similarity_threshold / 100), min_samples=1, metric='cosine')
                labels = model.fit_predict(tfidf_matrix)
                df['cluster'] = labels

            elif method == "RapidFuzz Ratio":
                texts = df[column_to_check].astype(str).tolist()
                cluster_id = 0
                labels = [-1] * len(texts)

                for i in range(len(texts)):
                    if labels[i] == -1:
                        labels[i] = cluster_id
                        for j in range(i + 1, len(texts)):
                            if labels[j] == -1:
                                score = fuzz.ratio(texts[i], texts[j])
                                if score >= similarity_threshold:
                                    labels[j] = cluster_id
                        cluster_id += 1
                df['cluster'] = labels

            # Hasil
            dupes = df.groupby('cluster').filter(lambda x: len(x) > 1)
            total_clusters = df['cluster'].nunique()
            total_rows = len(df)
            total_dupes = len(dupes)

            st.markdown(f"✅ **{total_dupes} baris terindikasi duplikat** dari total {total_rows} baris.")
            st.markdown(f"📊 **Jumlah cluster yang terbentuk:** {total_clusters}")

            if not dupes.empty:
                st.subheader("📌 Data Duplikat Ditemukan")
                st.dataframe(dupes.sort_values(by='cluster'))

                # Download hasil duplikat
                output_filename = "hasil_duplikat.xlsx"
                dupes.to_excel(output_filename, index=False, engine='openpyxl')
                with open(output_filename, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{output_filename}">⬇️ Download Hasil Duplikat (Excel)</a>'
                    st.markdown(href, unsafe_allow_html=True)

                # Download semua data + cluster
                all_filename = "data_dengan_cluster.xlsx"
                df.to_excel(all_filename, index=False, engine='openpyxl')
                with open(all_filename, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                    href2 = f'<a href="data:application/octet-stream;base64,{b64}" download="{all_filename}">⬇️ Download Semua Data + Cluster</a>'
                    st.markdown(href2, unsafe_allow_html=True)
            else:
                st.info("✅ Tidak ada potensi duplikasi yang ditemukan.")

        except Exception as e:
            st.error(f"❌ Error saat proses deteksi: {e}")
