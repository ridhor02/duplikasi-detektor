import streamlit as st
import pandas as pd
from rapidfuzz import fuzz
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from io import StringIO
import base64

# Judul Aplikasi
st.title("🚀 Deteksi Duplikasi Teks dengan TF-IDF & DBSCAN")

# Upload file
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    # Deteksi encoding
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

    # Baca ke DataFrame
    try:
        df = pd.read_csv(StringIO(decoded), on_bad_lines='skip')
        st.write("Contoh data:", df.head())
    except Exception as e:
        st.error(f"Gagal parsing CSV: {e}")
        st.stop()

    # Pilih kolom untuk deteksi duplikasi
    column_to_check = st.selectbox("Pilih kolom untuk dicek duplikasi:", df.columns)

    similarity_threshold = st.slider("Ambang kemiripan (dalam %):", 30, 100, 50)

    if st.button("🔍 Proses Deteksi Duplikasi"):
        try:
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
            tfidf_matrix = vectorizer.fit_transform(df[column_to_check].astype(str))

            model = DBSCAN(eps=(1 - similarity_threshold / 100), min_samples=1, metric='cosine')
            labels = model.fit_predict(tfidf_matrix)

            df['cluster'] = labels
            dupes = df.groupby('cluster').filter(lambda x: len(x) > 1)

            if not dupes.empty:
                st.subheader("📌 Potensi Data Duplikat Ditemukan:")
                st.dataframe(dupes.sort_values('cluster'))

                # Simpan ke Excel
                output_filename = "potensi_duplikat.xlsx"
                dupes.to_excel(output_filename, index=False, engine='openpyxl')

                # Tombol download
                with open(output_filename, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{output_filename}">⬇️ Download Hasil Excel</a>'
                    st.markdown(href, unsafe_allow_html=True)
            else:
                st.info("✅ Tidak ada potensi duplikasi yang terdeteksi.")
        except Exception as e:
            st.error(f"Gagal memproses: {e}")
