import streamlit as st
import pandas as pd
from rapidfuzz import fuzz
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from io import StringIO
import base64
from difflib import SequenceMatcher

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
    except Exception as e:
        st.error(f"Gagal parsing CSV: {e}")
        st.stop()

    catalog_file = st.file_uploader("📚 (Opsional) Upload file referensi katalog (CSV)", type=["csv"], key="catalog")
    catalog_df = None
    catalog_set = set()

    column_to_check = st.selectbox("📌 Pilih kolom untuk dicek duplikasi:", df.columns)
    df.reset_index(inplace=True)

    if catalog_file is not None:
        try:
            catalog_decoded = catalog_file.read().decode("utf-8")
            catalog_df = pd.read_csv(StringIO(catalog_decoded), on_bad_lines='skip')
            st.success(f"Katalog berhasil dimuat. Jumlah entri: {len(catalog_df)}")
            if column_to_check in catalog_df.columns:
                catalog_set = set(catalog_df[column_to_check].astype(str).str.lower())
        except Exception as e:
            st.warning(f"File katalog tidak dapat dibaca: {e}")

    similarity_threshold = st.slider(
        "🎯 Ambang kemiripan (persen):",
        min_value=30,
        max_value=100,
        value=50,
        help="Semakin tinggi ambang kemiripan, semakin ketat pencocokan."
    )

    method = st.radio(
        "🧠 Metode deteksi:",
        ["TF-IDF + DBSCAN", "RapidFuzz Ratio"],
        help="Pilih metode deteksi yang sesuai."
    )

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

            # Validasi terhadap katalog
            if catalog_set:
                def is_typo_match(val):
                    return any(fuzz.ratio(val.lower(), ref) >= 90 for ref in catalog_set)
                df['valid_catalog'] = df[column_to_check].astype(str).apply(
                    lambda x: x.lower() in catalog_set or is_typo_match(x)
                )

            dupes = df.groupby('cluster').filter(lambda x: len(x) > 1)
            total_clusters = df['cluster'].nunique()
            total_rows = len(df)
            total_dupes = len(dupes)

            st.markdown(f"✅ **{total_dupes} baris terindikasi duplikat** dari total {total_rows} baris.")
            st.markdown(f"📊 **Jumlah cluster yang terbentuk:** {total_clusters}")

            if not dupes.empty:
                st.markdown("### 🧾 Data Duplikat")
                st.dataframe(dupes.sort_values(by='cluster'))

                similarity_scores = []
                for cluster_id, group in dupes.groupby("cluster"):
                    texts = group[column_to_check].astype(str).tolist()
                    scores = []
                    for i in range(len(texts)):
                        for j in range(i + 1, len(texts)):
                            scores.append(fuzz.ratio(texts[i], texts[j]))
                    avg_score = sum(scores) / len(scores) if scores else 0
                    similarity_scores.append({
                        "cluster": cluster_id,
                        "rata2_kemiripan": round(avg_score, 2),
                        "jumlah_baris": len(group)
                    })

                score_df = pd.DataFrame(similarity_scores).sort_values(by="rata2_kemiripan", ascending=False)
                st.markdown("### 📈 Rata-rata Kemiripan per Cluster")
                st.dataframe(score_df)

                # 🔽 Download hasil duplikat
                output_filename = "hasil_duplikat.xlsx"
                dupes.to_excel(output_filename, index=False, engine='openpyxl')
                with open(output_filename, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{output_filename}">⬇️ Download Hasil Duplikat (Excel)</a>'
                    st.markdown(href, unsafe_allow_html=True)

                # 🔽 Download seluruh data + cluster
                all_filename = "data_dengan_cluster.xlsx"
                df.to_excel(all_filename, index=False, engine='openpyxl')
                with open(all_filename, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                    href2 = f'<a href="data:application/octet-stream;base64,{b64}" download="{all_filename}">⬇️ Download Semua Data + Cluster</a>'
                    st.markdown(href2, unsafe_allow_html=True)

                # 🔽 Download data kemiripan antar cluster
                output_score = "kemiripan_cluster.xlsx"
                score_df.to_excel(output_score, index=False, engine='openpyxl')
                with open(output_score, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                    href_score = f'<a href="data:application/octet-stream;base64,{b64}" download="{output_score}">⬇️ Download Data Kemiripan Cluster (Excel)</a>'
                    st.markdown(href_score, unsafe_allow_html=True)

            else:
                st.info("✅ Tidak ada potensi duplikasi yang ditemukan.")

        except Exception as e:
            st.error(f"❌ Error saat proses deteksi: {e}")
