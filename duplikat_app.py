import streamlit as st
import pandas as pd
import base64
from rapidfuzz import fuzz
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from io import StringIO, BytesIO

# Konfigurasi halaman
st.set_page_config(page_title="🔍 Deteksi Duplikasi Data", layout="wide")
st.title("🔎 Deteksi Duplikasi Data Teks")
st.markdown("Deteksi potensi data duplikat menggunakan metode **TF-IDF + DBSCAN** atau **RapidFuzz**. Hasil didasarkan pada nilai kemiripan antar baris.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan Deteksi")
    uploaded_file = st.file_uploader("📤 Upload file CSV utama", type=["csv"])
    catalog_file = st.file_uploader("📚 Upload file katalog referensi (Opsional)", type=["csv"], key="catalog")
    similarity_threshold = st.slider("🎯 Ambang Kemiripan (%)", 30, 100, 50)
    method = st.radio("🧠 Metode Deteksi", ["TF-IDF + DBSCAN", "RapidFuzz Ratio"])
    run_button = st.button("🚀 Jalankan Deteksi")

# Proses utama
if uploaded_file:
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
        st.error("❌ Gagal membaca file. Encoding tidak dikenali.")
        st.stop()

    try:
        df = pd.read_csv(StringIO(decoded), on_bad_lines='skip')
    except Exception as e:
        st.error(f"❌ Gagal parsing CSV: {e}")
        st.stop()

    column_to_check = st.selectbox("📌 Pilih kolom yang akan dicek duplikasi:", df.columns)

    catalog_df = None
    catalog_set = set()
    if catalog_file:
        try:
            catalog_decoded = catalog_file.read().decode("utf-8")
            catalog_df = pd.read_csv(StringIO(catalog_decoded), on_bad_lines='skip')
            st.success(f"📘 Katalog berhasil dimuat. Jumlah entri: {len(catalog_df)}")
            if column_to_check in catalog_df.columns:
                catalog_set = set(catalog_df[column_to_check].astype(str).str.lower())
        except Exception as e:
            st.warning(f"⚠️ File katalog tidak dapat dibaca: {e}")

    if run_button:
        with st.spinner("🔍 Sedang memproses..."):
            df.reset_index(inplace=True)  # untuk pelacakan baris
            df['cluster'] = -1

            if method == "TF-IDF + DBSCAN":
                vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
                tfidf_matrix = vectorizer.fit_transform(df[column_to_check].astype(str))
                model = DBSCAN(eps=(1 - similarity_threshold / 100), min_samples=1, metric='cosine')
                df['cluster'] = model.fit_predict(tfidf_matrix)

            elif method == "RapidFuzz Ratio":
                texts = df[column_to_check].astype(str).tolist()
                cluster_id = 0
                labels = [-1] * len(texts)
                for i in range(len(texts)):
                    if labels[i] == -1:
                        labels[i] = cluster_id
                        for j in range(i + 1, len(texts)):
                            if labels[j] == -1 and fuzz.ratio(texts[i], texts[j]) >= similarity_threshold:
                                labels[j] = cluster_id
                        cluster_id += 1
                df['cluster'] = labels

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

            st.success(f"✅ Ditemukan **{total_dupes} baris duplikat** dari total **{total_rows} baris**.")
            st.markdown(f"📊 **Jumlah cluster yang terbentuk:** `{total_clusters}`")

            if not dupes.empty:
                # Hitung rata-rata kemiripan per baris dalam cluster
                similarity_per_row = []
                for cluster_id, group in dupes.groupby("cluster"):
                    texts = group[column_to_check].astype(str).tolist()
                    indices = group.index.tolist()
                    for i, idx_i in enumerate(indices):
                        scores = [
                            fuzz.ratio(texts[i], texts[j])
                            for j in range(len(texts)) if i != j
                        ]
                        avg_sim = sum(scores) / len(scores) if scores else 0
                        similarity_per_row.append((idx_i, round(avg_sim, 2)))

                sim_df = pd.DataFrame(similarity_per_row, columns=['index', 'avg_similarity_in_cluster'])
                dupes = dupes.merge(sim_df, on='index')

                # Rata-rata tiap cluster
                summary_cluster = (
                    dupes.groupby('cluster')
                    .agg(rata2_kemiripan=('avg_similarity_in_cluster', 'mean'),
                         jumlah_baris=('index', 'count'))
                    .reset_index()
                    .sort_values(by="rata2_kemiripan", ascending=False)
                )
                summary_cluster['rata2_kemiripan'] = summary_cluster['rata2_kemiripan'].round(2)

                # Tampilkan hasil dengan tab
                tab1, tab2, tab3 = st.tabs(["📄 Data Duplikat", "📈 Summary Cluster", "🗃️ Semua Data"])

                with tab1:
                    st.dataframe(dupes.sort_values(by='cluster'), use_container_width=True)

                with tab2:
                    st.dataframe(summary_cluster, use_container_width=True)

                with tab3:
                    st.dataframe(df.drop(columns=['index']), use_container_width=True)

                # Ekspor ke Excel
                def to_excel_download(df_dict: dict):
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        for sheet, data in df_dict.items():
                            data.to_excel(writer, index=False, sheet_name=sheet)
                    return output.getvalue()

                excel_bytes = to_excel_download({
                    "Data Duplikat": dupes,
                    "Summary Cluster": summary_cluster,
                    "Seluruh Data": df
                })

                b64 = base64.b64encode(excel_bytes).decode()
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="hasil_deteksi_duplikasi.xlsx">⬇️ Download Hasil Deteksi (Excel)</a>'
                st.markdown(href, unsafe_allow_html=True)

            else:
                st.info("✅ Tidak ditemukan duplikasi berdasarkan ambang kemiripan yang dipilih.")
