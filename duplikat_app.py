import streamlit as st
import pandas as pd
import base64
from rapidfuzz import fuzz
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from io import StringIO, BytesIO

# Setup halaman
st.set_page_config(page_title="🔍 Deteksi Duplikasi Data", layout="wide")
st.title("🔎 Deteksi Duplikasi Data Teks")
st.markdown("Deteksi potensi data duplikat menggunakan metode **TF-IDF + DBSCAN** atau **RapidFuzz**.")

# Sidebar input
with st.sidebar:
    st.header("⚙️ Pengaturan Deteksi")
    uploaded_file = st.file_uploader("📤 Upload file CSV utama", type=["csv"])
    catalog_file = st.file_uploader("📚 Upload file katalog referensi (Opsional)", type=["csv"], key="catalog")
    similarity_threshold = st.slider("🎯 Ambang Kemiripan (%)", 30, 100, 50)
    method = st.radio("🧠 Metode Deteksi", ["TF-IDF + DBSCAN", "RapidFuzz Ratio"])
    run_button = st.button("🚀 Jalankan Deteksi")

# Proses data
if uploaded_file:
    content = uploaded_file.read()
    decoded = None
    for enc in ['utf-8', 'latin1', 'windows-1252']:
        try:
            decoded = content.decode(enc)
            st.success(f"✅ File berhasil didekode dengan encoding: {enc}")
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
            df.reset_index(inplace=True)
            df['cluster'] = -1

            if method == "TF-IDF + DBSCAN":
                vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
                tfidf_matrix = vectorizer.fit_transform(df[column_to_check].astype(str))
                model = DBSCAN(eps=(1 - similarity_threshold / 100), min_samples=1, metric='cosine')
                df['cluster'] = model.fit_predict(tfidf_matrix)

            elif method == "RapidFuzz Ratio":
                texts = df[column_to_check].astype(str).tolist()
                labels = [-1] * len(texts)
                cluster_id = 0
                for i in range(len(texts)):
                    if labels[i] == -1:
                        labels[i] = cluster_id
                        for j in range(i + 1, len(texts)):
                            if labels[j] == -1 and fuzz.ratio(texts[i], texts[j]) >= similarity_threshold:
                                labels[j] = cluster_id
                        cluster_id += 1
                df['cluster'] = labels

            # Validasi katalog
            if catalog_set:
                def is_typo_match(val):
                    return any(fuzz.ratio(val.lower(), ref) >= 90 for ref in catalog_set)
                df['valid_catalog'] = df[column_to_check].astype(str).apply(
                    lambda x: x.lower() in catalog_set or is_typo_match(x)
                )

            # Ambil duplikat
            dupes = df.groupby('cluster').filter(lambda x: len(x) > 1)
            total_clusters = df['cluster'].nunique()
            total_rows = len(df)
            total_dupes = len(dupes)

            st.success(f"✅ Ditemukan **{total_dupes} baris duplikat** dari total **{total_rows} baris**.")
            st.markdown(f"📊 Jumlah cluster terbentuk: `{total_clusters}`")

            if not dupes.empty:
                # Hitung rata-rata kemiripan
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

                summary_cluster = (
                    dupes.groupby('cluster')
                    .agg(rata2_kemiripan=('avg_similarity_in_cluster', 'mean'),
                         jumlah_baris=('index', 'count'))
                    .reset_index()
                    .sort_values(by="rata2_kemiripan", ascending=False)
                )
                summary_cluster['rata2_kemiripan'] = summary_cluster['rata2_kemiripan'].round(2)

                # Simpan hasil ke session state
                st.session_state['dupes'] = dupes
                st.session_state['df'] = df
                st.session_state['summary_cluster'] = summary_cluster
                st.session_state['column_to_check'] = column_to_check

# Tampilkan hasil jika sudah tersimpan
if 'dupes' in st.session_state:
    dupes = st.session_state['dupes']
    df = st.session_state['df']
    summary_cluster = st.session_state['summary_cluster']
    column_to_check = st.session_state['column_to_check']

    tab1, tab2, tab3 = st.tabs(["📄 Data Duplikat", "📈 Summary Cluster", "🗃️ Semua Data"])

    with tab1:
        st.subheader("🔍 Filter Data Duplikat")
        cluster_options = sorted(dupes['cluster'].unique().tolist())
        selected_clusters = st.multiselect("📂 Pilih Cluster ID:", cluster_options, default=cluster_options)
        min_similarity = st.slider("📈 Minimum Rata-rata Kemiripan:", 0, 100, 0)
        keyword = st.text_input("🔎 Cari teks dalam kolom:")

        filtered_dupes = dupes[
            dupes['cluster'].isin(selected_clusters) &
            (dupes['avg_similarity_in_cluster'] >= min_similarity)
        ]
        if keyword:
            filtered_dupes = filtered_dupes[
                filtered_dupes[column_to_check].astype(str).str.contains(keyword, case=False, na=False)
            ]

        st.markdown(f"📋 Menampilkan **{len(filtered_dupes)} baris** yang sesuai filter.")
        st.dataframe(filtered_dupes.sort_values(by='cluster'), use_container_width=True)

        # Fungsi download
        def to_excel_download(df_dict: dict):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                for sheet, data in df_dict.items():
                    data.to_excel(writer, index=False, sheet_name=sheet)
            return output.getvalue()

        # Tombol Download Semua
        excel_all = to_excel_download({
            "Data Duplikat": dupes,
            "Summary Cluster": summary_cluster,
            "Seluruh Data": df
        })
        b64_all = base64.b64encode(excel_all).decode()
        href_all = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_all}" download="hasil_semua_duplikasi.xlsx">✅ ⬇️ Download Semua Hasil Deteksi (Excel)</a>'
        st.markdown(href_all, unsafe_allow_html=True)

        # Tombol Download Filtered
        if not filtered_dupes.empty:
            excel_filtered = to_excel_download({
                "Filtered Duplikat": filtered_dupes,
                "Summary Cluster": summary_cluster
            })
            b64_filtered = base64.b64encode(excel_filtered).decode()
            href_filtered = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_filtered}" download="hasil_filtered_duplikasi.xlsx">🎯 ⬇️ Download Hasil yang Difilter Saja (Excel)</a>'
            st.markdown(href_filtered, unsafe_allow_html=True)
        else:
            st.info("⚠️ Tidak ada data yang cocok dengan filter untuk diunduh.")

    with tab2:
        st.dataframe(summary_cluster, use_container_width=True)

    with tab3:
        st.dataframe(df.drop(columns=['index']), use_container_width=True)
