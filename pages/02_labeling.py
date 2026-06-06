import os
import re
import uuid
from datetime import datetime
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils.sentiment import load_model, predict_batch, COLOR_MAP
from utils.preprocessing import preprocess

st.set_page_config(page_title="Labeling Sentimen", page_icon="🏷️", layout="wide")

# ── Session state defaults ────────────────────────────────
for key, default in {
    "labeling": False,
    "labeling_result": None,   # (df_result, saved_filename) setelah selesai
    "save_state": None,        # None | "pending" | "saved"
    "cancel_requested": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

labeling = st.session_state.labeling

# ── CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
.step-badge {
    display: inline-block;
    background: #1a1a2e;
    color: #e0e0ff;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 2px 10px;
    border-radius: 20px;
    margin-bottom: 6px;
    border: 1px solid #3a3a6e;
}
.pill {
    display: inline-block;
    padding: 3px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 4px 0;
}
.pill-done { background:#d1fae5; color:#065f46; border:1px solid #10b981; }
.pill-err  { background:#fee2e2; color:#991b1b; border:1px solid #ef4444; }
.confirm-box {
    background: #eff6ff;
    border: 1px solid #3b82f6;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

if labeling and st.session_state.save_state is None:
    st.markdown(
        "<div style='text-align:center;padding:0.5rem;background:#fff3cd;"
        "border:1px solid #ffc107;border-radius:8px;margin-bottom:1rem;font-weight:600;'>"
        "⏳ Labeling sedang berlangsung... Mohon jangan navigasi atau klik apapun.</div>",
        unsafe_allow_html=True,
    )

st.title("🏷️ Labeling Sentimen Otomatis")
st.markdown(
    "Label otomatis komentar YouTube dengan **sentimen (positif/netral/negatif)** "
    "menggunakan model IndoBERT `taufiqdp/indonesian-sentiment`."
)

# ── Load Model ────────────────────────────────────────────
clf = None
model_status = st.empty()

with model_status.container():
    st.markdown('<div class="step-badge">MEMUAT MODEL</div>', unsafe_allow_html=True)
    st.markdown("🤖 **Memuat model IndoBERT...**")
    bar_model = st.progress(0, text="Menginisialisasi...")

try:
    clf, is_cached = load_model()
    bar_model.empty()
    model_status.empty()
    source_info = "📁 cache lokal" if is_cached else "☁️ HuggingFace (tersimpan lokal)"
    st.success(f"✅ Model siap | Sumber: {source_info}")
except Exception as e:
    bar_model.empty()
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

st.markdown("---")

# ── Helper: bangun nama file ──────────────────────────────
def build_filename(custom_name: str, source_name: str) -> str:
    short_uid = uuid.uuid4().hex[:8]
    if custom_name:
        safe = re.sub(r"[^\w\-]", "_", custom_name).strip("_")
        base = f"{safe}_{short_uid}"
    else:
        src_base = os.path.splitext(os.path.basename(source_name))[0]
        base = f"labeled_{src_base}_{short_uid}"
    return f"data/labeling/{base}.csv"

# ── Helper: render download button ───────────────────────
def render_download_button(df: pd.DataFrame, filename: str, label: str = "⬇️ Download CSV"):
    csv_bytes = df.to_csv(index=False, sep=";").encode("utf-8-sig")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=os.path.basename(filename),
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )

# ── Helper: render chart distribusi sentimen ─────────────
def render_charts(df_result: pd.DataFrame, text_col: str):
    sentiment_counts = df_result["sentiment"].value_counts()
    colors = [COLOR_MAP.get(s, "#999") for s in sentiment_counts.index]

    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        ax1.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2),
        )
        ax1.set_title("Distribusi Sentimen (%)", fontweight="bold")
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        bars = ax2.bar(
            sentiment_counts.index,
            sentiment_counts.values,
            color=colors,
            edgecolor="white",
            width=0.5,
        )
        for bar, val in zip(bars, sentiment_counts.values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(val),
                ha="center",
                fontweight="bold",
            )
        ax2.set_title("Jumlah per Sentimen", fontweight="bold")
        ax2.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig2)
        plt.close(fig2)

    st.subheader("📋 Hasil Labeling")
    display_cols = [text_col, "clean", "sentiment", "confidence"]
    display_cols = [c for c in display_cols if c in df_result.columns]
    st.dataframe(df_result[display_cols], use_container_width=True)

    stats = (
        df_result.groupby("sentiment")["confidence"]
        .agg(["mean", "min", "max"])
        .round(3)
    )
    stats.columns = ["Rata-rata", "Minimum", "Maksimum"]
    st.subheader("📈 Confidence Score per Sentimen")
    st.dataframe(stats, use_container_width=True)


# ═════════════════════════════════════════════════════════
# BLOK HASIL — jika labeling sudah selesai
# ═════════════════════════════════════════════════════════
if st.session_state.labeling_result is not None:
    result_data = st.session_state.labeling_result
    save_state  = st.session_state.save_state

    df_result   = result_data["df"]
    proposed_fn = result_data["proposed_filename"]
    text_col    = result_data["text_col"]
    n_res       = len(df_result)

    # ── Menunggu konfirmasi simpan ────────────────────────
    if save_state == "pending":
        st.success(f"✅ Labeling selesai! **{n_res:,}** komentar terlabeli.")

        st.subheader("📊 Distribusi Sentimen")
        render_charts(df_result, text_col)

        # Tab download (sebelum simpan ke disk)
        st.markdown("---")
        st.info("💡 Anda bisa download CSV sekarang, atau simpan file ke server terlebih dahulu.")
        render_download_button(df_result, proposed_fn, "⬇️ Download CSV (tanpa simpan ke server)")

        # Kotak konfirmasi
        st.markdown('<div class="confirm-box">', unsafe_allow_html=True)
        st.markdown("### 💾 Konfirmasi Penyimpanan")
        st.markdown(
            f"File akan disimpan sebagai:\n\n"
            f"```\n{proposed_fn}\n```"
        )
        st.markdown("Apakah Anda ingin menyimpan hasil labeling ke server?")
        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("✅ Simpan File", type="primary", use_container_width=True):
                os.makedirs("data/labeling", exist_ok=True)
                df_result.to_csv(proposed_fn, index=False, sep=";", encoding="utf-8-sig")
                st.session_state.save_state = "saved"
                st.session_state.labeling   = False
                st.rerun()
        with bc2:
            if st.button("🚫 Batal & Buang Data", type="secondary", use_container_width=True):
                st.session_state.labeling_result = None
                st.session_state.save_state      = None
                st.session_state.labeling        = False
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── File sudah tersimpan ──────────────────────────────
    elif save_state == "saved":
        saved_fn = result_data.get("saved_filename", proposed_fn)

        st.markdown(
            f'<span class="pill pill-done">✅ File tersimpan: {saved_fn}</span>',
            unsafe_allow_html=True,
        )

        st.subheader("📊 Distribusi Sentimen")
        render_charts(df_result, text_col)

        st.markdown("---")
        render_download_button(df_result, saved_fn, "⬇️ Download Hasil CSV")
        st.info(f"📂 File juga tersimpan di server: `{saved_fn}`")

        if st.button("🔄 Labeling Baru", type="secondary", use_container_width=True):
            st.session_state.labeling_result = None
            st.session_state.save_state      = None
            st.session_state.labeling        = False
            st.rerun()

    st.stop()   # jangan render form di bawah saat hasil sudah tampil


# ═════════════════════════════════════════════════════════
# FORM INPUT — saat belum/setelah labeling
# ═════════════════════════════════════════════════════════
uploaded_df  = None
source_name  = ""

source_tab1, source_tab2 = st.tabs(["📂 Dari Folder data/", "📁 Upload CSV"])

with source_tab1:
    src_folder = st.radio(
        "Sumber folder",
        ["data/scraping/", "data/labeling/"],
        horizontal=True,
        disabled=labeling,
    )

    csv_files = []
    if os.path.isdir(src_folder):
        csv_files = sorted(
            [f for f in os.listdir(src_folder) if f.endswith(".csv")],
            reverse=True,
        )

    if csv_files:
        selected_file = st.selectbox("Pilih file CSV", csv_files, disabled=labeling)
        if selected_file:
            source_name = selected_file
            uploaded_df = pd.read_csv(
                os.path.join(src_folder, selected_file),
                sep=";",
                encoding="utf-8-sig",
            )
            uploaded_df.columns = uploaded_df.columns.str.strip()
            uploaded_df = uploaded_df.loc[:, ~uploaded_df.columns.str.contains("^Unnamed")]
            st.success(f"✅ **{selected_file}** — {len(uploaded_df)} baris")
    else:
        st.info(f"Belum ada file CSV di folder `{src_folder}`.")

with source_tab2:
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"], disabled=labeling)
    if uploaded_file:
        source_name = uploaded_file.name
        uploaded_df = pd.read_csv(uploaded_file)
        uploaded_df.columns = uploaded_df.columns.str.strip()
        uploaded_df = uploaded_df.loc[:, ~uploaded_df.columns.str.contains("^Unnamed")]
        st.success(f"✅ **{uploaded_file.name}** — {len(uploaded_df)} baris")

if uploaded_df is not None:
    st.write("**Pratinjau data:**")
    st.dataframe(uploaded_df.head(5), use_container_width=True)

    # Deteksi kolom teks
    text_col_candidates = ["comment", "clean", "text", "teks", "content", "isi"]
    detected_col = next(
        (c for c in text_col_candidates if c in uploaded_df.columns),
        uploaded_df.select_dtypes(include="object").columns[0],
    )
    text_col = st.selectbox(
        "📝 Pilih kolom yang berisi teks komentar",
        uploaded_df.select_dtypes(include="object").columns.tolist(),
        index=uploaded_df.columns.get_loc(detected_col) if detected_col in uploaded_df.columns else 0,
        disabled=labeling,
    )

    # ── Nama file kustom ──────────────────────────────────
    custom_name = st.text_input(
        "💾 **Nama File Hasil Labeling** *(opsional)*",
        placeholder="contoh: hasil_sentimen_video_A",
        help=(
            "Nama file yang akan disimpan. Karakter spesial diganti underscore. "
            "Jika kosong, nama default dari nama file sumber. "
            "Kode unik 8 karakter otomatis ditambahkan agar tidak bentrok."
        ),
        disabled=labeling,
    )

    if not labeling:
        st.write(f"**Total data:** {len(uploaded_df)} baris | **Kolom teks:** `{text_col}`")

    # ── Tombol Mulai ──────────────────────────────────────
    btn_clicked = st.button(
        "🔬 Mulai Labeling",
        type="primary",
        use_container_width=True,
        disabled=labeling,
    )

    if btn_clicked:
        st.session_state._label_source_name  = source_name
        st.session_state._label_text_col     = text_col
        st.session_state._label_custom_name  = custom_name.strip()
        st.session_state._label_df           = uploaded_df.copy()
        st.session_state.labeling            = True
        st.session_state.cancel_requested    = False
        st.session_state.labeling_result     = None
        st.session_state.save_state          = None
        st.rerun()

    # ═══════════════════════════════════════════════════════
    # PROSES LABELING
    # ═══════════════════════════════════════════════════════
    if labeling:
        source_df        = st.session_state._label_df.copy()
        text_col_run     = st.session_state._label_text_col
        source_name_run  = st.session_state._label_source_name
        custom_name_run  = st.session_state._label_custom_name
        total            = len(source_df)

        # Tombol Batal — di luar st.status agar bisa diklik
        cancel_placeholder = st.empty()
        with cancel_placeholder.container():
            if st.button(
                "⛔ Batalkan Labeling",
                type="secondary",
                use_container_width=True,
                key="btn_cancel_label",
            ):
                st.session_state.cancel_requested = True

        with st.status("⏳ Proses labeling berjalan...", expanded=True) as status:

            bar = st.progress(0, text="Inisialisasi...")

            # Cek pembatalan sebelum mulai
            if st.session_state.cancel_requested:
                bar.empty()
                status.update(label="🚫 Labeling dibatalkan.", state="error")
                st.session_state.labeling         = False
                st.session_state.cancel_requested = False
                cancel_placeholder.empty()
                st.rerun()

            # ── Langkah 1: Preprocessing ──────────────────
            status.write("🧹 **Langkah 1/3 — Preprocessing teks...**")
            texts_raw = source_df[text_col_run].astype(str).tolist()
            cleaned   = []
            for i, t in enumerate(texts_raw):
                if st.session_state.cancel_requested:
                    bar.empty()
                    status.update(label="🚫 Labeling dibatalkan.", state="error")
                    st.session_state.labeling         = False
                    st.session_state.cancel_requested = False
                    cancel_placeholder.empty()
                    st.rerun()
                cleaned.append(preprocess(t))
                bar.progress((i + 1) / total, text=f"Preprocessing {i+1}/{total}")

            source_df["clean"] = cleaned

            # ── Langkah 2: Inferensi ──────────────────────
            status.write("🤖 **Langkah 2/3 — Inferensi sentimen...**")

            def on_progress(done, n):
                if st.session_state.cancel_requested:
                    raise StopIteration("Dibatalkan oleh pengguna.")
                bar.progress(done / n, text=f"Labeling {done}/{n} komentar")
                status.update(label=f"⏳ Menganalisis {done}/{n} komentar...")

            cancelled = False
            try:
                labels, scores = predict_batch(
                    clf,
                    source_df["clean"].tolist(),
                    batch_size=16,
                    progress_callback=on_progress,
                )
            except StopIteration:
                cancelled = True
            except Exception as e:
                bar.empty()
                status.update(label=f"❌ Gagal: {e}", state="error")
                st.session_state.labeling         = False
                st.session_state.cancel_requested = False
                cancel_placeholder.empty()
                st.rerun()

            if cancelled or st.session_state.cancel_requested:
                bar.empty()
                status.update(label="🚫 Labeling dibatalkan oleh pengguna.", state="error")
                st.session_state.labeling         = False
                st.session_state.cancel_requested = False
                cancel_placeholder.empty()
                st.rerun()

            source_df["sentiment"]   = labels
            source_df["confidence"]  = scores

            # ── Langkah 3: Siapkan nama file ──────────────
            status.write("📝 **Langkah 3/3 — Menyiapkan konfirmasi penyimpanan...**")
            proposed_filename = build_filename(custom_name_run, source_name_run)
            bar.empty()
            status.update(
                label=f"✅ Selesai! {total} komentar terlabeli.",
                state="complete",
            )

        cancel_placeholder.empty()

        # Simpan hasil ke session_state, tunggu konfirmasi user
        st.session_state.labeling_result = {
            "df":                source_df,
            "proposed_filename": proposed_filename,
            "text_col":          text_col_run,
        }
        st.session_state.save_state      = "pending"
        st.session_state.cancel_requested = False
        st.rerun()