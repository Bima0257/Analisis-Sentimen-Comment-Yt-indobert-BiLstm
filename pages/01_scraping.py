import os
import re
import uuid
from datetime import date, datetime
import streamlit as st
import pandas as pd

from utils.youtube_scraper import YouTubeScraper, extract_video_id
from utils.preprocessing import preprocess

st.set_page_config(page_title="Scraping YouTube", page_icon="📥", layout="wide")

# ── Session state defaults ────────────────────────────────
for key, default in {
    "scraping": False,
    "scraping_result": None,
    "save_state": None,       # None | "pending" | "saved"
    "cancel_requested": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

scraping = st.session_state.scraping

# ── CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
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
.pill-warn { background:#fef9c3; color:#92400e; border:1px solid #f59e0b; }
.confirm-box {
    background: #eff6ff;
    border: 1px solid #3b82f6;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin: 1rem 0;
}
.cancel-box {
    background: #fff7ed;
    border: 1px solid #fb923c;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    text-align: center;
    font-weight: 600;
    color: #9a3412;
}
</style>
""", unsafe_allow_html=True)

# ── Banner saat scraping aktif ────────────────────────────
if scraping and st.session_state.save_state is None:
    st.markdown(
        "<div style='text-align:center;padding:0.5rem;background:#fff3cd;"
        "border:1px solid #ffc107;border-radius:8px;margin-bottom:1rem;font-weight:600;'>"
        "⏳ Scraping sedang berlangsung...</div>",
        unsafe_allow_html=True,
    )

st.title("📥 Scraping Komentar YouTube")
st.markdown(
    "Ambil komentar dari video YouTube menggunakan **Google YouTube Data API v3**. "
    "Hasil scraping akan disimpan ke folder `data/scraping/` sebagai file CSV setelah dikonfirmasi."
)

# ── API Key ───────────────────────────────────────────────
api_key = os.getenv("GOOGLE_API_KEY", "")
if not api_key or api_key == "your_youtube_api_key_here":
    st.sidebar.warning("⚠️ **API Key belum diatur**")
    with st.sidebar.expander("⚙️ Atur API Key", expanded=True):
        api_key_input = st.text_input(
            "Masukkan Google API Key",
            type="password",
            placeholder="AIzaSy...",
            help="Dapatkan API key di https://console.cloud.google.com/apis/credentials",
            disabled=scraping,
        )
        if api_key_input:
            os.environ["GOOGLE_API_KEY"] = api_key_input
            st.sidebar.success("✅ API Key tersimpan di session")
            api_key = api_key_input
        else:
            st.sidebar.info("💡 Buat file `.env` dengan isi:\n\n`GOOGLE_API_KEY=your_key_here`")
else:
    st.sidebar.success("Scraping siap digunakan")

st.sidebar.markdown("---")

# ── Form input ────────────────────────────────────────────
url = st.text_input(
    "🔗 **Link Video YouTube**",
    placeholder="https://www.youtube.com/watch?v=... atau https://youtu.be/...",
    help="Tempel link video YouTube yang ingin di-scrape komentarnya.",
    disabled=scraping,
)

if url and not scraping:
    vid_preview = extract_video_id(url)
    if vid_preview:
        st.caption(f"✓ Video ID terdeteksi: `{vid_preview}`")
    else:
        st.warning("URL tidak valid. Gunakan format: `youtube.com/watch?v=...` atau `youtu.be/...`")

col1, col2, col3 = st.columns(3)
with col1:
    today = date.today()
    start_date = st.date_input(
        "📅 Tanggal Mulai", value=None,
        min_value=date(2005, 1, 1), max_value=today,
        help="Kosongi untuk ambil semua komentar tanpa filter tanggal.",
        disabled=scraping,
    )
with col2:
    end_date = st.date_input(
        "📅 Tanggal Akhir", value=None,
        min_value=date(2005, 1, 1), max_value=today,
        help="Kosongi untuk ambil semua komentar tanpa filter tanggal.",
        disabled=scraping,
    )
with col3:
    max_comments = st.number_input(
        "📊 Maks. Komentar", min_value=10, max_value=5000, value=500, step=100,
        help="Batas maksimal komentar yang diambil.",
        disabled=scraping,
    )

custom_name = st.text_input(
    "💾 **Nama File Hasil Scraping** *(opsional)*",
    placeholder="contoh: komentar_video_analisis",
    help=(
        "Nama file yang akan disimpan. Karakter spesial diganti underscore. "
        "Jika kosong, nama default pakai Video ID. "
        "Kode unik 8 karakter otomatis ditambahkan di akhir agar tidak bentrok."
    ),
    disabled=scraping,
)

if not scraping and start_date and end_date and start_date > end_date:
    st.error("❌ Tanggal Mulai harus sebelum Tanggal Akhir.")

# ── Tombol Mulai Scraping ─────────────────────────────────
btn_clicked = st.button(
    "🚀 Mulai Scraping",
    type="primary",
    use_container_width=True,
    disabled=scraping,
)

if btn_clicked:
    if not url:
        st.warning("Masukkan link video YouTube terlebih dahulu.")
        st.stop()
    vid_check = extract_video_id(url)
    if not vid_check:
        st.error("URL YouTube tidak valid.")
        st.stop()

    st.session_state._scrape_url         = url
    st.session_state._scrape_start       = start_date if start_date else None
    st.session_state._scrape_end         = end_date if end_date else None
    st.session_state._scrape_max         = max_comments
    st.session_state._scrape_custom_name = custom_name.strip()
    st.session_state.scraping            = True
    st.session_state.save_state          = None
    st.session_state.scraping_result     = None
    st.session_state.cancel_requested    = False
    st.rerun()

# ── Helper: bangun nama file ──────────────────────────────
def build_filename(custom_name: str, vid: str) -> str:
    short_uid = uuid.uuid4().hex[:8]
    if custom_name:
        safe = re.sub(r"[^\w\-]", "_", custom_name).strip("_")
        base = f"{safe}_{short_uid}"
    else:
        base = f"scraping_{vid}_{short_uid}"
    return f"data/scraping/{base}.csv"

# ── Helper: render tombol download CSV ────────────────────
def render_download_button(df: pd.DataFrame, filename: str, label: str = "⬇️ Download CSV"):
    csv_bytes = df.to_csv(index=False, sep=';').encode('utf-8-sig')
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=os.path.basename(filename),
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )

# ═════════════════════════════════════════════════════════
# BLOK UTAMA — saat scraping aktif
# ═════════════════════════════════════════════════════════
if scraping:
    result    = st.session_state.scraping_result
    save_state = st.session_state.save_state

    # ── 1. Menunggu konfirmasi simpan ─────────────────────
    if result is not None and save_state == "pending":
        vid_res, df, proposed_filename = result
        n_comments = len(df)

        st.success(f"✅ Scraping selesai! **{n_comments:,}** komentar berhasil diambil.")

        st.subheader("📋 Pratinjau Data")
        st.dataframe(df.head(10), use_container_width=True)

        tab1, tab2, tab3 = st.tabs(["📊 Statistik", "📄 Semua Data", "⬇️ Download"])
        with tab1:
            cs1, cs2, cs3 = st.columns(3)
            cs1.metric("Total Komentar", f"{n_comments:,}")
            cs2.metric("Rata-rata Like", f"{df['like_count'].mean():.1f}")
            cs3.metric("Komentar Unik", f"{df['author'].nunique():,}")
            st.write("**10 Komentar Teratas (berdasarkan like):**")
            st.dataframe(
                df.nlargest(10, "like_count")[["author", "comment", "like_count"]],
                use_container_width=True,
            )
        with tab2:
            st.dataframe(df, use_container_width=True)
            st.caption(f"Total {n_comments} baris × {len(df.columns)} kolom")
        with tab3:
            st.info("Download CSV tanpa menyimpan file ke server.")
            render_download_button(df, proposed_filename)

        # Kotak konfirmasi penyimpanan
        st.markdown('<div class="confirm-box">', unsafe_allow_html=True)
        st.markdown("### 💾 Konfirmasi Penyimpanan")
        st.markdown(
            f"File akan disimpan sebagai:\n\n"
            f"```\n{proposed_filename}\n```"
        )
        st.markdown("Apakah Anda ingin menyimpan hasil scraping ke server?")
        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("✅ Simpan File", type="primary", use_container_width=True):
                os.makedirs("data/scraping", exist_ok=True)
                df.to_csv(proposed_filename, index=False, sep=';', encoding='utf-8-sig')
                st.session_state.save_state = "saved"
                st.rerun()
        with bc2:
            if st.button("🚫 Batal & Buang Data", type="secondary", use_container_width=True):
                st.session_state.scraping        = False
                st.session_state.scraping_result = None
                st.session_state.save_state      = None
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 2. File sudah tersimpan ───────────────────────────
    elif result is not None and save_state == "saved":
        vid_res, df, saved_filename = result
        n_comments = len(df)

        st.markdown(
            f'<span class="pill pill-done">✅ File tersimpan: {saved_filename}</span>',
            unsafe_allow_html=True
        )

        st.subheader("📋 Pratinjau Data")
        st.dataframe(df.head(10), use_container_width=True)

        tab1, tab2, tab3 = st.tabs(["📊 Statistik", "📄 Semua Data", "⬇️ Download"])
        with tab1:
            cs1, cs2, cs3 = st.columns(3)
            cs1.metric("Total Komentar", f"{n_comments:,}")
            cs2.metric("Rata-rata Like", f"{df['like_count'].mean():.1f}")
            cs3.metric("Komentar Unik", f"{df['author'].nunique():,}")
            st.write("**10 Komentar Teratas (berdasarkan like):**")
            st.dataframe(
                df.nlargest(10, "like_count")[["author", "comment", "like_count"]],
                use_container_width=True,
            )
        with tab2:
            st.dataframe(df, use_container_width=True)
            st.caption(f"Total {n_comments} baris × {len(df.columns)} kolom")
        with tab3:
            render_download_button(df, saved_filename)
            st.info(f"📂 File juga tersimpan di server: `{saved_filename}`")

        if st.button("🔄 Scraping Baru", type="secondary", use_container_width=True):
            st.session_state.scraping        = False
            st.session_state.scraping_result = None
            st.session_state.save_state      = None
            st.rerun()

    # ── 3. Proses scraping berjalan ───────────────────────
    else:
        vid_input        = extract_video_id(st.session_state.get("_scrape_url", ""))
        custom_name_saved = st.session_state.get("_scrape_custom_name", "")

        # Tombol Batal — ditampilkan DI LUAR st.status agar bisa diklik
        cancel_placeholder = st.empty()
        with cancel_placeholder.container():
            if st.button(
                "⛔ Batalkan Scraping",
                type="secondary",
                use_container_width=True,
                key="btn_cancel",
            ):
                st.session_state.cancel_requested = True

        with st.status("⏳ Proses scraping berjalan...", expanded=True) as status:

            # ── Langkah 1: Info video ─────────────────────
            status.write("🔍 **Langkah 1/4 — Mendapatkan info video...**")
            bar = st.progress(0, text="Inisialisasi...")

            # Cek pembatalan sebelum mulai
            if st.session_state.cancel_requested:
                bar.empty()
                status.update(label="🚫 Scraping dibatalkan.", state="error")
                st.session_state.scraping         = False
                st.session_state.scraping_result  = None
                st.session_state.save_state       = None
                st.session_state.cancel_requested = False
                cancel_placeholder.empty()
                st.rerun()

            try:
                scraper    = YouTubeScraper(api_key=api_key)
                video_info = scraper.get_video_info(vid_input)
            except Exception as e:
                bar.empty()
                status.update(label=f"❌ Gagal: {e}", state="error")
                st.session_state.scraping         = False
                st.session_state.scraping_result  = None
                st.session_state.save_state       = None
                st.session_state.cancel_requested = False
                cancel_placeholder.empty()
                st.rerun()

            st.success(f"✅ **{video_info['title']}** — {video_info['channel']}")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("💬 Komentar", f"{video_info['comment_count']:,}")
            mc2.metric("👁️ Views",    f"{video_info['view_count']:,}")
            mc3.metric("👍 Likes",    f"{video_info['like_count']:,}")

            # ── Langkah 2: Ambil komentar ─────────────────
            status.write("⏳ **Langkah 2/4 — Mengambil komentar dari YouTube...**")

            def on_progress(fetched, total, page):
                """Callback progress; raise StopIteration jika dibatalkan."""
                if st.session_state.cancel_requested:
                    raise StopIteration("Dibatalkan oleh pengguna.")
                pct = min(fetched / max(total, 1), 1.0)
                bar.progress(pct, text=f"Halaman {page}: {fetched}/{total} komentar")
                status.update(
                    label=f"⏳ Mengambil {fetched}/{total} komentar (halaman {page})..."
                )

            cancelled = False
            try:
                df = scraper.scrape_comments(
                    url=st.session_state.get("_scrape_url", ""),
                    start_date=st.session_state.get("_scrape_start"),
                    end_date=st.session_state.get("_scrape_end"),
                    max_comments=st.session_state.get("_scrape_max", 500),
                    progress_callback=on_progress,
                )
            except StopIteration:
                # Dibatalkan saat iterasi komentar
                cancelled = True
            except Exception as e:
                bar.empty()
                status.update(label=f"❌ Gagal: {e}", state="error")
                st.session_state.scraping         = False
                st.session_state.scraping_result  = None
                st.session_state.save_state       = None
                st.session_state.cancel_requested = False
                cancel_placeholder.empty()
                st.rerun()

            if cancelled or st.session_state.cancel_requested:
                bar.empty()
                status.update(label="🚫 Scraping dibatalkan oleh pengguna.", state="error")
                st.session_state.scraping         = False
                st.session_state.scraping_result  = None
                st.session_state.save_state       = None
                st.session_state.cancel_requested = False
                cancel_placeholder.empty()
                st.rerun()

            n_comments = len(df)

            # ── Langkah 3: Preprocessing ──────────────────
            status.write("🧹 **Langkah 3/4 — Membersihkan teks komentar...**")
            cleaned = []
            for i, c in enumerate(df["comment"]):
                # Cek pembatalan di setiap iterasi preprocessing
                if st.session_state.cancel_requested:
                    bar.empty()
                    status.update(label="🚫 Scraping dibatalkan oleh pengguna.", state="error")
                    st.session_state.scraping         = False
                    st.session_state.scraping_result  = None
                    st.session_state.save_state       = None
                    st.session_state.cancel_requested = False
                    cancel_placeholder.empty()
                    st.rerun()
                cleaned.append(preprocess(str(c)))
                bar.progress((i + 1) / n_comments, text=f"Preprocessing {i+1}/{n_comments}")
            df["clean"] = cleaned

            # ── Langkah 4: Siapkan nama file ──────────────
            status.write("📝 **Langkah 4/4 — Menyiapkan konfirmasi penyimpanan...**")
            proposed_filename = build_filename(custom_name_saved, vid_input)
            bar.empty()
            status.update(
                label=f"✅ Scraping selesai! {n_comments} komentar siap disimpan.",
                state="complete",
            )

        cancel_placeholder.empty()   # sembunyikan tombol batal
        st.session_state.scraping_result  = (vid_input, df, proposed_filename)
        st.session_state.save_state       = "pending"
        st.session_state.cancel_requested = False
        st.rerun()