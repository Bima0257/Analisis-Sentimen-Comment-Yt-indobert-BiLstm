import os
import re
import uuid
from datetime import datetime
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Kelola Data", page_icon="📂", layout="wide")

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
.confirm-box {
    background: #eff6ff;
    border: 1px solid #3b82f6;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

st.title("📂 Kelola Data")
st.markdown(
    "Kelola file CSV hasil scraping (`data/scraping/`) dan hasil labeling (`data/labeling/`). "
    "Anda dapat **menambah**, **mengubah nama**, atau **menghapus** file."
)

DIRS = ["data/scraping/", "data/labeling/"]
for d in DIRS:
    os.makedirs(d, exist_ok=True)


def list_csv(directory: str):
    if not os.path.isdir(directory):
        return []
    return sorted(
        [f for f in os.listdir(directory) if f.endswith(".csv")],
        reverse=True,
    )


def safe_name(name: str) -> str:
    safe = re.sub(r"[^\w\-]", "_", name).strip("_")
    if not safe:
        safe = "unnamed"
    return safe


tab_scraping, tab_labeling = st.tabs(["📥 data/scraping/", "🏷️ data/labeling/"])

for tab, dir_path in [(tab_scraping, DIRS[0]), (tab_labeling, DIRS[1])]:
    with tab:
        csv_files = list_csv(dir_path)

        # ── Daftar file saat ini ──────────────────────────────────────────
        with st.expander("📋 File Saat Ini", expanded=True):
            if csv_files:
                rows = []
                for f in csv_files:
                    fp = os.path.join(dir_path, f)
                    size_kb = round(os.path.getsize(fp) / 1024, 1)
                    mtime = os.path.getmtime(fp)
                    rows.append({
                        "Nama File": f,
                        "Ukuran": f"{size_kb} KB",
                        "Terakhir Diubah": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info(f"Belum ada file CSV di `{dir_path}`.")

        st.markdown("---")

        # ═════════════════════════════════════════════════════════════════
        # 1. TAMBAH DATA
        # ═════════════════════════════════════════════════════════════════
        st.subheader("➕ Tambah Data")
        uploaded_file = st.file_uploader(
            f"Upload CSV ke `{dir_path}`",
            type=["csv"],
            key=f"upload_{dir_path}",
        )

        if uploaded_file is not None:
            df_preview = pd.read_csv(uploaded_file)
            st.write("**Pratinjau data:**")
            st.dataframe(df_preview.head(5), use_container_width=True)
            st.caption(f"Total: {len(df_preview)} baris × {len(df_preview.columns)} kolom")

            custom_name = st.text_input(
                "Nama file (tanpa ekstensi)",
                placeholder=os.path.splitext(uploaded_file.name)[0],
                key=f"name_{dir_path}",
                help="Kosongi untuk memakai nama file asli. UUID 8 karakter otomatis ditambahkan.",
            )

            if st.button("💾 Simpan ke Server", type="primary", use_container_width=True, key=f"save_{dir_path}"):
                stem = safe_name(custom_name) if custom_name.strip() else os.path.splitext(uploaded_file.name)[0]
                short_uid = uuid.uuid4().hex[:8]
                filename = f"{stem}_{short_uid}.csv"
                save_path = os.path.join(dir_path, filename)

                uploaded_file.seek(0)
                df_save = pd.read_csv(uploaded_file)
                df_save.to_csv(save_path, index=False, sep=";", encoding="utf-8-sig")
                st.success(f"✅ Data tersimpan: `{save_path}`")
                st.rerun()

        st.markdown("---")

        # ═════════════════════════════════════════════════════════════════
        # 2. UBAH NAMA
        # ═════════════════════════════════════════════════════════════════
        st.subheader("✏️ Ubah Nama File")

        if not csv_files:
            st.info(f"Belum ada file di `{dir_path}` untuk diubah namanya.")
        else:
            rename_selected = st.selectbox(
                "Pilih file yang ingin diubah namanya",
                csv_files,
                key=f"rename_select_{dir_path}",
            )

            old_stem = os.path.splitext(rename_selected)[0]
            new_stem = st.text_input(
                "Nama baru (tanpa ekstensi)",
                value=old_stem,
                key=f"rename_input_{dir_path}",
                help="Hanya huruf, angka, underscore, dan strip.",
            )

            if st.button("✏️ Rename", type="primary", use_container_width=True, key=f"rename_btn_{dir_path}"):
                new_stem = safe_name(new_stem)
                if not new_stem:
                    st.error("❌ Nama tidak boleh kosong.")
                elif new_stem == old_stem:
                    st.warning("⚠️ Nama baru sama dengan nama lama.")
                else:
                    new_filename = f"{new_stem}.csv"
                    new_path = os.path.join(dir_path, new_filename)
                    old_path = os.path.join(dir_path, rename_selected)

                    if os.path.exists(new_path):
                        st.error(f"❌ File `{new_filename}` sudah ada. Pilih nama lain.")
                    else:
                        try:
                            os.rename(old_path, new_path)
                            st.success(f"✅ `{rename_selected}` → `{new_filename}` berhasil diubah.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Gagal rename: {e}")

        st.markdown("---")

        # ═════════════════════════════════════════════════════════════════
        # 3. HAPUS DATA
        # ═════════════════════════════════════════════════════════════════
        st.subheader("🗑️ Hapus File")

        if not csv_files:
            st.info(f"Belum ada file di `{dir_path}` untuk dihapus.")
        else:
            del_selected = st.multiselect(
                "Pilih file yang ingin dihapus (bisa lebih dari satu)",
                csv_files,
                key=f"del_select_{dir_path}",
            )

            if del_selected:
                st.markdown("**File yang akan dihapus:**")
                del_preview = []
                for f in del_selected:
                    fp = os.path.join(dir_path, f)
                    size_kb = round(os.path.getsize(fp) / 1024, 1)
                    del_preview.append({
                        "File": f,
                        "Ukuran": f"{size_kb} KB",
                    })
                st.dataframe(pd.DataFrame(del_preview), use_container_width=True, hide_index=True)

                del_confirm = st.checkbox(
                    f"Saya yakin ingin menghapus **{len(del_selected)} file** secara permanen. Aksi ini tidak bisa dibatalkan.",
                    key=f"del_confirm_{dir_path}",
                )

                if st.button(
                    f"🗑️ Hapus {len(del_selected)} File",
                    type="primary",
                    use_container_width=True,
                    disabled=not del_confirm,
                    key=f"del_btn_{dir_path}",
                ):
                    deleted_ok = []
                    deleted_err = []
                    for f in del_selected:
                        fp = os.path.join(dir_path, f)
                        try:
                            os.remove(fp)
                            deleted_ok.append(f)
                        except Exception as e:
                            deleted_err.append((f, str(e)))

                    if deleted_ok:
                        st.success(f"✅ Berhasil dihapus: {', '.join(deleted_ok)}")
                    for f, err in deleted_err:
                        st.error(f"❌ Gagal hapus `{f}`: {err}")
                    st.rerun()
