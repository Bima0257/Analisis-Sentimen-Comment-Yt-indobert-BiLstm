import os
import json
import shutil
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

from utils.preprocessing import preprocess
from utils.sentiment import COLOR_MAP

st.set_page_config(page_title="Kelola Model", page_icon="🗂️", layout="wide")

st.title("🗂️ Kelola Model")
st.markdown(
    "Kelola model BiLSTM yang tersimpan di `data/training/` dan `models/bilstm/`. "
    "Rename, overwrite, promosikan, hapus, atau bandingkan model."
)

# ── Constants ─────────────────────────────────────────────────────────────────
TRAINING_DIR = "./data/training"
ACTIVE_DIR   = "./models/bilstm"
ACTIVE_MODEL = "best_model.keras"
ACTIVE_CFG   = "bilstm_config.json"

# ── Helper ────────────────────────────────────────────────────────────────────
def scan_dir(directory: str):
    """Return list of (stem, keras_path, config_path, mtime) for a directory."""
    entries = []
    if not os.path.isdir(directory):
        return entries
    for fname in sorted(os.listdir(directory), reverse=True):
        if not fname.endswith(".keras"):
            continue
        stem        = os.path.splitext(fname)[0]
        keras_path  = os.path.join(directory, fname)
        cfg_name    = f"{stem}_config.json"
        cfg_path    = os.path.join(directory, cfg_name)
        # fallback for active dir
        if not os.path.exists(cfg_path) and directory == ACTIVE_DIR:
            cfg_path = os.path.join(ACTIVE_DIR, ACTIVE_CFG)
        mtime = os.path.getmtime(keras_path)
        entries.append({
            "stem":       stem,
            "fname":      fname,
            "keras_path": keras_path,
            "cfg_path":   cfg_path if os.path.exists(cfg_path) else None,
            "mtime":      datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "size_mb":    round(os.path.getsize(keras_path) / 1024 / 1024, 2),
        })
    return entries


def load_config(cfg_path):
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_config(cfg_path, data):
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def model_info_row(entry, cfg):
    classes = cfg.get("classes", [])
    max_len = cfg.get("max_len", "-")
    vocab   = cfg.get("vocab_size", "-")
    return {
        "Nama File":  entry["fname"],
        "Kelas":      ", ".join(classes) if classes else "-",
        "Max Len":    max_len,
        "Vocab":      vocab,
        "Ukuran":     f"{entry['size_mb']} MB",
        "Dimodifikasi": entry["mtime"],
    }


# ── Scan directories ──────────────────────────────────────────────────────────
training_models = scan_dir(TRAINING_DIR)
active_models   = scan_dir(ACTIVE_DIR)

os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(ACTIVE_DIR,   exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — Ikhtisar Model
# ══════════════════════════════════════════════════════════════════════════════
st.header("📋 Ikhtisar Model Tersimpan")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader(f"🏷️ Model Aktif  (`models/bilstm/`)")
    if active_models:
        rows = [model_info_row(e, load_config(e["cfg_path"])) for e in active_models]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Belum ada model di `models/bilstm/`.")

with col_right:
    st.subheader(f"🗄️ Model Tersimpan  (`data/training/`)")
    if training_models:
        rows = [model_info_row(e, load_config(e["cfg_path"])) for e in training_models]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Belum ada model di `data/training/`.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — Rename Model (data/training/)
# ══════════════════════════════════════════════════════════════════════════════
st.header("✏️ Rename Model di `data/training/`")

if not training_models:
    st.info("Belum ada model di `data/training/` untuk di-rename.")
else:
    stems        = [e["stem"] for e in training_models]
    rename_idx   = st.selectbox(
        "Pilih model yang ingin di-rename",
        range(len(stems)),
        format_func=lambda i: training_models[i]["fname"],
        key="rename_select",
    )
    selected_ren = training_models[rename_idx]
    cfg_ren      = load_config(selected_ren["cfg_path"])

    col_r1, col_r2 = st.columns([2, 1])
    with col_r1:
        new_name = st.text_input(
            "Nama baru (tanpa ekstensi)",
            value=selected_ren["stem"],
            key="rename_input",
            help="Hanya huruf, angka, underscore, dan strip. Spasi tidak diizinkan.",
        )
    with col_r2:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        do_rename = st.button("✏️ Rename", type="primary", use_container_width=True)

    if do_rename:
        new_name = new_name.strip().replace(" ", "_")
        if not new_name:
            st.error("❌ Nama tidak boleh kosong.")
        elif new_name == selected_ren["stem"]:
            st.warning("⚠️ Nama baru sama dengan nama lama.")
        else:
            new_keras = os.path.join(TRAINING_DIR, f"{new_name}.keras")
            new_cfg   = os.path.join(TRAINING_DIR, f"{new_name}_config.json")

            # Check collision
            if os.path.exists(new_keras):
                st.error(f"❌ File `{new_name}.keras` sudah ada. Pilih nama lain.")
            else:
                try:
                    os.rename(selected_ren["keras_path"], new_keras)
                    if selected_ren["cfg_path"] and os.path.exists(selected_ren["cfg_path"]):
                        os.rename(selected_ren["cfg_path"], new_cfg)
                    st.success(
                        f"✅ `{selected_ren['fname']}` → `{new_name}.keras` berhasil di-rename."
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Gagal rename: {e}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — Overwrite antar model di data/training/
# ══════════════════════════════════════════════════════════════════════════════
st.header("🔄 Overwrite Model di `data/training/`")
st.markdown(
    "Timpa isi (bobot) sebuah model dengan bobot dari model lain, "
    "tanpa mengubah nama file tujuan."
)

if len(training_models) < 2:
    st.info("Dibutuhkan minimal **2 model** di `data/training/` untuk overwrite.")
else:
    col_ow1, col_ow2 = st.columns(2)
    stems_ow = [e["stem"] for e in training_models]

    with col_ow1:
        src_idx = st.selectbox(
            "📤 Sumber (model yang akan disalin bobotnya)",
            range(len(training_models)),
            format_func=lambda i: training_models[i]["fname"],
            key="ow_src",
        )
    with col_ow2:
        dst_options = [i for i in range(len(training_models)) if i != src_idx]
        dst_idx_in_opts = st.selectbox(
            "📥 Tujuan (model yang akan ditimpa)",
            dst_options,
            format_func=lambda i: training_models[i]["fname"],
            key="ow_dst",
        )

    src_entry = training_models[src_idx]
    dst_entry = training_models[dst_idx_in_opts]

    st.info(
        f"**`{src_entry['fname']}`** → menimpa → **`{dst_entry['fname']}`**  \n"
        f"Config tujuan akan diperbarui sesuai config sumber."
    )

    ow_confirm = st.checkbox(
        f"Saya yakin ingin menimpa `{dst_entry['fname']}` dengan bobot dari `{src_entry['fname']}`",
        key="ow_confirm",
    )
    if ow_confirm:
        if st.button("🔄 Overwrite Sekarang", type="primary", use_container_width=True):
            try:
                with st.spinner("Menyalin file model…"):
                    shutil.copy2(src_entry["keras_path"], dst_entry["keras_path"])
                    # Copy config too, but rename key so target filename stays
                    if src_entry["cfg_path"] and os.path.exists(src_entry["cfg_path"]):
                        shutil.copy2(src_entry["cfg_path"], dst_entry["cfg_path"] or
                                     os.path.join(TRAINING_DIR, f"{dst_entry['stem']}_config.json"))
                st.success(
                    f"✅ `{dst_entry['fname']}` berhasil ditimpa dengan bobot dari `{src_entry['fname']}`."
                )
                st.rerun()
            except Exception as e:
                st.error(f"❌ Gagal overwrite: {e}")
    else:
        st.button("🔄 Overwrite Sekarang", type="primary", use_container_width=True, disabled=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Section 4 — Promosikan ke Model Aktif (best_model.keras)
# ══════════════════════════════════════════════════════════════════════════════
st.header("🚀 Promosikan ke Model Aktif")
st.markdown(
    "Salin model dari `data/training/` ke `models/bilstm/best_model.keras` "
    "agar langsung digunakan di menu **Prediksi** dan **Evaluasi**."
)

if not training_models:
    st.info("Belum ada model di `data/training/`.")
else:
    promo_idx = st.selectbox(
        "Pilih model yang ingin dipromosikan",
        range(len(training_models)),
        format_func=lambda i: training_models[i]["fname"],
        key="promo_select",
    )
    promo_entry = training_models[promo_idx]
    promo_cfg   = load_config(promo_entry["cfg_path"])

    # Show info
    info_cols = st.columns(4)
    info_cols[0].metric("Kelas", ", ".join(promo_cfg.get("classes", ["-"])))
    info_cols[1].metric("Max Len", promo_cfg.get("max_len", "-"))
    info_cols[2].metric("Vocab", promo_cfg.get("vocab_size", "-"))
    info_cols[3].metric("Ukuran", f"{promo_entry['size_mb']} MB")

    active_path     = os.path.join(ACTIVE_DIR, ACTIVE_MODEL)
    active_cfg_path = os.path.join(ACTIVE_DIR, ACTIVE_CFG)

    if os.path.exists(active_path):
        st.warning(
            f"⚠️ `{ACTIVE_MODEL}` saat ini sudah ada "
            f"(dimodifikasi: {datetime.fromtimestamp(os.path.getmtime(active_path)).strftime('%Y-%m-%d %H:%M:%S')}). "
            "Aksi ini akan menimpanya."
        )

    promo_confirm = st.checkbox(
        f"Saya yakin ingin mempromosikan `{promo_entry['fname']}` → `best_model.keras`",
        key="promo_confirm",
    )
    if promo_confirm:
        if st.button("🚀 Promosikan Sekarang", type="primary", use_container_width=True):
            try:
                with st.spinner("Menyalin model ke slot aktif…"):
                    shutil.copy2(promo_entry["keras_path"], active_path)
                    if promo_entry["cfg_path"] and os.path.exists(promo_entry["cfg_path"]):
                        shutil.copy2(promo_entry["cfg_path"], active_cfg_path)
                st.success(
                    f"✅ `{promo_entry['fname']}` berhasil dipromosikan ke `best_model.keras`!"
                )
                st.info("💡 Menu **Prediksi** dan **Evaluasi** kini menggunakan model ini.")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Gagal promosi: {e}")
    else:
        st.button("🚀 Promosikan Sekarang", type="primary", use_container_width=True, disabled=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Section 5 — Edit Metadata Config
# ══════════════════════════════════════════════════════════════════════════════
st.header("🔧 Edit Metadata Config")
st.markdown("Ubah threshold atau metadata lain di file `_config.json` tanpa re-training.")

all_entries = (
    [(e, "training") for e in training_models] +
    [(e, "active")   for e in active_models]
)

if not all_entries:
    st.info("Tidak ada model ditemukan.")
else:
    meta_labels = [
        f"{'[aktif]' if src=='active' else '[training]'} {e['fname']}"
        for e, src in all_entries
    ]
    meta_idx = st.selectbox(
        "Pilih model untuk edit config",
        range(len(all_entries)),
        format_func=lambda i: meta_labels[i],
        key="meta_select",
    )
    meta_entry, meta_src = all_entries[meta_idx]
    meta_cfg = load_config(meta_entry["cfg_path"])

    if not meta_cfg:
        st.warning("Config tidak ditemukan atau kosong untuk model ini.")
    else:
        # Threshold editor
        thresholds = meta_cfg.get("thresholds", {})
        classes    = meta_cfg.get("classes", [])

        st.markdown("**🎚️ Threshold per Kelas**")
        if classes:
            th_cols = st.columns(len(classes))
            new_thresholds = {}
            for col, cls in zip(th_cols, sorted(classes)):
                with col:
                    new_thresholds[cls] = st.slider(
                        cls.capitalize(),
                        0.10, 0.80,
                        float(thresholds.get(cls, 0.40)),
                        0.05,
                        key=f"meta_th_{meta_idx}_{cls}",
                    )
        else:
            new_thresholds = thresholds
            st.info("Tidak ada kelas ditemukan di config.")

        # Raw JSON editor (advanced)
        with st.expander("🛠️ Edit JSON Langsung (Advanced)", expanded=False):
            raw_json = st.text_area(
                "JSON Config",
                value=json.dumps(
                    {k: v for k, v in meta_cfg.items() if k != "tokenizer_config"},
                    ensure_ascii=False, indent=2
                ),
                height=300,
                key=f"meta_raw_{meta_idx}",
                help="`tokenizer_config` disembunyikan karena ukurannya besar.",
            )

        col_save1, col_save2 = st.columns(2)
        with col_save1:
            if st.button("💾 Simpan Threshold", type="primary", use_container_width=True):
                meta_cfg["thresholds"] = new_thresholds
                save_config(meta_entry["cfg_path"], meta_cfg)
                st.success(f"✅ Threshold tersimpan: {new_thresholds}")

        with col_save2:
            if st.button("💾 Simpan JSON Langsung", type="secondary", use_container_width=True):
                try:
                    parsed = json.loads(raw_json)
                    # preserve tokenizer_config
                    if "tokenizer_config" in meta_cfg:
                        parsed["tokenizer_config"] = meta_cfg["tokenizer_config"]
                    save_config(meta_entry["cfg_path"], parsed)
                    st.success("✅ Config JSON berhasil disimpan.")
                except json.JSONDecodeError as e:
                    st.error(f"❌ JSON tidak valid: {e}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Section 6 — Hapus Model
# ══════════════════════════════════════════════════════════════════════════════
st.header("🗑️ Hapus Model")
st.markdown(
    "Hapus model dari `data/training/` secara permanen. "
    "Model di `models/bilstm/` (aktif) **tidak bisa dihapus** dari sini untuk keamanan."
)

if not training_models:
    st.info("Belum ada model di `data/training/` untuk dihapus.")
else:
    # Multi-select untuk hapus banyak sekaligus
    del_options = {e["fname"]: e for e in training_models}
    del_selected = st.multiselect(
        "Pilih model yang ingin dihapus (bisa lebih dari satu)",
        options=list(del_options.keys()),
        key="del_multiselect",
    )

    if del_selected:
        # Preview apa yang akan dihapus
        st.markdown("**File yang akan dihapus:**")
        preview_rows = []
        for fname in del_selected:
            e = del_options[fname]
            cfg = load_config(e["cfg_path"])
            has_cfg = e["cfg_path"] is not None and os.path.exists(e["cfg_path"])
            preview_rows.append({
                "Model (.keras)":   e["keras_path"],
                "Config (.json)":   e["cfg_path"] if has_cfg else "—",
                "Ukuran":           f"{e['size_mb']} MB",
                "Dimodifikasi":     e["mtime"],
            })
        st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

        del_confirm = st.checkbox(
            f"Saya yakin ingin menghapus **{len(del_selected)} model** secara permanen. "
            "Aksi ini tidak bisa dibatalkan.",
            key="del_confirm",
        )

        col_del1, col_del2 = st.columns([1, 3])
        with col_del1:
            del_btn = st.button(
                f"🗑️ Hapus {len(del_selected)} Model",
                type="primary",
                use_container_width=True,
                disabled=not del_confirm,
            )
        with col_del2:
            if not del_confirm:
                st.caption("☑️ Centang konfirmasi di atas untuk mengaktifkan tombol hapus.")

        if del_btn and del_confirm:
            deleted_ok  = []
            deleted_err = []
            for fname in del_selected:
                e = del_options[fname]
                try:
                    os.remove(e["keras_path"])
                    if e["cfg_path"] and os.path.exists(e["cfg_path"]):
                        os.remove(e["cfg_path"])
                    deleted_ok.append(fname)
                except Exception as ex:
                    deleted_err.append((fname, str(ex)))

            if deleted_ok:
                st.success(f"✅ Berhasil dihapus: {', '.join(deleted_ok)}")
            for fname, err in deleted_err:
                st.error(f"❌ Gagal hapus `{fname}`: {err}")
            st.rerun()

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Section 7 — Bandingkan Model
# ══════════════════════════════════════════════════════════════════════════════
st.header("⚖️ Bandingkan Model")
st.markdown(
    "Jalankan dua model pada data test yang sama, lalu bandingkan "
    "metrik dan confusion matrix-nya berdampingan."
)

# Kumpulkan semua model dari kedua direktori
all_model_entries = (
    [(e, "active")   for e in active_models] +
    [(e, "training") for e in training_models]
)
all_model_labels = [
    f"{'[aktif]' if src == 'active' else '[training]'} {e['fname']}"
    for e, src in all_model_entries
]

if len(all_model_entries) < 2:
    st.info("Dibutuhkan minimal **2 model** (dari direktori manapun) untuk perbandingan.")
else:
    col_cmp1, col_cmp2 = st.columns(2)
    with col_cmp1:
        cmp_a_idx = st.selectbox(
            "🔵 Model A",
            range(len(all_model_entries)),
            format_func=lambda i: all_model_labels[i],
            key="cmp_a",
        )
    with col_cmp2:
        cmp_b_options = [i for i in range(len(all_model_entries)) if i != cmp_a_idx]
        cmp_b_idx = st.selectbox(
            "🔴 Model B",
            cmp_b_options,
            format_func=lambda i: all_model_labels[i],
            key="cmp_b",
        )

    entry_a, src_a = all_model_entries[cmp_a_idx]
    entry_b, src_b = all_model_entries[cmp_b_idx]
    cfg_a = load_config(entry_a["cfg_path"])
    cfg_b = load_config(entry_b["cfg_path"])

    # ── Config comparison table ───────────────────────────────────────────────
    st.subheader("📋 Perbandingan Konfigurasi")
    cfg_compare = {
        "Properti": ["Kelas", "Max Len", "Vocab Size", "Ukuran File", "Dimodifikasi"],
        all_model_labels[cmp_a_idx]: [
            ", ".join(cfg_a.get("classes", ["-"])),
            cfg_a.get("max_len", "-"),
            cfg_a.get("vocab_size", "-"),
            f"{entry_a['size_mb']} MB",
            entry_a["mtime"],
        ],
        all_model_labels[cmp_b_idx]: [
            ", ".join(cfg_b.get("classes", ["-"])),
            cfg_b.get("max_len", "-"),
            cfg_b.get("vocab_size", "-"),
            f"{entry_b['size_mb']} MB",
            entry_b["mtime"],
        ],
    }
    st.dataframe(
        pd.DataFrame(cfg_compare).set_index("Properti"),
        use_container_width=True,
    )

    # ── Upload data test ──────────────────────────────────────────────────────
    st.subheader("📁 Data Test untuk Perbandingan")

    cmp_src_tab1, cmp_src_tab2 = st.tabs(["📂 Dari Folder data/", "📁 Upload CSV"])
    df_cmp = None
    cmp_source_name = ""

    with cmp_src_tab1:
        cmp_folder = st.radio(
            "Sumber folder",
            ["data/labeling/", "data/training/", "data/scraping/"],
            horizontal=True,
            key="cmp_folder_radio",
        )
        cmp_csv_files = []
        if os.path.isdir(cmp_folder):
            cmp_csv_files = sorted(
                [f for f in os.listdir(cmp_folder) if f.endswith(".csv")], reverse=True
            )
        if cmp_csv_files:
            cmp_sel = st.selectbox("Pilih CSV test", cmp_csv_files, key="cmp_folder_sel")
            if cmp_sel:
                cmp_source_name = cmp_sel
                df_cmp = pd.read_csv(
                    os.path.join(cmp_folder, cmp_sel), sep=";", encoding="utf-8-sig"
                )
        else:
            st.info(f"Belum ada file di `{cmp_folder}`.")

    with cmp_src_tab2:
        cmp_upload = st.file_uploader(
            "Upload CSV berlabel (harus punya kolom `comment`/`text` + `sentiment`/`label`)",
            type=["csv"],
            key="cmp_upload",
        )
        if cmp_upload:
            cmp_source_name = cmp_upload.name
            df_cmp = pd.read_csv(cmp_upload)

    if df_cmp is not None:
        df_cmp.columns = df_cmp.columns.str.strip()
        df_cmp = df_cmp.loc[:, ~df_cmp.columns.str.contains("^Unnamed")]

        # Detect columns
        cmp_text_col = None
        for c in ["comment", "clean", "text", "teks"]:
            if c in df_cmp.columns:
                cmp_text_col = c
                break
        if not cmp_text_col:
            cmp_text_col = df_cmp.select_dtypes(include="object").columns[0]

        cmp_label_col = None
        for c in ["sentiment", "label", "ground_truth", "actual"]:
            if c in df_cmp.columns:
                cmp_label_col = c
                break

        if cmp_label_col is None:
            st.error("❌ Kolom label tidak ditemukan (`sentiment` / `label`).")
        else:
            st.write(
                f"**Data:** `{cmp_source_name}` | {len(df_cmp)} baris | "
                f"Teks: `{cmp_text_col}` | Label: `{cmp_label_col}`"
            )
            st.dataframe(df_cmp[[cmp_text_col, cmp_label_col]].head(5), use_container_width=True)

            # Check kelas compatible
            classes_a = cfg_a.get("classes", [])
            classes_b = cfg_b.get("classes", [])
            if set(classes_a) != set(classes_b):
                st.warning(
                    f"⚠️ Kelas model berbeda — A: {classes_a} | B: {classes_b}. "
                    "Hasil perbandingan mungkin tidak sebanding."
                )

            if st.button("⚖️ Jalankan Perbandingan", type="primary", use_container_width=True):

                def run_model_eval(entry, cfg, df, text_col, label_col):
                    """Load model, preprocess, predict, return metrics dict."""
                    classes       = cfg.get("classes", [])
                    label_mapping = cfg.get("label_mapping", {})
                    idx_to_label  = {int(v): k for k, v in label_mapping.items()}
                    max_len_m     = cfg.get("max_len", 100)
                    thresholds_m  = cfg.get("thresholds", {})
                    num_cls       = len(classes)

                    # Tokenizer
                    tc = cfg.get("tokenizer_config", "")
                    if isinstance(tc, str):
                        tok = tf.keras.preprocessing.text.tokenizer_from_json(tc)
                    else:
                        tok = tf.keras.preprocessing.text.tokenizer_from_json(json.dumps(tc))

                    # Load model
                    mdl = tf.keras.models.load_model(entry["keras_path"])

                    # Preprocess
                    texts   = df[text_col].astype(str).tolist()
                    cleaned = [preprocess(t) for t in texts]
                    seq     = tok.texts_to_sequences(cleaned)
                    padded  = pad_sequences(seq, maxlen=max_len_m, padding="pre", truncating="pre")

                    # Predict
                    probas = mdl.predict(padded, verbose=0)

                    # Ground truth
                    y_true_raw = df[label_col].astype(str).str.strip().str.lower()
                    y_true_raw = y_true_raw.replace({
                        "positive": "positif", "negative": "negatif", "neutral": "netral",
                        "pos": "positif", "neg": "negatif", "net": "netral",
                        "1": "positif", "-1": "negatif", "0": "netral",
                    })
                    true_map   = {cls: i for i, cls in idx_to_label.items()}
                    y_true_idx = np.array([true_map.get(lbl, -1) for lbl in y_true_raw])
                    valid      = y_true_idx != -1
                    y_true_idx = y_true_idx[valid]
                    probas     = probas[valid]

                    # Threshold or argmax
                    if thresholds_m:
                        y_pred_idx = []
                        for p in probas:
                            scores = {classes[i]: float(p[i]) for i in range(num_cls)}
                            passed = {k: v for k, v in scores.items() if v >= thresholds_m.get(k, 0.5)}
                            y_pred_idx.append(
                                label_mapping[max(passed, key=passed.get)]
                                if passed else label_mapping[max(scores, key=scores.get)]
                            )
                        y_pred_idx = np.array(y_pred_idx)
                    else:
                        y_pred_idx = np.argmax(probas, axis=1)

                    report = classification_report(
                        y_true_idx, y_pred_idx, target_names=classes, output_dict=True,
                        zero_division=0,
                    )
                    cm = confusion_matrix(y_true_idx, y_pred_idx)

                    return {
                        "acc":         accuracy_score(y_true_idx, y_pred_idx),
                        "f1_macro":    f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0),
                        "f1_weighted": f1_score(y_true_idx, y_pred_idx, average="weighted", zero_division=0),
                        "precision":   precision_score(y_true_idx, y_pred_idx, average="macro", zero_division=0),
                        "recall":      recall_score(y_true_idx, y_pred_idx, average="macro", zero_division=0),
                        "report":      report,
                        "cm":          cm,
                        "classes":     classes,
                        "n_valid":     int(valid.sum()),
                    }

                with st.status("⏳ Menjalankan kedua model…", expanded=True) as cmp_status:
                    cmp_status.write(f"🔵 Memuat & mengevaluasi **{entry_a['fname']}**…")
                    try:
                        res_a = run_model_eval(entry_a, cfg_a, df_cmp, cmp_text_col, cmp_label_col)
                    except Exception as ex:
                        st.error(f"❌ Model A gagal: {ex}")
                        st.stop()

                    cmp_status.write(f"🔴 Memuat & mengevaluasi **{entry_b['fname']}**…")
                    try:
                        res_b = run_model_eval(entry_b, cfg_b, df_cmp, cmp_text_col, cmp_label_col)
                    except Exception as ex:
                        st.error(f"❌ Model B gagal: {ex}")
                        st.stop()

                    cmp_status.update(label="✅ Perbandingan selesai!", state="complete")

                label_a = all_model_labels[cmp_a_idx]
                label_b = all_model_labels[cmp_b_idx]

                # ── Summary metric cards ──────────────────────────────────────
                st.subheader("📊 Ringkasan Metrik")

                METRIC_KEYS = [
                    ("Accuracy",    "acc"),
                    ("F1 Macro",    "f1_macro"),
                    ("F1 Weighted", "f1_weighted"),
                    ("Precision",   "precision"),
                    ("Recall",      "recall"),
                ]

                header_cols = st.columns([2] + [1] * len(METRIC_KEYS))
                header_cols[0].markdown("**Model**")
                for col, (name, _) in zip(header_cols[1:], METRIC_KEYS):
                    col.markdown(f"**{name}**")

                for label, res, color in [
                    (f"🔵 {label_a}", res_a, "#1a6fc4"),
                    (f"🔴 {label_b}", res_b, "#c0392b"),
                ]:
                    row_cols = st.columns([2] + [1] * len(METRIC_KEYS))
                    row_cols[0].markdown(
                        f"<span style='color:{color};font-weight:600'>{label}</span>",
                        unsafe_allow_html=True,
                    )
                    for col, (_, key) in zip(row_cols[1:], METRIC_KEYS):
                        val_a = res_a[key]
                        val_b = res_b[key]
                        val   = res[key]
                        # Winner highlight
                        is_better = (res is res_a and val_a >= val_b) or (res is res_b and val_b > val_a)
                        badge = " 🏆" if is_better else ""
                        col.markdown(f"`{val*100:.2f}%`{badge}")

                # ── Delta table ───────────────────────────────────────────────
                st.subheader("📐 Selisih Metrik (A − B)")
                delta_rows = []
                for name, key in METRIC_KEYS:
                    diff = (res_a[key] - res_b[key]) * 100
                    winner = label_a if diff > 0 else (label_b if diff < 0 else "Draw")
                    delta_rows.append({
                        "Metrik":    name,
                        "Model A":   f"{res_a[key]*100:.2f}%",
                        "Model B":   f"{res_b[key]*100:.2f}%",
                        "Selisih":   f"{diff:+.2f}%",
                        "Unggul":    winner,
                    })
                st.dataframe(
                    pd.DataFrame(delta_rows).set_index("Metrik"),
                    use_container_width=True,
                )

                # ── Per-class bar chart side-by-side ──────────────────────────
                st.subheader("📊 F1 per Kelas — Perbandingan")
                all_classes = sorted(set(res_a["classes"]) | set(res_b["classes"]))
                f1_a = [res_a["report"].get(c, {}).get("f1-score", 0) for c in all_classes]
                f1_b = [res_b["report"].get(c, {}).get("f1-score", 0) for c in all_classes]
                x    = np.arange(len(all_classes))
                w    = 0.35

                fig, ax = plt.subplots(figsize=(8, 4))
                bars_a = ax.bar(x - w / 2, f1_a, w, label=f"🔵 A", color="#2980b9", alpha=0.85)
                bars_b = ax.bar(x + w / 2, f1_b, w, label=f"🔴 B", color="#c0392b", alpha=0.85)
                for bar, val in list(zip(bars_a, f1_a)) + list(zip(bars_b, f1_b)):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8,
                    )
                ax.set_xticks(x)
                ax.set_xticklabels([c.capitalize() for c in all_classes])
                ax.set_ylim(0, 1.15)
                ax.set_title("F1-Score per Kelas", fontweight="bold")
                ax.legend([f"A: {entry_a['fname']}", f"B: {entry_b['fname']}"])
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig); plt.close(fig)

                # ── Confusion matrices side-by-side ───────────────────────────
                st.subheader("🔲 Confusion Matrix Berdampingan")
                col_cm_a, col_cm_b = st.columns(2)

                def plot_cm(cm, classes, title, cmap, ax):
                    sns.heatmap(
                        cm, annot=True, fmt="d", cmap=cmap,
                        xticklabels=classes, yticklabels=classes, ax=ax,
                    )
                    ax.set_title(title, fontweight="bold")
                    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

                with col_cm_a:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    plot_cm(res_a["cm"], res_a["classes"], f"🔵 {entry_a['fname']}", "Blues", ax)
                    st.pyplot(fig); plt.close(fig)

                with col_cm_b:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    plot_cm(res_b["cm"], res_b["classes"], f"🔴 {entry_b['fname']}", "Reds", ax)
                    st.pyplot(fig); plt.close(fig)

                # ── Normalized confusion matrices ─────────────────────────────
                col_cmn_a, col_cmn_b = st.columns(2)
                with col_cmn_a:
                    cm_n = res_a["cm"].astype(float) / res_a["cm"].sum(axis=1)[:, np.newaxis]
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="Blues",
                                xticklabels=res_a["classes"], yticklabels=res_a["classes"],
                                ax=ax, vmin=0, vmax=1)
                    ax.set_title(f"🔵 A — Normalized", fontweight="bold")
                    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                    st.pyplot(fig); plt.close(fig)

                with col_cmn_b:
                    cm_n = res_b["cm"].astype(float) / res_b["cm"].sum(axis=1)[:, np.newaxis]
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="Reds",
                                xticklabels=res_b["classes"], yticklabels=res_b["classes"],
                                ax=ax, vmin=0, vmax=1)
                    ax.set_title(f"🔴 B — Normalized", fontweight="bold")
                    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                    st.pyplot(fig); plt.close(fig)

                # ── Per-class detail report ───────────────────────────────────
                with st.expander("📄 Classification Report Lengkap", expanded=False):
                    rep_col_a, rep_col_b = st.columns(2)
                    with rep_col_a:
                        st.markdown(f"**🔵 {entry_a['fname']}**")
                        st.dataframe(
                            pd.DataFrame(res_a["report"]).T.round(4),
                            use_container_width=True,
                        )
                    with rep_col_b:
                        st.markdown(f"**🔴 {entry_b['fname']}**")
                        st.dataframe(
                            pd.DataFrame(res_b["report"]).T.round(4),
                            use_container_width=True,
                        )

                st.info(
                    f"Data valid — 🔵 Model A: {res_a['n_valid']} baris | "
                    f"🔴 Model B: {res_b['n_valid']} baris"
                )