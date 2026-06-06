import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.preprocessing import preprocess
from utils.sentiment import COLOR_MAP

st.set_page_config(page_title="Prediksi Sentimen", page_icon="🔮", layout="wide")

st.title("🔮 Prediksi Sentimen")
st.markdown(
    "Masukkan kalimat atau upload CSV untuk diprediksi sentimennya "
    "menggunakan model BiLSTM yang sudah dilatih."
)

# ── Helper ────────────────────────────────────────────────────────────────────
def load_tokenizer_from_json(config):
    tc = config.get("tokenizer_config")
    if not tc:
        st.error("❌ Config tidak mengandung `tokenizer_config`.")
        st.stop()
    if isinstance(tc, str):
        return tf.keras.preprocessing.text.tokenizer_from_json(tc)
    return tf.keras.preprocessing.text.tokenizer_from_json(json.dumps(tc))


def scan_models():
    """
    Scan both model directories.
    Returns list of (display_label, model_path, config_path) tuples.
    """
    entries = []

    # ── models/bilstm/ ────────────────────────────────────────────────────────
    bilstm_dir = "./models/bilstm"
    if os.path.isdir(bilstm_dir):
        for fname in sorted(os.listdir(bilstm_dir)):
            if not fname.endswith(".keras"):
                continue
            model_path = os.path.join(bilstm_dir, fname)
            stem = os.path.splitext(fname)[0]
            cfg_same    = os.path.join(bilstm_dir, f"{stem}_config.json")
            cfg_default = os.path.join(bilstm_dir, "bilstm_config.json")
            config_path = cfg_same if os.path.exists(cfg_same) else (
                cfg_default if os.path.exists(cfg_default) else None
            )
            entries.append((f"[aktif] {fname}", model_path, config_path))

    # ── data/training/ ────────────────────────────────────────────────────────
    training_dir = "./data/training"
    if os.path.isdir(training_dir):
        for fname in sorted(os.listdir(training_dir), reverse=True):
            if not fname.endswith(".keras"):
                continue
            model_path = os.path.join(training_dir, fname)
            stem       = os.path.splitext(fname)[0]
            cfg_path   = os.path.join(training_dir, f"{stem}_config.json")
            config_path = cfg_path if os.path.exists(cfg_path) else None
            entries.append((f"[training] {fname}", model_path, config_path))

    return entries


# ── Sidebar: Pilih Model ──────────────────────────────────────────────────────
st.sidebar.header("⚙️ Model")

all_models = scan_models()

if not all_models:
    st.sidebar.warning("⚠️ Belum ada model di `./models/bilstm/` maupun `./data/training/`.")
    st.info("Latih model dulu di menu **BiLSTM** sebelum prediksi.")
    st.stop()

model_labels = [m[0] for m in all_models]
model_paths  = [m[1] for m in all_models]
config_paths = [m[2] for m in all_models]

default_idx = 0
for i, lbl in enumerate(model_labels):
    if "best_model.keras" in lbl:
        default_idx = i
        break

selected_idx = st.sidebar.selectbox(
    "🧠 Pilih Model",
    range(len(model_labels)),
    format_func=lambda i: model_labels[i],
    index=default_idx,
)

model_path  = model_paths[selected_idx]
config_path = config_paths[selected_idx]

if config_path is None:
    st.sidebar.error("❌ Config JSON tidak ditemukan untuk model ini.")
    st.sidebar.caption(
        "Pastikan file `<nama_model>_config.json` berada satu folder dengan `.keras`-nya."
    )
    st.stop()

with open(config_path, "r", encoding="utf-8") as f:
    config_data = json.load(f)

tokenizer     = load_tokenizer_from_json(config_data)
label_mapping = config_data.get("label_mapping", {})
classes       = config_data.get("classes", list(label_mapping.keys()))
max_len       = config_data.get("max_len", 100)
thresholds    = config_data.get("thresholds", {})
idx_to_label  = {int(v): k for k, v in label_mapping.items()}
num_classes   = len(classes)

st.sidebar.success(f"✅ {os.path.basename(model_path)}")
st.sidebar.caption(f"Config: `{os.path.basename(config_path)}`")
st.sidebar.markdown(f"**Kelas:** {', '.join(classes)}")
st.sidebar.markdown(f"**Max len:** {max_len}")
if thresholds:
    st.sidebar.markdown("**Threshold:**")
    for cls, thr in thresholds.items():
        st.sidebar.markdown(f"- `{cls}`: {thr}")

# ── Load Model ────────────────────────────────────────────────────────────────
with st.spinner("Memuat model…"):
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()

st.success(f"✅ Model siap: **{os.path.basename(model_path)}**")
st.markdown("---")


# ── Predict helpers ───────────────────────────────────────────────────────────
def predict_single(text: str):
    """Returns (pred_class, pred_conf, clean_text, proba_array) or (None,…) if empty."""
    clean = preprocess(text)
    if not clean.strip():
        return None, None, clean, None
    seq   = tokenizer.texts_to_sequences([clean])
    pad   = pad_sequences(seq, maxlen=max_len, padding="pre", truncating="pre")
    proba = model.predict(pad, verbose=0)[0]
    if thresholds:
        scores = {classes[i]: float(proba[i]) for i in range(num_classes)}
        passed = {k: v for k, v in scores.items() if v >= thresholds.get(k, 0.5)}
        pred_class = max(passed, key=passed.get) if passed else max(scores, key=scores.get)
    else:
        pred_class = classes[np.argmax(proba)]
    pred_conf = float(proba[label_mapping[pred_class]])
    return pred_class, pred_conf, clean, proba


def predict_batch(texts: list):
    """Returns (cleaned_list, [(pred_class, pred_conf, proba), …])."""
    cleaned = [preprocess(t) for t in texts]
    seq     = tokenizer.texts_to_sequences(cleaned)
    pad     = pad_sequences(seq, maxlen=max_len, padding="pre", truncating="pre")
    probas  = model.predict(pad, verbose=0)

    results = []
    for proba in probas:
        if thresholds:
            scores = {classes[i]: float(proba[i]) for i in range(num_classes)}
            passed = {k: v for k, v in scores.items() if v >= thresholds.get(k, 0.5)}
            pred   = max(passed, key=passed.get) if passed else max(scores, key=scores.get)
        else:
            pred = classes[np.argmax(proba)]
        conf = float(proba[label_mapping[pred]])
        results.append((pred, conf, proba))
    return cleaned, results


def draw_proba_bar(proba, classes, thresholds, ax):
    """Draw horizontal probability bar chart onto ax."""
    bar_colors = [COLOR_MAP.get(c, "#999") for c in classes]
    bars = ax.barh(classes, proba, color=bar_colors)
    for bar, p, cls in zip(bars, proba, classes):
        ax.text(
            min(float(p) + 0.01, 0.93),
            bar.get_y() + bar.get_height() / 2,
            f"{float(p)*100:.1f}%", va="center",
        )
        if thresholds and cls in thresholds:
            ax.plot(
                [thresholds[cls]] * 2,
                [bar.get_y(), bar.get_y() + bar.get_height()],
                color="black", linewidth=1.5, linestyle="--", alpha=0.6,
            )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.spines[["top", "right"]].set_visible(False)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["✏️ Input Manual", "📁 Batch dari CSV"])

# ════════════════════════════════════════════════════════
# Tab 1 — Input Manual
# ════════════════════════════════════════════════════════
with tab1:
    user_text = st.text_area(
        "Masukkan kalimat:",
        placeholder="Contoh: Video ini sangat membantu, penjelasannya jelas banget!",
        height=120,
    )

    if st.button("🔮 Prediksi", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("Masukkan kalimat terlebih dahulu.")
        else:
            pred_class, pred_conf, clean, proba = predict_single(user_text)

            if pred_class is None:
                st.warning("Teks kosong setelah preprocessing.")
            else:
                emoji_map = {"positif": "😊", "netral": "😐", "negatif": "😠"}
                em = emoji_map.get(pred_class, "❓")

                col1, col2, col3 = st.columns(3)
                col1.metric("Sentimen",              f"{em} {pred_class.capitalize()}")
                col2.metric("Confidence",             f"{pred_conf*100:.1f}%")
                col3.metric("Kata (setelah clean)",   f"{len(clean.split())} kata")

                with st.expander("📝 Detail teks"):
                    st.markdown(f"**Asli:** {user_text}")
                    st.markdown(f"**Clean:** {clean}")

                fig, ax = plt.subplots(figsize=(6, 3))
                draw_proba_bar(proba, classes, thresholds, ax)
                ax.set_title("Probabilitas per Kelas", fontweight="bold")
                st.pyplot(fig)
                plt.close(fig)

# ════════════════════════════════════════════════════════
# Tab 2 — Batch dari CSV
# ════════════════════════════════════════════════════════
with tab2:
    # ── Sumber data: folder atau upload ──────────────────
    src_tab_a, src_tab_b = st.tabs(["📂 Dari Folder data/", "📁 Upload CSV"])

    df_batch    = None
    source_name = ""

    with src_tab_a:
        src_folder = st.radio(
            "Sumber folder",
            ["data/scraping/", "data/labeling/", "data/training/"],
            horizontal=True,
            key="pred_folder_radio",
        )
        csv_files = []
        if os.path.isdir(src_folder):
            csv_files = sorted(
                [f for f in os.listdir(src_folder) if f.endswith(".csv")], reverse=True
            )
        if csv_files:
            sel = st.selectbox("Pilih CSV", csv_files, key="pred_tab_folder")
            if sel:
                source_name = sel
                df_batch = pd.read_csv(
                    os.path.join(src_folder, sel), sep=";", encoding="utf-8-sig"
                )
        else:
            st.info(f"Belum ada file di folder `{src_folder}`.")

    with src_tab_b:
        uploaded_file = st.file_uploader(
            "Upload CSV (kolom: comment atau text)", type=["csv"], key="pred_upload"
        )
        if uploaded_file is not None:
            source_name = uploaded_file.name
            df_batch = pd.read_csv(uploaded_file)

    # ── Preview + Prediksi ────────────────────────────────
    if df_batch is not None:
        df_batch.columns = df_batch.columns.str.strip()
        df_batch = df_batch.loc[:, ~df_batch.columns.str.contains("^Unnamed")]

        text_col = None
        for c in ["comment", "clean", "text", "teks", "content"]:
            if c in df_batch.columns:
                text_col = c
                break
        if not text_col:
            text_col = df_batch.select_dtypes(include="object").columns[0]

        st.write(
            f"**Data:** `{source_name}` | **Kolom teks:** `{text_col}` | "
            f"**Total:** {len(df_batch)} baris"
        )
        st.dataframe(df_batch[[text_col]].head(5), use_container_width=True)

        if st.button("🔮 Prediksi Batch", type="primary", use_container_width=True):
            with st.status("⏳ Prediksi batch berjalan…", expanded=True) as status:
                bar = st.progress(0, text="Memproses…")

                status.write("🧹 **Preprocessing & tokenisasi…**")
                texts          = df_batch[text_col].astype(str).tolist()
                cleaned, results = predict_batch(texts)
                bar.progress(1.0, text="Selesai")
                bar.empty()

                df_batch["clean"]      = cleaned
                df_batch["sentiment"]  = [r[0] for r in results]
                df_batch["confidence"] = [round(r[1], 4) for r in results]

                status.update(label="✅ Prediksi selesai!", state="complete")

            st.success(f"✅ {len(df_batch)} komentar diprediksi.")

            # ── Distribusi Sentimen ───────────────────────
            st.subheader("📊 Distribusi Sentimen")
            sentiment_counts = df_batch["sentiment"].value_counts()
            colors = [COLOR_MAP.get(s, "#999") for s in sentiment_counts.index]

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.pie(
                    sentiment_counts,
                    labels=[c.capitalize() for c in sentiment_counts.index],
                    autopct="%1.1f%%",
                    colors=colors,
                    startangle=90,
                    wedgeprops=dict(edgecolor="white", linewidth=2),
                )
                ax.set_title("Distribusi Sentimen", fontweight="bold")
                st.pyplot(fig); plt.close(fig)

            with col_p2:
                fig, ax = plt.subplots(figsize=(5, 4))
                bars = ax.bar(
                    [c.capitalize() for c in sentiment_counts.index],
                    sentiment_counts.values,
                    color=colors,
                    edgecolor="white",
                    width=0.5,
                )
                for b, v in zip(bars, sentiment_counts.values):
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        b.get_height() + 0.3,
                        str(v), ha="center", fontweight="bold",
                    )
                ax.set_title("Jumlah per Sentimen", fontweight="bold")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig); plt.close(fig)

            # ── Distribusi Confidence ─────────────────────
            st.subheader("📈 Distribusi Confidence per Kelas")
            fig, ax = plt.subplots(figsize=(8, 3))
            for cls in classes:
                cls_conf = df_batch[df_batch["sentiment"] == cls]["confidence"]
                if len(cls_conf) > 0:
                    ax.hist(
                        cls_conf, bins=20, alpha=0.55,
                        label=cls.capitalize(), color=COLOR_MAP.get(cls, "#999"),
                    )
            ax.set_xlabel("Confidence"); ax.set_ylabel("Jumlah")
            ax.set_title("Distribusi Confidence per Kelas", fontweight="bold")
            ax.legend(); ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig); plt.close(fig)

            # ── Tabel Hasil ───────────────────────────────
            st.subheader("📋 Hasil Prediksi")
            display_cols = [text_col, "clean", "sentiment", "confidence"]
            # include ground truth column if present
            for gt_col in ["sentiment_asli", "label", "ground_truth"]:
                if gt_col in df_batch.columns and gt_col not in display_cols:
                    display_cols.insert(2, gt_col)
                    break
            st.dataframe(df_batch[display_cols], use_container_width=True)

            # ── Download ──────────────────────────────────
            st.markdown("---")
            stem      = os.path.splitext(source_name)[0] if source_name else "batch"
            csv_bytes = df_batch.to_csv(index=False, sep=";").encode("utf-8-sig")
            st.download_button(
                "⬇️ Download Hasil CSV",
                data=csv_bytes,
                file_name=f"prediction_{stem}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )