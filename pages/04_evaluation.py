import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.preprocessing import preprocess
from utils.sentiment import COLOR_MAP

st.set_page_config(page_title="Evaluasi Model", page_icon="📊", layout="wide")

st.title("📊 Evaluasi Model BiLSTM")
st.markdown(
    "Pilih model BiLSTM yang sudah dilatih, upload data test berlabel, "
    "dan lihat metrik evaluasi lengkap."
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
            # Config: same-name _config.json first, fallback bilstm_config.json
            stem = os.path.splitext(fname)[0]
            cfg_same = os.path.join(bilstm_dir, f"{stem}_config.json")
            cfg_default = os.path.join(bilstm_dir, "bilstm_config.json")
            if os.path.exists(cfg_same):
                config_path = cfg_same
            elif os.path.exists(cfg_default):
                config_path = cfg_default
            else:
                config_path = None
            entries.append((f"[aktif] {fname}", model_path, config_path))

    # ── data/training/ ────────────────────────────────────────────────────────
    training_dir = "./data/training"
    if os.path.isdir(training_dir):
        for fname in sorted(os.listdir(training_dir), reverse=True):
            if not fname.endswith(".keras"):
                continue
            model_path = os.path.join(training_dir, fname)
            stem = os.path.splitext(fname)[0]
            cfg_path = os.path.join(training_dir, f"{stem}_config.json")
            config_path = cfg_path if os.path.exists(cfg_path) else None
            entries.append((f"[training] {fname}", model_path, config_path))

    return entries


# ── Sidebar: Pilih Model ──────────────────────────────────────────────────────
st.sidebar.header("⚙️ Konfigurasi")

all_models = scan_models()

if not all_models:
    st.sidebar.warning("⚠️ Belum ada model di `./models/bilstm/` maupun `./data/training/`.")
    st.info("Latih model dulu di menu **BiLSTM** sebelum evaluasi.")
    st.stop()

model_labels  = [m[0] for m in all_models]
model_paths   = [m[1] for m in all_models]
config_paths  = [m[2] for m in all_models]

# Default ke best_model.keras jika ada
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
vocab_size    = config_data.get("vocab_size", 5000)
thresholds    = config_data.get("thresholds", {})

idx_to_label  = {int(v): k for k, v in label_mapping.items()}
num_classes   = len(classes)

st.sidebar.success(f"✅ {os.path.basename(model_path)}")
st.sidebar.caption(f"Config: `{os.path.basename(config_path)}`")
st.sidebar.markdown(f"**Kelas:** {', '.join(classes)}")
st.sidebar.markdown(f"**Max len:** {max_len} | **Vocab:** {vocab_size}")
if thresholds:
    st.sidebar.markdown("**Threshold:**")
    for cls, thr in thresholds.items():
        st.sidebar.markdown(f"- `{cls}`: {thr}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Parameter")
batch_size_eval = st.sidebar.select_slider("Batch size", [16, 32, 64, 128], value=32)

# ── Load Model ────────────────────────────────────────────────────────────────
with st.spinner("Memuat model…"):
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()

st.success(f"✅ Model dimuat: **{os.path.basename(model_path)}**")
st.markdown("---")

# ── Test Data ─────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📂 Dari Folder data/", "📁 Upload CSV"])

df_test     = None
source_name = ""

with tab1:
    src_folder = st.radio(
        "Sumber folder",
        ["data/labeling/", "data/training/", "data/scraping/"],
        horizontal=True,
    )
    csv_files = []
    if os.path.isdir(src_folder):
        csv_files = sorted(
            [f for f in os.listdir(src_folder) if f.endswith(".csv")], reverse=True
        )
    if csv_files:
        sel = st.selectbox("Pilih CSV test", csv_files, key="eval_tab1")
        if sel:
            source_name = sel
            df_test = pd.read_csv(os.path.join(src_folder, sel), sep=";", encoding="utf-8-sig")
    else:
        st.info(f"Belum ada file di folder `{src_folder}`.")

with tab2:
    up = st.file_uploader("Upload CSV test berlabel", type=["csv"], key="eval_tab2")
    if up:
        source_name = up.name
        df_test = pd.read_csv(up)

if df_test is None:
    st.warning("Pilih atau upload data test terlebih dahulu.")
    st.stop()

df_test.columns = df_test.columns.str.strip()
df_test = df_test.loc[:, ~df_test.columns.str.contains("^Unnamed")]

text_col = None
for c in ["comment", "clean", "text", "teks"]:
    if c in df_test.columns:
        text_col = c
        break
if not text_col:
    text_col = df_test.select_dtypes(include="object").columns[0]

label_col = None
for c in ["sentiment", "label", "ground_truth", "actual", "class"]:
    if c in df_test.columns:
        label_col = c
        break
if label_col is None:
    st.error("❌ Kolom label tidak ditemukan. Data harus punya kolom `sentiment` atau `label`.")
    st.stop()

st.write(
    f"**Data:** `{source_name}` — {len(df_test)} baris | "
    f"**Teks:** `{text_col}` | **Label:** `{label_col}`"
)
st.dataframe(df_test[[text_col, label_col]].head(5), use_container_width=True)

# ── Evaluate ──────────────────────────────────────────────────────────────────
if st.button("🔬 Evaluasi Model", type="primary", use_container_width=True):
    with st.status("⏳ Evaluasi berjalan…", expanded=True) as status:
        bar = st.progress(0, text="Memulai…")

        # Preprocessing
        status.write("🧹 **Preprocessing teks…**")
        texts   = df_test[text_col].astype(str).tolist()
        cleaned = []
        for i, t in enumerate(texts):
            cleaned.append(preprocess(t))
            bar.progress((i + 1) / len(texts), text=f"Preprocessing {i+1}/{len(texts)}")
        df_test["clean"] = cleaned

        # Tokenize + pad
        status.write("🔄 **Tokenisasi & padding…**")
        seq = tokenizer.texts_to_sequences(cleaned)
        X   = pad_sequences(seq, maxlen=max_len, padding="pre", truncating="pre")

        # Predict
        status.write("🤖 **Menjalankan prediksi…**")
        bar.progress(0, text="Predicting…")
        proba = model.predict(X, batch_size=batch_size_eval, verbose=0)
        bar.empty()

        # Align ground-truth labels
        y_true_raw = df_test[label_col].astype(str).str.strip().str.lower()
        y_true_raw = y_true_raw.replace({
            "positive": "positif", "negative": "negatif", "neutral": "netral",
            "pos": "positif", "neg": "negatif", "net": "netral",
            "1": "positif", "-1": "negatif", "0": "netral",
        })
        true_label_map = {cls: i for i, cls in idx_to_label.items()}
        y_true_idx = np.array([true_label_map.get(lbl, -1) for lbl in y_true_raw])

        valid_mask = y_true_idx != -1
        if valid_mask.sum() == 0:
            status.update(label="❌ Tidak ada label yang cocok dengan kelas model.", state="error")
            st.stop()
        if valid_mask.sum() < len(y_true_idx):
            st.warning(
                f"⚠️ {len(y_true_idx) - valid_mask.sum()} baris dengan label tidak valid dihapus."
            )

        y_true_idx   = y_true_idx[valid_mask]
        y_pred_proba = proba[valid_mask]
        df_valid     = df_test[valid_mask].reset_index(drop=True)

        # Threshold or argmax
        if thresholds:
            from utils.bilstm import predict_with_threshold
            y_pred_idx = predict_with_threshold(y_pred_proba, classes, thresholds)
        else:
            y_pred_idx = np.argmax(y_pred_proba, axis=1)

        y_pred_label = np.array([idx_to_label[i] for i in y_pred_idx])
        status.update(label="✅ Evaluasi selesai!", state="complete")

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc         = accuracy_score(y_true_idx, y_pred_idx)
    f1_macro    = f1_score(y_true_idx, y_pred_idx, average="macro")
    f1_weighted = f1_score(y_true_idx, y_pred_idx, average="weighted")
    prec_macro  = precision_score(y_true_idx, y_pred_idx, average="macro")
    rec_macro   = recall_score(y_true_idx, y_pred_idx, average="macro")

    report_dict = classification_report(
        y_true_idx, y_pred_idx, target_names=classes, output_dict=True
    )
    cm = confusion_matrix(y_true_idx, y_pred_idx)

    # ── 1. Summary cards ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Ringkasan Metrik")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",    f"{acc*100:.2f}%")
    c2.metric("F1 Macro",    f"{f1_macro*100:.2f}%")
    c3.metric("F1 Weighted", f"{f1_weighted*100:.2f}%")
    c4.metric("Precision",   f"{prec_macro*100:.2f}%")
    c5.metric("Recall",      f"{rec_macro*100:.2f}%")

    # ── 2. Per-class table ────────────────────────────────────────────────────
    st.subheader("📋 Metrik per Kelas")
    df_report = pd.DataFrame(report_dict).T.round(4)
    df_report = df_report[~df_report.index.isin(["accuracy", "macro avg", "weighted avg"])]
    st.dataframe(df_report, use_container_width=True)

    with st.expander("📄 Classification Report (teks)", expanded=False):
        st.code(
            classification_report(y_true_idx, y_pred_idx, target_names=classes),
            language="text",
        )

    # ── 3. Per-class bar chart (F1 / Precision / Recall) ─────────────────────
    st.subheader("📊 F1 / Precision / Recall per Kelas")
    metrics_data = {
        "Precision": [report_dict[c]["precision"] for c in classes],
        "Recall":    [report_dict[c]["recall"]    for c in classes],
        "F1-Score":  [report_dict[c]["f1-score"]  for c in classes],
    }
    x     = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (metric_name, values) in enumerate(metrics_data.items()):
        bars = ax.bar(x + i * width, values, width, label=metric_name)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8,
            )
    ax.set_xticks(x + width)
    ax.set_xticklabels([c.capitalize() for c in classes])
    ax.set_ylim(0, 1.15)
    ax.set_title("Precision / Recall / F1 per Kelas", fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    plt.close(fig)

    # ── 4. Confusion matrices ─────────────────────────────────────────────────
    st.subheader("🔲 Confusion Matrix")
    col_cm1, col_cm2 = st.columns(2)
    with col_cm1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title("Confusion Matrix (count)", fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig); plt.close(fig)
    with col_cm2:
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
                    xticklabels=classes, yticklabels=classes, ax=ax, vmin=0, vmax=1)
        ax.set_title("Confusion Matrix (normalized)", fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig); plt.close(fig)

    # ── 5. Distribusi Actual vs Predicted ─────────────────────────────────────
    st.subheader("📊 Distribusi Actual vs Predicted")
    actual_counts = pd.Series(y_true_idx).map(idx_to_label).value_counts()
    pred_counts   = pd.Series(y_pred_idx).map(idx_to_label).value_counts()
    dist_df = (
        pd.DataFrame({"Actual": actual_counts, "Predicted": pred_counts})
        .fillna(0).astype(int)
    )

    col_d1, col_d2 = st.columns([1, 2])
    with col_d1:
        st.dataframe(dist_df, use_container_width=True)
    with col_d2:
        fig, ax = plt.subplots(figsize=(7, 4))
        x_pos = np.arange(len(dist_df.index))
        w     = 0.35
        bar_colors = [COLOR_MAP.get(c, "#999") for c in dist_df.index]
        ax.bar(x_pos - w / 2, dist_df["Actual"],    w, label="Actual",    color=bar_colors, alpha=0.85)
        ax.bar(x_pos + w / 2, dist_df["Predicted"], w, label="Predicted", color=bar_colors, alpha=0.50)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([c.capitalize() for c in dist_df.index])
        ax.set_title("Actual vs Predicted", fontweight="bold")
        ax.legend(); ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig); plt.close(fig)

    # ── 6. Confidence distribution ────────────────────────────────────────────
    st.subheader("📈 Distribusi Confidence per Kelas")
    df_result = df_valid.copy()
    df_result["predicted"]    = y_pred_label
    df_result["actual_label"] = [idx_to_label[i] for i in y_true_idx]
    df_result["confidence"]   = np.max(y_pred_proba, axis=1)
    df_result["correct"]      = y_true_idx == y_pred_idx

    fig, ax = plt.subplots(figsize=(8, 4))
    for cls in classes:
        cls_conf = df_result[df_result["predicted"] == cls]["confidence"]
        if len(cls_conf) > 0:
            ax.hist(cls_conf, bins=20, alpha=0.55, label=cls, color=COLOR_MAP.get(cls, "#999"))
    ax.set_xlabel("Confidence"); ax.set_ylabel("Jumlah")
    ax.set_title("Distribusi Confidence per Kelas (Predicted)", fontweight="bold")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig); plt.close(fig)

    # Correct vs wrong confidence side-by-side
    col_conf1, col_conf2 = st.columns(2)
    with col_conf1:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(df_result[df_result["correct"]]["confidence"],  bins=20, color="#2ecc71", alpha=0.8, label="Benar")
        ax.hist(df_result[~df_result["correct"]]["confidence"], bins=20, color="#e74c3c", alpha=0.7, label="Salah")
        ax.set_title("Confidence: Benar vs Salah", fontweight="bold")
        ax.set_xlabel("Confidence"); ax.legend(); ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig); plt.close(fig)
    with col_conf2:
        stats = df_result.groupby("correct")["confidence"].describe().T
        stats.columns = ["Salah", "Benar"]
        st.dataframe(stats.round(4), use_container_width=True)

    # ── 7. Error Analysis ─────────────────────────────────────────────────────
    st.subheader("❌ Error Analysis (Misclassified Samples)")
    errors = df_result[~df_result["correct"]].copy()
    if len(errors) > 0:
        errors_display = errors[[text_col, "actual_label", "predicted", "confidence"]].copy()
        errors_display.columns = ["Teks", "Actual", "Predicted", "Confidence"]
        errors_display = errors_display.sort_values("Confidence", ascending=False).head(50)
        st.dataframe(errors_display, use_container_width=True)
        st.caption(
            f"Menampilkan {min(50, len(errors))} dari {len(errors)} sampel "
            f"salah klasifikasi ({len(errors)/len(df_result)*100:.1f}% error rate)."
        )
    else:
        st.success("🎉 Tidak ada kesalahan klasifikasi!")

    # ── 8. Download ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⬇️ Download Hasil Evaluasi")

    df_export = df_valid[[text_col, label_col]].copy()
    df_export["clean"]      = df_valid["clean"]
    df_export["predicted"]  = y_pred_label
    df_export["confidence"] = np.max(y_pred_proba, axis=1).round(4)
    df_export["correct"]    = y_true_idx == y_pred_idx

    csv_bytes = df_export.to_csv(index=False, sep=";").encode("utf-8-sig")
    st.download_button(
        "⬇️ Download Hasil Evaluasi (CSV)",
        data=csv_bytes,
        file_name=f"evaluation_{os.path.splitext(source_name)[0]}.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )

    st.success("✅ Evaluasi selesai.")