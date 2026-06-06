import os
import json
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from utils.bilstm import BILSTM_PRESETS, build_model, predict_with_threshold
from utils.preprocessing import preprocess
from utils.sentiment import COLOR_MAP

st.set_page_config(page_title="Training BiLSTM", page_icon="🧠", layout="wide")

# ── Session State Init ────────────────────────────────────────────────────────
for key, default in {
    "training": False,
    "cancel_requested": False,
    "training_done": False,
    # Artifacts produced after a finished run (persisted across reruns)
    "train_artifacts": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

training = st.session_state.training

# ── Top banner ────────────────────────────────────────────────────────────────
if training:
    col_banner, col_cancel = st.columns([5, 1])
    with col_banner:
        st.markdown(
            "<div style='text-align:center;padding:0.5rem;background:#fff3cd;"
            "border:1px solid #ffc107;border-radius:8px;font-weight:600;'>"
            "⏳ Training sedang berlangsung… Jangan navigasi atau klik tombol lain.</div>",
            unsafe_allow_html=True,
        )
    with col_cancel:
        if st.button("🛑 Batalkan", type="secondary", use_container_width=True):
            st.session_state.cancel_requested = True
            st.rerun()

st.title("🧠 Training BiLSTM Pipeline")
st.markdown(
    "Latih model BiLSTM dari dataset berlabel. CSV harus memiliki kolom "
    "**`comment`** dan **`sentiment`** (positif / netral / negatif)."
)

# ── Sumber Data ───────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📂 Dari Folder data/", "📁 Upload CSV Berlabel"])

df = None
source_name = ""

with tab1:
    src_folder = st.radio(
        "Sumber folder",
        ["data/labeling/", "data/training/"],
        horizontal=True,
        disabled=training,
    )
    csv_files = []
    if os.path.isdir(src_folder):
        csv_files = sorted(
            [f for f in os.listdir(src_folder) if f.endswith(".csv")], reverse=True
        )
    if csv_files:
        sel = st.selectbox("Pilih CSV berlabel", csv_files, disabled=training, key="src_tab1")
        if sel:
            source_name = sel
            df = pd.read_csv(os.path.join(src_folder, sel), sep=";", encoding="utf-8-sig")
    else:
        st.info(f"Belum ada file di folder `{src_folder}`.")

with tab2:
    up = st.file_uploader("Upload CSV (comment + sentiment)", type=["csv"], disabled=training, key="src_tab2")
    if up:
        source_name = up.name
        df = pd.read_csv(up)

# ── Tampilkan hasil training sebelumnya jika ada ──────────────────────────────
if st.session_state.training_done and st.session_state.train_artifacts:
    _show_results = True
else:
    _show_results = False

if df is not None:
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    st.write("**Pratinjau:**")
    st.dataframe(df.head(5), use_container_width=True)

    # ── Validasi ──────────────────────────────────────────────────────────────
    text_col = "comment" if "comment" in df.columns else (
        "clean" if "clean" in df.columns else df.select_dtypes("object").columns[0]
    )
    label_col = "sentiment" if "sentiment" in df.columns else None

    if label_col is None:
        st.error("❌ Kolom `sentiment` tidak ditemukan.")
        st.stop()

    df = df.dropna(subset=[text_col, label_col])
    df = df[df[text_col].astype(str).str.strip() != ""]
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    df[label_col] = df[label_col].replace({
        "positive": "positif", "negative": "negatif", "neutral": "netral",
        "pos": "positif", "neg": "negatif", "net": "netral",
        "1": "positif", "-1": "negatif", "0": "netral",
    })
    valid_labels = {"positif", "netral", "negatif"}
    invalid = ~df[label_col].isin(valid_labels)
    if invalid.sum() > 0:
        st.warning(f"⚠️ {invalid.sum()} label tidak valid dihapus.")
        df = df[~invalid].reset_index(drop=True)

    if len(df) < 10:
        st.error("❌ Minimal 10 data valid untuk training.")
        st.stop()

    # ── Preprocessing ─────────────────────────────────────────────────────────
    prog = st.progress(0, text="Preprocessing...")
    cleaned = []
    for i, t in enumerate(df[text_col].astype(str)):
        cleaned.append(preprocess(t))
        prog.progress((i + 1) / len(df))
    df["clean"] = cleaned
    empty = df["clean"].str.strip() == ""
    if empty.sum() > 0:
        st.warning(f"⚠️ {empty.sum()} baris kosong setelah preprocessing dihapus.")
        df = df[~empty].reset_index(drop=True)
    prog.empty()

    n_rows = len(df)
    st.write(f"**Total data valid:** {n_rows} baris")

    class_dist = df[label_col].value_counts()
    cols = st.columns(3)
    for col, (lbl, cnt) in zip(cols, class_dist.items()):
        col.metric(lbl.capitalize(), f"{cnt} ({cnt/n_rows*100:.1f}%)")

    st.dataframe(df[[text_col, "clean", label_col]].head(5), use_container_width=True)

    # ── Preset ────────────────────────────────────────────────────────────────
    preset_keys = list(BILSTM_PRESETS.keys())
    if n_rows < 200:
        default_idx = 0
    elif n_rows < 1000:
        default_idx = 1
    elif n_rows < 5000:
        default_idx = 2
    elif n_rows < 20000:
        default_idx = 3
    else:
        default_idx = 4

    preset_name = st.selectbox("🎯 Pilih Preset", preset_keys, index=default_idx, disabled=training)
    hp = BILSTM_PRESETS[preset_name]

    st.markdown("### ⚙️ Konfigurasi Lanjutan")
    c1, c2, c3 = st.columns(3)
    with c1:
        use_attn = st.checkbox("🧠 Self-Attention", value=True, disabled=training)
    with c2:
        use_l2_reg = st.checkbox("🔒 L2 Regularization", value=True, disabled=training)
    with c3:
        label_sm = st.slider("🎯 Label Smoothing", 0.0, 0.2, 0.1, 0.01, disabled=training)

    custom_epochs = st.number_input(
        "🔄 Epoch", min_value=1, max_value=500, value=hp["epochs"], step=5,
        disabled=training,
        help="Override jumlah epoch dari preset. EarlyStopping tetap berjalan.",
    )

    st.info(
        f"⚙️ **{preset_name}** | {n_rows} data | "
        f"vocab {hp['max_words']} | max_len {hp['max_len']} | "
        f"LSTM {hp['lstm_units']} | dropout {hp['dropout_rate']} | "
        f"batch {hp['batch_size']} | epoch {custom_epochs}"
    )

    # ── Tombol Training ───────────────────────────────────────────────────────
    btn = st.button(
        "🏋️ Latih BiLSTM", type="primary", use_container_width=True,
        disabled=training,
    )
    if btn:
        st.session_state.training = True
        st.session_state.cancel_requested = False
        st.session_state.training_done = False
        st.session_state.train_artifacts = None
        st.rerun()

    # ── TRAINING BLOCK ────────────────────────────────────────────────────────
    if training:
        # Custom callback that checks for cancel flag each epoch
        class CancelCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if st.session_state.get("cancel_requested", False):
                    self.model.stop_training = True

        class StreamlitCallback(tf.keras.callbacks.Callback):
            def __init__(self, total_epochs, bar, table_placeholder):
                self.epoch_logs = []
                self._total = int(total_epochs)
                self._best_val_acc = 0.0
                self._bar = bar
                self._table = table_placeholder

            def on_epoch_end(self, epoch, logs=None):
                if st.session_state.get("cancel_requested", False):
                    return
                val_acc = logs.get("val_accuracy", 0)
                is_best = val_acc > self._best_val_acc
                if is_best:
                    self._best_val_acc = val_acc
                self.epoch_logs.append({
                    "Epoch": epoch + 1,
                    "Loss": round(logs.get("loss", 0), 4),
                    "Acc": round(logs.get("accuracy", 0), 4),
                    "Val Loss": round(logs.get("val_loss", 0), 4),
                    "Val Acc": round(val_acc, 4),
                    "LR": round(
                        float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)), 6
                    ),
                    "Best": "⭐" if is_best else "",
                })
                pct = (epoch + 1) / self._total
                self._bar.progress(
                    min(pct, 1.0),
                    text=f"Epoch {epoch+1}/{self._total} | acc: {logs.get('accuracy',0):.3f} | val_acc: {val_acc:.3f}",
                )
                self._table.dataframe(pd.DataFrame(self.epoch_logs), use_container_width=True)

        with st.status("⏳ Training berjalan…", expanded=True) as status:
            bar = st.progress(0, text="Inisialisasi…")
            epoch_table = st.empty()

            # ── Label Encoding ────────────────────────────────────────────
            status.write("🔢 **Langkah 5 — Label Encoding**")
            le = LabelEncoder()
            y = le.fit_transform(df[label_col])
            label_mapping = {c: int(i) for i, c in enumerate(le.classes_)}
            num_classes = len(le.classes_)

            # ── Train-Test Split ──────────────────────────────────────────
            status.write("✂️ **Langkah 6 — Train-Test Split**")
            X_texts = df["clean"].tolist()
            X_train_txt, X_test_txt, y_train, y_test = train_test_split(
                X_texts, y, test_size=hp["test_size"], random_state=42, stratify=y
            )

            # ── Tokenizer ─────────────────────────────────────────────────
            status.write("📖 **Langkah 7 — Fit Tokenizer**")
            tokenizer = KerasTokenizer(
                num_words=hp["max_words"], oov_token="<OOV>", lower=True,
                filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            )
            tokenizer.fit_on_texts(X_train_txt)
            vocab_size = min(hp["max_words"], len(tokenizer.word_index) + 1)

            # ── Text to Sequence ──────────────────────────────────────────
            status.write("🔄 **Langkah 8 — Text to Sequence**")
            X_train_seq = tokenizer.texts_to_sequences(X_train_txt)
            X_test_seq = tokenizer.texts_to_sequences(X_test_txt)

            # ── Padding ───────────────────────────────────────────────────
            status.write("🔧 **Langkah 9 — Padding**")
            X_train_pad = pad_sequences(X_train_seq, maxlen=hp["max_len"], padding="pre", truncating="pre")
            X_test_pad = pad_sequences(X_test_seq, maxlen=hp["max_len"], padding="pre", truncating="pre")
            y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
            y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

            # ── Build Model ───────────────────────────────────────────────
            status.write("🏗️ **Langkah 10 & 11 — Membangun Model**")
            model = build_model(
                vocab_size=vocab_size,
                max_len=hp["max_len"],
                embed_dim=128,
                lstm_units=hp["lstm_units"],
                dropout_rate=hp["dropout_rate"],
                dense_units=hp["dense_units"],
                num_classes=num_classes,
                use_attention=use_attn,
                use_l2=use_l2_reg,
            )
            loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_sm)
            model.compile(
                optimizer=tf.keras.optimizers.AdamW(
                    learning_rate=1e-3, weight_decay=1e-4, clipnorm=1.0
                ),
                loss=loss_fn,
                metrics=["accuracy"],
            )
            summary_lines = []
            model.summary(print_fn=lambda x: summary_lines.append(x))
            st.code("\n".join(summary_lines), language="text")

            # ── Training ──────────────────────────────────────────────────
            status.write("🚀 **Langkah 12 — Training**")
            os.makedirs("./models/bilstm", exist_ok=True)
            early_stop = EarlyStopping(
                monitor="val_accuracy", patience=8, restore_best_weights=True,
                min_delta=0.001, verbose=0,
            )
            lr_scheduler = ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=0
            )
            train_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_ckpt = f"./models/bilstm/.temp_{train_timestamp}.keras"
            checkpoint = ModelCheckpoint(
                temp_ckpt, monitor="val_accuracy", save_best_only=True, verbose=0
            )

            raw_w = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
            log_w = 1.0 + np.log(np.maximum(raw_w, 1.0))
            log_w = log_w / log_w.mean()
            class_weight_dict = dict(enumerate(log_w))

            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train_pad, y_train_cat, test_size=0.15, random_state=42,
                stratify=np.argmax(y_train_cat, axis=1),
            )

            sl_cb = StreamlitCallback(custom_epochs, bar, epoch_table)
            cancel_cb = CancelCallback()

            history = model.fit(
                X_tr, y_tr,
                validation_data=(X_val, y_val),
                epochs=custom_epochs,
                batch_size=hp["batch_size"],
                class_weight=class_weight_dict,
                callbacks=[early_stop, lr_scheduler, checkpoint, sl_cb, cancel_cb],
                verbose=0,
            )
            bar.empty()

            # Was training cancelled?
            was_cancelled = st.session_state.get("cancel_requested", False)
            if was_cancelled:
                status.update(label="🛑 Training dibatalkan oleh pengguna.", state="error")
                st.warning("Training dihentikan. Model tidak disimpan.")
                st.session_state.training = False
                st.session_state.cancel_requested = False
                if os.path.exists(temp_ckpt):
                    os.remove(temp_ckpt)
                st.rerun()

            # ── Completed normally ────────────────────────────────────────
            best_epoch = (
                max(range(len(history.history["val_accuracy"])),
                    key=lambda i: history.history["val_accuracy"][i]) + 1
            )

            st.success(
                f"✅ Training selesai — {len(history.history['loss'])} epoch | "
                f"Best epoch: {best_epoch}"
            )

            # ── Evaluation ────────────────────────────────────────────────
            status.write("📊 **Langkah 13 — Evaluasi Model**")
            y_pred_proba = model.predict(X_test_pad, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)

            acc_test    = accuracy_score(y_test, y_pred)
            f1_macro    = f1_score(y_test, y_pred, average="macro")
            f1_weighted = f1_score(y_test, y_pred, average="weighted")
            prec_macro  = precision_score(y_test, y_pred, average="macro")
            rec_macro   = recall_score(y_test, y_pred, average="macro")

            # Per-class metrics
            report_dict = classification_report(
                y_test, y_pred, target_names=le.classes_, output_dict=True
            )
            cm = confusion_matrix(y_test, y_pred)

            # Build config_data
            class_counts = {le.classes_[i]: int(np.sum(y_train == i)) for i in range(num_classes)}
            total_train = sum(class_counts.values())
            th_default = {
                cls: round(max(0.25, min(0.60, 0.5 * cnt / total_train + 0.20)), 2)
                for cls, cnt in class_counts.items()
            }
            active_th = st.session_state.get("bilstm_thresholds", th_default)
            config_data = {
                "tokenizer_config": tokenizer.to_json(),
                "label_mapping": label_mapping,
                "max_len": hp["max_len"],
                "vocab_size": vocab_size,
                "classes": le.classes_.tolist(),
                "thresholds": active_th,
            }

            # Auto-save config only (no model overwrite yet)
            os.makedirs("./models/bilstm", exist_ok=True)
            with open("./models/bilstm/bilstm_config.json", "w", encoding="utf-8") as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)

            # Cleanup temp checkpoint
            if os.path.exists(temp_ckpt):
                os.remove(temp_ckpt)

            # Persist artifacts in session_state so they survive rerun
            st.session_state.train_artifacts = {
                "model": model,
                "le": le,
                "tokenizer": tokenizer,
                "hp": hp,
                "history": history.history,
                "best_epoch": best_epoch,
                "config_data": config_data,
                "train_timestamp": train_timestamp,
                "num_classes": num_classes,
                "label_mapping": label_mapping,
                "y_test": y_test,
                "y_pred": y_pred,
                "y_pred_proba": y_pred_proba,
                "X_test_pad": X_test_pad,
                "y_train": y_train,
                "acc_test": acc_test,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "prec_macro": prec_macro,
                "rec_macro": rec_macro,
                "report_dict": report_dict,
                "cm": cm,
                "class_counts": class_counts,
                "total_train": total_train,
                "th_default": th_default,
            }

            status.update(
                label=f"✅ Training selesai — Best epoch: {best_epoch}", state="complete"
            )

        st.session_state.training = False
        st.session_state.training_done = True
        st.rerun()

# ── POST-TRAINING RESULTS ─────────────────────────────────────────────────────
if st.session_state.training_done and st.session_state.train_artifacts:
    arts = st.session_state.train_artifacts

    model         = arts["model"]
    le            = arts["le"]
    tokenizer     = arts["tokenizer"]
    hp            = arts["hp"]
    hist          = arts["history"]
    best_epoch    = arts["best_epoch"]
    config_data   = arts["config_data"]
    train_ts      = arts["train_timestamp"]
    num_classes   = arts["num_classes"]
    label_mapping = arts["label_mapping"]
    y_test        = arts["y_test"]
    y_pred        = arts["y_pred"]
    y_pred_proba  = arts["y_pred_proba"]
    X_test_pad    = arts["X_test_pad"]
    y_train       = arts["y_train"]
    acc_test      = arts["acc_test"]
    f1_macro      = arts["f1_macro"]
    f1_weighted   = arts["f1_weighted"]
    prec_macro    = arts["prec_macro"]
    rec_macro     = arts["rec_macro"]
    report_dict   = arts["report_dict"]
    cm            = arts["cm"]
    class_counts  = arts["class_counts"]
    total_train   = arts["total_train"]
    th_default    = arts["th_default"]

    # ── Section: Evaluasi ─────────────────────────────────────────────────────
    st.markdown("---")
    st.header("📊 Hasil Evaluasi Model")

    # Summary metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",    f"{acc_test*100:.2f}%")
    m2.metric("F1 Macro",    f"{f1_macro*100:.2f}%")
    m3.metric("F1 Weighted", f"{f1_weighted*100:.2f}%")
    m4.metric("Precision",   f"{prec_macro*100:.2f}%")
    m5.metric("Recall",      f"{rec_macro*100:.2f}%")

    # Training curves
    st.subheader("📈 Kurva Training")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(hist["accuracy"],     label="Train Acc", marker="o")
        ax.plot(hist["val_accuracy"], label="Val Acc",   marker="s", linestyle="--")
        ax.axvline(x=best_epoch - 1, color="gray", linestyle=":", alpha=0.7, label=f"Best (ep {best_epoch})")
        ax.set_title("Accuracy per Epoch", fontweight="bold")
        ax.legend(); ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig); plt.close(fig)
    with col_a2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(hist["loss"],     label="Train Loss", marker="o",  color="#e74c3c")
        ax.plot(hist["val_loss"], label="Val Loss",   marker="s",  linestyle="--", color="#c0392b")
        ax.axvline(x=best_epoch - 1, color="gray", linestyle=":", alpha=0.7)
        ax.set_title("Loss per Epoch", fontweight="bold")
        ax.legend(); ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig); plt.close(fig)

    # Per-class classification report
    st.subheader("📋 Classification Report per Kelas")
    report_df = pd.DataFrame(report_dict).T.round(4)
    st.dataframe(report_df, use_container_width=True)

    # Per-class bar chart
    st.subheader("📊 F1 / Precision / Recall per Kelas")
    class_labels = le.classes_.tolist()
    metrics_data = {
        "Precision": [report_dict[c]["precision"] for c in class_labels],
        "Recall":    [report_dict[c]["recall"]    for c in class_labels],
        "F1-Score":  [report_dict[c]["f1-score"]  for c in class_labels],
    }
    x = np.arange(len(class_labels))
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
    ax.set_xticklabels([c.capitalize() for c in class_labels])
    ax.set_ylim(0, 1.1)
    ax.set_title("Precision / Recall / F1 per Kelas", fontweight="bold")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig); plt.close(fig)

    # Confusion matrices (raw + normalized)
    st.subheader("🔢 Confusion Matrix")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix (raw)", fontweight="bold")
        st.pyplot(fig); plt.close(fig)
    with col_c2:
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
                    xticklabels=le.classes_, yticklabels=le.classes_,
                    ax=ax, vmin=0, vmax=1)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix (normalized)", fontweight="bold")
        st.pyplot(fig); plt.close(fig)

    # ── Section: Threshold Tuning ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🎚️ Threshold Tuning per Kelas")

    thresh_cols = st.columns(num_classes)
    thresholds = {}
    for col, cls in zip(thresh_cols, sorted(le.classes_)):
        freq_pct = class_counts.get(cls, 0) / total_train * 100
        with col:
            thresholds[cls] = st.slider(
                f"{cls} ({freq_pct:.1f}%)", 0.10, 0.80,
                th_default.get(cls, 0.40), 0.05,
                key=f"th_{cls}",
            )

    if st.button("🔁 Terapkan Threshold", type="primary"):
        y_pred_thresh = predict_with_threshold(y_pred_proba, le.classes_, thresholds)
        acc_before = accuracy_score(y_test, y_pred)
        f1_before  = f1_score(y_test, y_pred, average="macro")
        acc_after  = accuracy_score(y_test, y_pred_thresh)
        f1_after   = f1_score(y_test, y_pred_thresh, average="macro")

        comp_cols = st.columns(4)
        comp_cols[0].metric("Accuracy (sebelum)", f"{acc_before*100:.2f}%")
        comp_cols[1].metric("Accuracy (sesudah)",  f"{acc_after*100:.2f}%",
                            f"{(acc_after-acc_before)*100:+.2f}%")
        comp_cols[2].metric("Macro F1 (sebelum)", f"{f1_before*100:.2f}%")
        comp_cols[3].metric("Macro F1 (sesudah)",  f"{f1_after*100:.2f}%",
                            f"{(f1_after-f1_before)*100:+.2f}%")

        rows = [
            {
                "Kelas": cls,
                "Threshold": thresholds.get(cls, 0.5),
                "Prec (before)": round(report_dict[cls]["precision"], 3),
                "Recall (before)": round(report_dict[cls]["recall"], 3),
                "F1 (before)": round(report_dict[cls]["f1-score"], 3),
            }
            for cls in le.classes_
        ]
        st.dataframe(pd.DataFrame(rows).set_index("Kelas"), use_container_width=True)

        cm_t = confusion_matrix(y_test, y_pred_thresh)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_t, annot=True, fmt="d", cmap="Purples",
                    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_title("CM setelah Threshold", fontweight="bold")
        st.pyplot(fig); plt.close(fig)

        st.session_state["bilstm_thresholds"] = thresholds.copy()
        # Update config in artifacts too
        st.session_state.train_artifacts["config_data"]["thresholds"] = thresholds.copy()
        config_data["thresholds"] = thresholds.copy()
        st.success(f"✅ Threshold tersimpan: {thresholds}")

    # ── Section: Simpan Model ─────────────────────────────────────────────────
    st.markdown("---")
    st.header("💾 Simpan Model")

    save_name = st.text_input(
        "Nama model",
        value=f"bilstm_{train_ts}",
        help="Digunakan sebagai nama file .keras dan _config.json",
    )

    col_save1, col_save2 = st.columns(2)

    # ── Opsi 1: simpan ke data/training/ ─────────────────────────────────────
    with col_save1:
        st.markdown("#### 📁 Simpan ke `data/training/`")
        st.caption("Model baru disimpan tanpa menimpa model lama di `models/bilstm/`.")
        if st.button("📁 Simpan ke data/training/", type="primary", use_container_width=True):
            os.makedirs("./data/training", exist_ok=True)
            model_path  = f"./data/training/{save_name}.keras"
            config_path = f"./data/training/{save_name}_config.json"
            model.save(model_path)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            st.success(f"✅ Model  → `{model_path}`")
            st.success(f"✅ Config → `{config_path}`")
            st.info("💡 Buka menu **Evaluasi** atau **Prediksi** untuk menggunakan model ini.")

    # ── Opsi 2: timpa model lama di models/bilstm/ ────────────────────────────
    with col_save2:
        st.markdown("#### ⚠️ Timpa Model di `models/bilstm/`")
        st.caption("Menimpa `best_model.keras` dan `bilstm_config.json` yang sedang aktif.")
        overwrite_confirm = st.checkbox(
            "Saya yakin ingin menimpa model & config yang ada di `models/bilstm/`",
            key="overwrite_chk",
        )
        if overwrite_confirm:
            if st.button("📦 Timpa best_model.keras", type="secondary", use_container_width=True):
                model.save("./models/bilstm/best_model.keras")
                with open("./models/bilstm/bilstm_config.json", "w", encoding="utf-8") as f:
                    json.dump(config_data, f, ensure_ascii=False, indent=2)
                st.success("✅ `best_model.keras` & `bilstm_config.json` berhasil diperbarui!")
        else:
            st.button(
                "📦 Timpa best_model.keras", type="secondary",
                use_container_width=True, disabled=True,
            )

    # Download config JSON
    tok_bytes = json.dumps(config_data, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(
        "⬇️ Download Config (.json)", tok_bytes,
        f"{save_name}_config.json", "application/json",
    )

    # ── Section: Coba Prediksi ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Coba Prediksi")
    active_th = st.session_state.get("bilstm_thresholds", None)
    bilstm_input = st.text_area(
        "Masukkan komentar:",
        placeholder="Contoh: video ini sangat membantu!",
        key="predict_textarea",
    )
    if st.button("🔮 Prediksi", key="predict_btn"):
        if bilstm_input.strip():
            clean = preprocess(bilstm_input)
            if not clean.strip():
                st.warning("Teks kosong setelah preprocessing.")
            else:
                seq  = tokenizer.texts_to_sequences([clean])
                pad  = pad_sequences(seq, maxlen=hp["max_len"], padding="pre", truncating="pre")
                proba = model.predict(pad, verbose=0)[0]

                if active_th:
                    scores = {le.classes_[i]: float(proba[i]) for i in range(num_classes)}
                    passed = {k: v for k, v in scores.items() if v >= active_th.get(k, 0.5)}
                    pred_class = max(passed, key=passed.get) if passed else max(scores, key=scores.get)
                else:
                    pred_class = le.classes_[np.argmax(proba)]

                pred_conf  = float(proba[label_mapping[pred_class]])
                emoji_map  = {"positif": "😊", "netral": "😐", "negatif": "😠"}
                c1, c2 = st.columns(2)
                c1.metric("Sentimen",   f"{emoji_map.get(pred_class, '❓')} {pred_class.capitalize()}")
                c2.metric("Confidence", f"{pred_conf*100:.1f}%")

                fig, ax = plt.subplots(figsize=(5, 3))
                bar_colors = [COLOR_MAP.get(c, "#999") for c in le.classes_]
                bars = ax.barh(le.classes_, proba, color=bar_colors)
                for bar, p, cls in zip(bars, proba, le.classes_):
                    ax.text(
                        min(float(p) + 0.01, 0.93),
                        bar.get_y() + bar.get_height() / 2,
                        f"{float(p)*100:.1f}%", va="center",
                    )
                    if active_th and cls in active_th:
                        ax.plot(
                            [active_th[cls]] * 2,
                            [bar.get_y(), bar.get_y() + bar.get_height()],
                            color="black", linewidth=1.5, linestyle="--", alpha=0.6,
                        )
                ax.set_xlim(0, 1)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig); plt.close(fig)
        else:
            st.warning("Masukkan teks terlebih dahulu.")