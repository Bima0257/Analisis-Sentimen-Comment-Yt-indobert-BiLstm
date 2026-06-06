import os
import torch
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "taufiqdp/indonesian-sentiment"
LOCAL_DIR = "./models/taufiqdp-indonesian-sentiment"

LABEL_NORMALIZE = {
    "positif": "positif", "netral": "netral", "negatif": "negatif",
    "positive": "positif", "neutral": "netral", "negative": "negatif",
    "label_0": "negatif", "label_1": "netral", "label_2": "positif",
}

COLOR_MAP = {
    "positif": "#2ecc71", "netral": "#3498db", "negatif": "#e74c3c",
}


@st.cache_resource(show_spinner=False)
def load_model():
    is_cached = os.path.isdir(LOCAL_DIR) and os.path.exists(
        os.path.join(LOCAL_DIR, "config.json")
    )
    source = LOCAL_DIR if is_cached else MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source)
    if not is_cached:
        os.makedirs(LOCAL_DIR, exist_ok=True)
        tokenizer.save_pretrained(LOCAL_DIR)
        model.save_pretrained(LOCAL_DIR)
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        return_all_scores=True,
    )
    return clf, is_cached


def get_label_from_results(results):
    if isinstance(results, list) and len(results) > 0:
        if isinstance(results[0], list):
            results = results[0]
    best = max(results, key=lambda x: x["score"])
    label = LABEL_NORMALIZE.get(best["label"].lower(), best["label"].lower())
    return label, round(best["score"], 4)


def predict_single(classifier, text: str):
    if not text or len(text.strip()) == 0:
        return "netral", 0.0
    text = text[:1000]
    try:
        results = classifier(text)
        if isinstance(results[0], list):
            results = results[0]
        return get_label_from_results(results)
    except Exception:
        return "netral", 0.0


def predict_batch(classifier, texts: list, batch_size: int = 16, progress_callback=None):
    all_labels, all_scores = [], []
    n = len(texts)

    for i in range(0, n, batch_size):
        batch = [t[:1000] for t in texts[i: i + batch_size]]
        done = min(i + batch_size, n)
        try:
            results = classifier(batch)
            for res in results:
                if isinstance(res, dict):
                    res = [res]
                label, score = get_label_from_results(res)
                all_labels.append(label)
                all_scores.append(score)
        except Exception:
            for _ in batch:
                all_labels.append("netral")
                all_scores.append(0.0)

        if progress_callback:
            progress_callback(done, n)

    return all_labels, all_scores
