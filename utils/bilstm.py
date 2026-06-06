import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Dense, Dropout, SpatialDropout1D,
    BatchNormalization, GlobalMaxPooling1D, GlobalAveragePooling1D,
    Input, Concatenate, MultiHeadAttention, LayerNormalization, Add,
)
from tensorflow.keras.regularizers import l2


BILSTM_PRESETS = {
    "🪶 Sangat Ringan (< 200 data)": {
        "max_words": 5000,   "max_len": 50,   "lstm_units": 32,  "dropout_rate": 0.5,
        "batch_size": 8,     "epochs": 50,    "test_size": 0.25, "dense_units": 32,
    },
    "📗 Ringan (200 - 1000 data)": {
        "max_words": 10000,  "max_len": 75,   "lstm_units": 64,  "dropout_rate": 0.4,
        "batch_size": 16,    "epochs": 40,    "test_size": 0.20, "dense_units": 64,
    },
    "📘 Standar (1000 - 5000 data)": {
        "max_words": 15000,  "max_len": 100,  "lstm_units": 96,  "dropout_rate": 0.3,
        "batch_size": 32,    "epochs": 30,    "test_size": 0.20, "dense_units": 128,
    },
    "📕 Berat (5000 - 20000 data)": {
        "max_words": 25000,  "max_len": 120,  "lstm_units": 128, "dropout_rate": 0.25,
        "batch_size": 64,    "epochs": 25,    "test_size": 0.20, "dense_units": 128,
    },
    "🏆 Sangat Berat (> 20000 data)": {
        "max_words": 40000,  "max_len": 150,  "lstm_units": 196, "dropout_rate": 0.2,
        "batch_size": 128,   "epochs": 20,    "test_size": 0.20, "dense_units": 256,
    },
}


def build_model(
    vocab_size: int,
    max_len: int,
    embed_dim: int = 128,
    lstm_units: int = 64,
    dropout_rate: float = 0.3,
    dense_units: int = 64,
    num_classes: int = 3,
    use_attention: bool = True,
    use_l2: bool = True,
):
    l2_reg = l2(1e-4) if use_l2 else None
    inputs = Input(shape=(max_len,), name="Input")
    x = Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        input_length=max_len,
        embeddings_regularizer=l2(1e-5) if use_l2 else None,
        name="Embedding",
    )(inputs)
    x = SpatialDropout1D(dropout_rate, name="SpatialDropout")(x)
    bilstm_out = Bidirectional(
        LSTM(
            lstm_units,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate / 2,
            kernel_regularizer=l2_reg,
            recurrent_regularizer=l2_reg,
        ),
        name="BiLSTM_1",
    )(x)

    if use_attention:
        num_heads = 4 if lstm_units >= 64 else 2
        key_dim = max(lstm_units // num_heads, 8)
        attn_out, _ = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout_rate / 2,
            name="MultiHeadAttention",
        )(bilstm_out, bilstm_out, return_attention_scores=True)
        attn_out = Add(name="Residual")([bilstm_out, attn_out])
        attn_out = LayerNormalization(name="LayerNorm")(attn_out)
        pool_max = GlobalMaxPooling1D(name="GlobalMaxPool")(attn_out)
        pool_avg = GlobalAveragePooling1D(name="GlobalAvgPool")(attn_out)
        x = Concatenate(name="ConcatPool")([pool_max, pool_avg])
    else:
        bilstm_out2 = Bidirectional(
            LSTM(
                lstm_units // 2,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate / 2,
                kernel_regularizer=l2_reg,
            ),
            name="BiLSTM_2",
        )(bilstm_out)
        x = bilstm_out2

    x = Dense(dense_units, activation="relu",
              kernel_regularizer=l2_reg, name="Dense_Hidden")(x)
    x = BatchNormalization(name="BatchNorm")(x)
    x = Dropout(dropout_rate, name="Dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="Output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="BiLSTM_Sentiment")
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=1e-3, weight_decay=1e-4, clipnorm=1.0
        ),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    return model


def predict_with_threshold(proba_matrix, classes, thresh_dict):
    label_mapping = {cls: i for i, cls in enumerate(classes)}
    results = []
    for proba in proba_matrix:
        scores = {classes[i]: float(proba[i]) for i in range(len(classes))}
        passed = {k: v for k, v in scores.items() if v >= thresh_dict.get(k, 0.5)}
        if passed:
            pred = max(passed, key=passed.get)
        else:
            pred = max(scores, key=scores.get)
        results.append(label_mapping[pred])
    return np.array(results)
