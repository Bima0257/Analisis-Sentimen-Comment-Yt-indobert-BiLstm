# 📊 YouTube Sentiment Analysis — IndoBERT + BiLSTM

Aplikasi analisis sentimen komentar YouTube berbahasa Indonesia menggunakan dua pendekatan model: **IndoBERT (transformer pre-trained)** untuk inferensi cepat, dan **BiLSTM (deep learning dari scratch)** untuk pelatihan pada dataset kustom.

---

## 🗂️ Struktur Proyek

```
project/
├── app_bilstm.py          # Aplikasi utama Streamlit
├── slang_indo.csv         # Kamus normalisasi kata slang Indonesia
├── requirements.txt       # Daftar dependensi Python
├── models/
│   ├── taufiqdp-indonesian-sentiment/   # Cache model IndoBERT
│   ├── w11wo-roberta-sentiment/         # Cache model RoBERTa
│   └── bilstm/
│       └── bilstm_model.keras           # Model BiLSTM hasil training
└── README.md
```

---

## 🧠 Konsep Dasar

### 1. Analisis Sentimen
Analisis sentimen adalah tugas NLP (Natural Language Processing) untuk menentukan **polaritas emosi** dari sebuah teks — apakah bersifat positif, netral, atau negatif. Pada konteks ini, teks yang dianalisis adalah komentar YouTube berbahasa Indonesia.

### 2. Dua Pendekatan Model

| Aspek | IndoBERT / RoBERTa | BiLSTM |
|---|---|---|
| Tipe | Pre-trained Transformer | Dilatih dari scratch |
| Data | Sudah dilatih jutaan data | Butuh data berlabel sendiri |
| Akurasi | Tinggi (transfer learning) | Bergantung kualitas data |
| Kecepatan inferensi | Lebih lambat | Lebih cepat |
| Kebutuhan GPU | Disarankan | Opsional |
| Kustomisasi | Terbatas (fine-tune) | Penuh |

---

## 🔄 Alur Program (Flow)

```
Input Teks / CSV
      │
      ▼
┌─────────────────────┐
│   PREPROCESSING     │
│  • HTML unescape    │
│  • Hapus URL/tag    │
│  • Lowercase        │
│  • Hapus emoji      │
│  • Normalisasi char │
│  • Normalisasi slang│
└─────────────────────┘
      │
      ├──────────────────────────────────────┐
      ▼                                      ▼
┌─────────────────┐                ┌──────────────────────┐
│  IndoBERT/      │                │  BiLSTM Pipeline     │
│  RoBERTa        │                │  (Training Mode)     │
│  (Inference)    │                └──────────────────────┘
└─────────────────┘                          │
      │                           ┌──────────▼──────────┐
      ▼                           │  5. Label Encoding  │
┌─────────────────┐               └──────────┬──────────┘
│  Label +        │                          │
│  Confidence     │               ┌──────────▼──────────┐
│  Score          │               │  6. Train-Test Split│
└─────────────────┘               └──────────┬──────────┘
      │                                      │
      ▼                           ┌──────────▼──────────┐
┌─────────────────┐               │  7. Fit Tokenizer   │
│  Visualisasi    │               │     (train only)    │
│  • Pie chart    │               └──────────┬──────────┘
│  • Bar chart    │                          │
│  • Statistik    │               ┌──────────▼──────────┐
└─────────────────┘               │  8. Text to Sequence│
                                  └──────────┬──────────┘
                                             │
                                  ┌──────────▼──────────┐
                                  │  9. Padding         │
                                  └──────────┬──────────┘
                                             │
                                  ┌──────────▼──────────┐
                                  │  10. Embedding Layer│
                                  └──────────┬──────────┘
                                             │
                                  ┌──────────▼──────────┐
                                  │  11. BiLSTM Layers  │
                                  └──────────┬──────────┘
                                             │
                                  ┌──────────▼──────────┐
                                  │  12. Evaluation     │
                                  │  • Accuracy/Loss    │
                                  │  • Classification   │
                                  │    Report           │
                                  │  • Confusion Matrix │
                                  └─────────────────────┘
```

---

## 🔧 Penjelasan Detail Setiap Tahap

### Preprocessing

Sebelum masuk ke model, teks mentah perlu dibersihkan agar model dapat memproses data secara konsisten.

| Tahap | Fungsi | Contoh |
|---|---|---|
| HTML Unescape | Konversi entity HTML | `&amp;` → `&` |
| Hapus tag HTML | Bersihkan markup | `<b>teks</b>` → `teks` |
| Lowercase | Seragamkan huruf | `BAGUS` → `bagus` |
| Hapus URL | Buang link | `https://...` → ` ` |
| Hapus emoji | Konversi emoji jadi spasi | `😊` → ` ` |
| Normalisasi char berulang | Potong pengulangan | `bangggg` → `bangg` |
| Normalisasi slang | Ganti kata informal | `gak` → `tidak`, `bgt` → `sangat` |

---

### Pipeline IndoBERT / RoBERTa

**Konsep:** Model transformer yang sudah di-pre-train pada data teks Indonesia skala besar (Wikipedia, berita, media sosial). Kita hanya melakukan **inferensi** (prediksi), bukan training ulang.

**Flow:**
1. Model diunduh dari HuggingFace dan di-cache lokal (hanya sekali)
2. Teks yang sudah dipreprocess dikirim ke pipeline `text-classification`
3. Model mengembalikan skor untuk setiap label (positif/netral/negatif)
4. Label dengan skor tertinggi dipilih sebagai hasil prediksi

**Model yang tersedia:**
- `taufiqdp/indonesian-sentiment` — dilatih pada dataset SmSA + Indonesian Sentiment
- `w11wo/indonesian-roberta-base-sentiment-classifier` — akurasi 94.36%
- `w11wo/indonesian-roberta-base-indolem-sentiment-classifier-fold-0` — dilatih pada Tweet + Hotel review

---

### Pipeline BiLSTM (Langkah 5–12)

#### Langkah 5 — Label Encoding
Mengonversi label teks menjadi angka integer agar bisa diproses model.

```
"negatif" → 0
"netral"  → 1
"positif" → 2
```

Menggunakan `sklearn.preprocessing.LabelEncoder`. Label kemudian dikonversi ke format **one-hot** untuk `categorical_crossentropy`.

---

#### Langkah 6 — Train-Test Split
Dataset dibagi menjadi dua bagian:
- **Train set** — digunakan untuk melatih model
- **Test set** — digunakan untuk mengevaluasi performa model pada data yang belum pernah dilihat

Parameter `stratify=y` memastikan proporsi setiap kelas tetap seimbang di kedua split.

```
Dataset (N data)
    ├── Train set (80%)  → untuk training
    └── Test set  (20%)  → untuk evaluasi
```

---

#### Langkah 7 — Fit Tokenizer pada Train Set
`Tokenizer` Keras membangun **vocabulary** (kamus kata → index) **hanya dari data train**. Ini penting untuk menghindari *data leakage* — model tidak boleh "melihat" kata dari test set saat membangun vocabulary.

Token khusus `<OOV>` (Out Of Vocabulary) menangani kata yang muncul di test set tapi tidak ada di vocabulary train.

```python
tokenizer.fit_on_texts(X_train)   # ✅ hanya train
# BUKAN:
tokenizer.fit_on_texts(X_all)     # ❌ data leakage
```

---

#### Langkah 8 — Text to Sequence
Setiap kata dalam teks dikonversi menjadi integer sesuai index vocabulary.

```
"video ini bagus" → [142, 7, 56]
"tidak suka"      → [3, 89]
```

---

#### Langkah 9 — Padding
Karena panjang teks berbeda-beda, semua sequence disamakan panjangnya dengan menambahkan nol (`padding="post"`) atau memotong (`truncating="post"`) hingga `max_len`.

```
[142, 7, 56]          → [142, 7,  56, 0,  0,  0,  0,  0]   (panjang 8)
[3, 89, 12, 44, 7, 2] → [3,   89, 12, 44, 7,  2,  0,  0]   (panjang 8)
```

Output: matriks 2D berukuran `(jumlah_data × max_len)`.

---

#### Langkah 10 — Embedding Layer
Layer pertama dalam model neural network. Mengonversi setiap index kata menjadi **vektor berdimensi tetap** (`embed_dim`) yang dipelajari selama training.

```
index kata 56 → vektor [0.21, -0.44, 0.87, ..., 0.12]  (embed_dim dimensi)
```

Berbeda dengan one-hot encoding yang sparse, embedding menghasilkan representasi **dense** yang menangkap makna semantik kata.

---

#### Langkah 11 — BiLSTM (Bidirectional Long Short-Term Memory)
Inti dari model. LSTM adalah jenis RNN yang mampu mengingat konteks jangka panjang dalam teks.

**Bidirectional** berarti teks dibaca dari dua arah:
- **Forward** → membaca kiri ke kanan
- **Backward** → membaca kanan ke kiri

```
Teks: "video ini tidak bagus sama sekali"

Forward  →  [video] [ini] [tidak] [bagus] [sama] [sekali]
Backward ←  [sekali] [sama] [bagus] [tidak] [ini] [video]

Output BiLSTM = concat(forward_output, backward_output)
```

Ini memungkinkan model memahami konteks dari **kedua arah** sekaligus, sangat berguna untuk menangkap negasi seperti "tidak bagus" atau "kurang memuaskan".

**Arsitektur lengkap:**
```
Embedding(vocab_size, embed_dim)
    ↓
SpatialDropout1D                    ← regularisasi
    ↓
Bidirectional(LSTM(units, return_sequences=True))   ← layer 1
    ↓
Bidirectional(LSTM(units//2))       ← layer 2
    ↓
Dense(64, activation='relu')        ← fully connected
    ↓
Dropout
    ↓
Dense(num_classes, activation='softmax')  ← output
```

---

#### Langkah 12 — Evaluation
Model dievaluasi menggunakan beberapa metrik:

**Learning Curve** — memantau apakah model overfit atau underfit selama training.

**Classification Report** — metrik per kelas:

| Metrik | Penjelasan |
|---|---|
| Precision | Dari semua yang diprediksi positif, berapa % yang benar-benar positif |
| Recall | Dari semua data positif, berapa % yang berhasil terdeteksi |
| F1-Score | Rata-rata harmonik precision dan recall |
| Support | Jumlah data aktual per kelas |

**Confusion Matrix** — matriks yang menunjukkan prediksi benar dan salah per kelas secara visual.

```
               Predicted
               Negatif  Netral  Positif
Actual Negatif [  85      3       2  ]
       Netral  [   4     76       5  ]
       Positif [   1      2      92  ]
```

---

## 📦 Instalasi

```bash
# Clone atau download project
pip install streamlit pandas numpy torch transformers emoji
pip install tensorflow scikit-learn seaborn matplotlib
```

Atau install sekaligus dari `requirements.txt`:

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
streamlit
pandas
numpy
torch
transformers
emoji
tensorflow-cpu
scikit-learn
seaborn
matplotlib
```

> Gunakan `tensorflow-cpu` jika tidak memiliki GPU untuk menghemat ukuran instalasi.

---

## ▶️ Cara Menjalankan

```bash
streamlit run app_bilstm.py
```

Akses di browser: `http://localhost:8501`

---

## 📋 Format CSV

### Untuk Analisis IndoBERT (kolom `comment` saja):
```csv
comment
Video ini sangat membantu!
Penjelasannya kurang jelas
Biasa aja sih
```

### Untuk Training BiLSTM (kolom `comment` + `sentiment`):
```csv
comment,sentiment
Video ini sangat membantu!,positif
Penjelasannya kurang jelas,negatif
Biasa aja sih,netral
```

Label yang valid: `positif`, `netral`, `negatif` (case-insensitive).

---

## ⚙️ Hyperparameter BiLSTM

| Parameter | Default | Penjelasan |
|---|---|---|
| `max_words` | 10.000 | Ukuran vocabulary maksimum |
| `max_len` | 100 | Panjang sequence setelah padding |
| `embed_dim` | 128 | Dimensi vektor embedding |
| `lstm_units` | 64 | Jumlah unit pada layer LSTM |
| `dropout_rate` | 0.3 | Tingkat dropout untuk regularisasi |
| `epochs` | 20 | Jumlah epoch maksimum training |
| `test_size` | 0.2 | Proporsi data test (20%) |

---

## 🔬 Perbandingan Output

| | IndoBERT | BiLSTM |
|---|---|---|
| Input | Teks mentah (auto-preprocess) | CSV berlabel |
| Output | Label + confidence score | Label + probability per kelas |
| Visualisasi | Pie & bar chart distribusi | Learning curve + confusion matrix |
| Export | CSV hasil analisis | Model `.keras` + tokenizer `.json` |

---

## 📚 Referensi Model

- [taufiqdp/indonesian-sentiment](https://huggingface.co/taufiqdp/indonesian-sentiment) — IndoBERT fine-tuned
- [w11wo/indonesian-roberta-base-sentiment-classifier](https://huggingface.co/w11wo/indonesian-roberta-base-sentiment-classifier) — RoBERTa (akurasi 94.36%)
- Framework: [Hugging Face Transformers](https://huggingface.co/docs/transformers), [TensorFlow/Keras](https://www.tensorflow.org/), [Streamlit](https://streamlit.io/)
