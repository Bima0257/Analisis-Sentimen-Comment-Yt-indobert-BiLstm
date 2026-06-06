import streamlit as st

st.set_page_config(
    page_title="YouTube Sentiment Analysis",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
}
.sub-title {
    color: #666;
    font-size: 1rem;
    margin-bottom: 2rem;
}
.card {
    background: black;
    border: 1px solid black;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}
.card-title {
    font-weight: 700;
    font-size: 1.1rem;
    margin-bottom: 0.3rem;
}
.card-desc {
    color: #ffff;
    font-size: 0.9rem;
}
.st-emotion-cache-1y4p8pa {
    max-width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 YouTube Sentiment Analysis</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-title">Analisis sentimen komentar YouTube berbahasa Indonesia menggunakan IndoBERT + BiLSTM</div>',
            unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-icon">📥</div>
        <div class="card-title">Scraping YouTube</div>
        <div class="card-desc">
            Ambil komentar dari video YouTube via Google API. Filter tanggal, simpan CSV.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-icon">🏷️</div>
        <div class="card-title">Labeling Sentimen</div>
        <div class="card-desc">
            Label otomatis komentar (positif/netral/negatif) menggunakan model transformer IndoBERT.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-icon">🧠</div>
        <div class="card-title">Training BiLSTM</div>
        <div class="card-desc">
            Latih model BiLSTM dari dataset berlabel. Lengkap dengan evaluasi, threshold tuning, dan save model.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-icon">📊</div>
        <div class="card-title">Evaluasi Model</div>
        <div class="card-desc">
            Evaluasi model BiLSTM: accuracy, precision, recall, F1, confusion matrix, error analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-icon">🔮</div>
        <div class="card-title">Prediksi Sentimen</div>
        <div class="card-desc">
            Prediksi sentimen kalimat menggunakan model BiLSTM. Manual input atau batch dari CSV.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.info(
    "👈 **Navigasi sidebar**: Pilih menu di sebelah kiri untuk memulai.\n\n"
    "📥 **Scraping** → Ambil data komentar YouTube\n"
    "🏷️ **Labeling** → Label sentimen otomatis pakai IndoBERT\n"
    "🧠 **BiLSTM** → Training model BiLSTM dari data berlabel\n"
    "📊 **Evaluasi** → Evaluasi model dengan metrik lengkap\n"
    "🔮 **Prediksi** → Prediksi sentimen dengan model terlatih"
)
