"""EUR/USD AI Forecast – Streamlit demo (iskelet, Hafta 10'da doldurulacak)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_raw_eurusd

RAW_CSV = PROJECT_ROOT / "data" / "raw" / "eurusd_h1.csv"

st.set_page_config(
    page_title="EUR/USD AI Forecast",
    page_icon="💶",
    layout="wide",
)

st.title("EUR/USD 1H — AI Forecast Demo")
st.caption(
    "Yapay Zekaya Giriş – Dönem Projesi · SUBÜ EEF · "
    "Akademik amaçlıdır; finansal tavsiye değildir."
)


@st.cache_data(show_spinner=False)
def _load_data() -> pd.DataFrame:
    return load_raw_eurusd(RAW_CSV)


if not RAW_CSV.exists():
    st.error(f"Ham veri bulunamadı: {RAW_CSV}. Lütfen `data/raw/eurusd_h1.csv` dosyasını yerleştirin.")
    st.stop()

df = _load_data()

with st.sidebar:
    st.header("Ayarlar")
    horizon = st.slider("Tahmin ufku (saat)", 1, 168, 24, step=1)
    model_name = st.selectbox(
        "Model",
        ["Naive (son değer)", "Drift", "XGBoost (yakında)", "LSTM (yakında)"],
    )
    st.write(
        "Bu sayfa Hafta 10 teslimi öncesi **iskelet** sürümdür. "
        "Modeller eklenince tahmin görselleştirmeleri ve walk-forward paneli bu bölümde yer alacaktır."
    )

col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Son 30 günlük kapanış")
    last_month = df.tail(24 * 30)
    st.line_chart(last_month["close"], height=320)

with col2:
    st.subheader("Özet")
    st.metric("Toplam mum", f"{len(df):,}")
    st.metric("İlk tarih", df.index.min().strftime("%Y-%m-%d"))
    st.metric("Son tarih", df.index.max().strftime("%Y-%m-%d"))
    st.metric("Seçilen ufuk", f"{horizon} saat")

st.divider()
st.info(
    "⏳ Modelleme ve forward-testing paneli Hafta 6-10 arasında tamamlanacaktır. "
    "Şu anda veri yükleme ve arayüz kabuğu hazır."
)
