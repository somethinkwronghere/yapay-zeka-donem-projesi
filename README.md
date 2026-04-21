# Yapay Zekaya Giriş – Dönem Projesi

**EUR/USD 1H Forex Zaman Serisi Tahmini ve Simülasyon Tabanlı Demo**

Sakarya Uygulamalı Bilimler Üniversitesi – Elektrik-Elektronik Mühendisliği
Yapay Zekaya Giriş (2024-2025 Bahar)

---

## 1. Proje Özeti

Bu proje, ders kılavuzundaki **"Stock Market Prediction"** başlığının döviz piyasasına uyarlanmış versiyonudur (öğretim üyesi onaylıdır). Amaç, **EUR/USD paritesinin 1 saatlik (H1) kapanış serisi** üzerinde:

1. Keşifsel Veri Analizi (EDA) ve öznitelik mühendisliği,
2. Klasik makine öğrenmesi ve derin öğrenme modelleriyle **çok adımlı tahmin (multi-step forecasting)**,
3. Geçmiş veri üzerinde **forward testing (walk-forward validation)**,
4. Model destekli **senaryo üretimi** (olası fiyat patikaları),
5. Streamlit tabanlı, canlı çalışan bir **demo arayüzü**

ortaya koymaktır.

> **Not:** Bu proje finansal tavsiye değildir. Yalnızca akademik öğrenme amacıyla geliştirilmiştir.

## 2. Veri Seti

- **Kaynak:** Kaggle – *"EURUSD - 1H - 2020-2024 September FOREX"* veri seti ve MetaTrader 5 export'u ile genişletilmiş ham seri.
- **Sembol:** EURUSD
- **Zaman Dilimi:** 1 saatlik (H1)
- **Dönem:** 2009-10-16 → günümüz (~100 bin mum)
- **Sütunlar:** `time, open, high, low, close, tick_volume, spread, real_volume`
- **Dosya:** `data/raw/eurusd_h1.csv` (ayrıca detay için `data/README.md`)

## 3. Repo Yapısı

```text
.
├── data/            # Ham ve işlenmiş veri
│   ├── raw/         # MT5/Kaggle export orijinal CSV'ler
│   └── processed/   # EDA sonrası üretilen parquet/feather dosyaları (gitignore)
├── notebooks/       # Jupyter defterleri
│   ├── 01_eda.ipynb # Hafta 3 EDA teslimi
│   └── archive/     # Önceki Colab denemeleri (MarketGPT vb.)
├── src/             # Yeniden kullanılabilir Python kaynak kodu
│   ├── data_loader.py
│   ├── features.py
│   ├── models/
│   └── utils.py
├── api/             # Streamlit / Flask demo uygulaması
│   └── app.py
├── docs/            # Proje raporu, sunum, hoca kılavuzu
├── requirements.txt
├── .gitignore
└── README.md
```

(Bu yapı ders kılavuzundaki zorunluluklarla birebir uyumludur: `/data`, `/notebooks`, `/src`, `/api`, `/docs`.)

## 4. Kurulum

```powershell
# 1) Repo'yu klonla
git clone https://github.com/somethinkwronghere/yapay-zeka-donem-projesi.git
cd yapay-zeka-donem-projesi

# 2) Sanal ortam oluştur (Python 3.10+ önerilir)
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux

# 3) Bağımlılıkları yükle
pip install -r requirements.txt

# 4) Jupyter'ı başlat
jupyter lab
```

## 5. Kullanım

### EDA defterini çalıştırma

```powershell
jupyter lab notebooks/01_eda.ipynb
```

### Streamlit demo'yu başlatma

```powershell
streamlit run api/app.py
```

## 6. Proje Yol Haritası

Ders kılavuzunun 12 haftalık takvimine göre ilerleme:

| Hafta | Teslim | Durum |
|-------|--------|-------|
| 1 | Grup formu & konu seçimi | ✅ |
| 2 | Veri envanteri | ✅ |
| **3** | **EDA raporu (`01_eda.ipynb`)** | 🚧 Devam ediyor |
| 4 | Baseline ML modeli (sklearn) | ⏳ |
| 5 | En az 3 algoritma karşılaştırması | ⏳ |
| 6 | Derin öğrenme modeli (LSTM/Transformer) | ⏳ |
| 7 | Metrik raporu | ⏳ |
| 8-9 | Model optimizasyonu | ⏳ |
| 10 | Streamlit demo | ⏳ |
| 11 | Doküman + kullanım kılavuzu | ⏳ |
| 12 | Final sunumu + canlı demo | ⏳ |

Detaylı plan için: [`docs/project_plan.md`](docs/project_plan.md)

## 7. Geliştirici

- **Ad:** Bilal (`somethinkwronghere`)
- **Öğrenci No:** b200101025
- **Bölüm:** Elektrik-Elektronik Mühendisliği – SUBÜ

## 8. Lisans

MIT License – bkz. `LICENSE`.
