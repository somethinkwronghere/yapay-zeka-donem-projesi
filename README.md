# Yapay Zekaya Giriş – Dönem Projesi

**EUR/USD 1H Forex Zaman Serisi Tahmini ve Simülasyon Tabanlı Demo**

Sakarya Uygulamalı Bilimler Üniversitesi – Elektrik-Elektronik Mühendisliği
Yapay Zekaya Giriş Dersi

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
├── data/                         # Ham ve işlenmiş veri
│   ├── raw/eurusd_h1.csv         # MT5 export, ~100k mum
│   └── processed/
│       └── baseline_scores.csv   # Baseline skor tablosu (02 tarafından üretilir)
├── notebooks/                    # Jupyter defterleri
│   ├── 01_eda.ipynb              # Hafta 3 – çalıştırılmış, çıktılar gömülü
│   ├── 02_baselines.ipynb        # Hafta 4 – Naive, Drift, Seasonal, MA, AR
│   └── archive/
│       └── prior_experiments/    # 13 eski Colab denemesi (Chronos, Diffusion, TimeGAN)
├── src/                          # Yeniden kullanılabilir Python kaynak kodu
│   ├── data_loader.py            # Load + train/val/test split
│   ├── features.py               # Return, rolling stat, calendar features
│   ├── metrics.py                # RMSE, MAE, MAPE, yön doğruluğu, pip RMSE
│   ├── models/baseline.py        # NaiveLastValue, DriftModel
│   └── utils.py
├── api/                          # Streamlit demo uygulaması
│   └── app.py                    # İskelet (Hafta 10'da tamamlanacak)
├── docs/
│   ├── project_plan.md           # 12 haftalık detaylı plan
│   ├── prior_work.md             # 13 eski modelin karşılaştırmalı özeti
│   ├── proje_kilavuzu.pdf        # Hoca kılavuzu
│   └── images/                   # EDA + baseline figürleri
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

| Hafta | Teslim | Durum | Çıktı |
|-------|--------|-------|-------|
| 1 | Grup formu & konu seçimi | ✅ Tamamlandı | Hoca onayı |
| 2 | Veri envanteri | ✅ Tamamlandı | `data/raw/eurusd_h1.csv` + `data/README.md` |
| 3 | EDA raporu (`01_eda.ipynb`) | ✅ Tamamlandı | Çalıştırılmış defter + 6 figür (`docs/images/eda_*.png`) + ADF / ACF / QQ / seans analizi |
| 4 | Baseline ML modeli | ✅ Tamamlandı | `02_baselines.ipynb` — Naive, Drift, Seasonal-24/168, MA(24), AR(p) + `baseline_scores.csv` |
| 5 | En az 3 algoritma karşılaştırması | ✅ Tamamlandı | `03_ml_comparison.ipynb` — RF / XGBoost / LightGBM, 25 feature, yön doğruluğu **%51.3–51.5** |
| **6** | **Derin öğrenme modeli (LSTM/Transformer)** | ⏳ Sırada | `04_deep_learning.ipynb` |
| 7 | Metrik raporu | 🟡 Kısmen | `src/metrics.py` ve baseline skor tablosu hazır; ML/DL skorları eklenince kapanır |
| 8–9 | Model optimizasyonu | ⏳ | Walk-forward tuning + GARCH ölçekleme |
| 10 | Streamlit demo | 🟡 İskelet | `api/app.py` temel yapı hazır; forward test + senaryo üretimi eklenecek |
| 11 | Doküman + kullanım kılavuzu | 🟡 Kısmen | README, `docs/project_plan.md`, `docs/prior_work.md`, `data/README.md` yazıldı; final rapor bekliyor |
| 12 | Final sunumu + canlı demo | ⏳ | Hafta 12 |

**Şu ana kadar üretilenler:**

- ✅ GitHub repo'su + 15+ anlamlı commit, 12 haftalık plan belgesi
- ✅ Önceki 13 Colab denemesinin [envanteri ve karşılaştırması](docs/prior_work.md) (Chronos, Diffusion+Transformer, TimeGAN aileleri)
- ✅ EDA — durağanlık testleri, ACF/PACF, volatilite kümelenmesi kanıtı, seans analizi
- ✅ 6 baseline modelin validation + test skorları (**naive RMSE ≈ 10 pip** → ML/DL için kırılması gereken taban)
- ✅ 3 gradient-boosting ensemble (RF/XGBoost/LightGBM) ile 25 feature'lı karşılaştırma; **yön doğruluğu baseline'ın %1.5 üzerinde**
- ✅ `src/` altında ortak yardımcılar: veri yükleyici, özellik mühendisliği (lag, volatilite, RSI, ATR, takvim), metrik kütüphanesi, baseline sınıfları

Detaylı plan için: [`docs/project_plan.md`](docs/project_plan.md)

## 7. Geliştirici

- **Ad:** Bilal Burak Tekin (`somethinkwronghere`)
- **Öğrenci No:** b200101025
- **Bölüm:** Elektrik-Elektronik Mühendisliği – SUBÜ

## 8. Lisans

MIT License – bkz. `LICENSE`.
