# Proje Planı – EUR/USD 1H AI Forecast

Bu doküman, ders kılavuzundaki 12 haftalık yaşam döngüsüne göre proje adımlarının detaylı planıdır. Her hafta için çıktı, metrik ve git teslim hedefi tanımlanmıştır.

## 1. Problem Tanımı

### 1.1 İş Problemi

Döviz piyasasında EUR/USD paritesi; makroekonomik şoklara, haber akışına ve teknik yapılara bağlı olarak yüksek volatilite gösterir. Yatırımcılar ve araştırmacılar, kısa vadeli hareketleri anlamak için veriye dayalı modeller kullanmak ister. Bu proje:

- **Geçmişe dönük eğitilmiş modellerle** belirli bir zamandaki geleceğe ait çok adımlı fiyat tahmini üretmeyi,
- **Walk-forward (forward testing)** yaklaşımıyla gerçekçi performans ölçümü yapmayı,
- **Senaryo üretimi** (olası birden çok fiyat patikası) ile belirsizliği görselleştirmeyi,
- Tüm bu akışı bir **Streamlit arayüzünde** canlı demo olarak sunmayı amaçlar.

> **Uyarı:** Bu çalışma finansal tavsiye değildir. Model çıktıları yalnızca akademik değerlendirme içindir.

### 1.2 Başarı Metrikleri

| Metrik | Türü | Hedef |
|--------|------|-------|
| MAE (Mean Absolute Error) | Regresyon | Baseline'dan < olmalı |
| RMSE | Regresyon | Baseline karşılaştırması |
| MAPE | Regresyon (%) | < %1.0 (1H ölçekte makul) |
| Directional Accuracy | Sınıflandırma türevi | > %55 (random > %50) |
| Sharpe-like Ratio | Walk-forward | Bilgi amaçlı raporlanır |

### 1.3 Paydaşlar

- **Öğretim Üyesi:** Değerlendirici, süreç danışmanı.
- **Kendim:** Geliştirici + kullanıcı.
- **Akademik okuyucu:** Final raporunu inceleyen.

---

## 2. Haftalık Plan

| Hafta | Ders Konusu | Proje Görevi | Çıktı | Durum |
|-------|-------------|--------------|-------|-------|
| 1 | YZ'ye Giriş, Proje Tanıtımı | Konu seçimi | Konu önerisi e-postası | ✅ |
| 2 | Veri Toplama & Etiketleme | MT5 + Kaggle verilerinin toplanması | `data/raw/eurusd_h1.csv` + veri envanteri | ✅ |
| **3** | **Veri Ön İşleme & EDA** | **Pandas ile temizlik, görselleştirme** | **`notebooks/01_eda.ipynb`** | 🚧 |
| 4 | Makine Öğrenmesi Temelleri | Baseline: naive + lineer regresyon | `src/models/baseline.py` + rapor | ⏳ |
| 5 | Model Seçimi & HP Tuning | XGBoost / LightGBM / RandomForest karşılaştırması | Model karşılaştırma tablosu | ⏳ |
| 6 | Derin Öğrenme Temelleri | LSTM / GRU / 1D-CNN denemesi | `src/models/dl_lstm.py` | ⏳ |
| 7 | Değerlendirme & Metrikler | Walk-forward test, tüm metrikler | `notebooks/07_metrics.ipynb` | ⏳ |
| 8-9 | Optimizasyon & Transfer Learning | En iyi modeli optimize et, gerekirse Transformer | Optimize model artifact | ⏳ |
| 10 | API / Arayüz Geliştirme | Streamlit demo | `api/app.py` çalışır hâlde | ⏳ |
| 11 | Dokümantasyon & Test | README, kullanım kılavuzu, birim testler | Rapor taslağı | ⏳ |
| 12 | Final Sunumu & Canlı Demo | 15 slayt + canlı demo | `docs/report/final.pdf` + sunum | ⏳ |

---

## 3. Modelleme Yaklaşımı

### 3.1 Görev Formülasyonu

- **Girdi (X):** Son `L` adet 1H mumun OHLCV + teknik öznitelikleri (pencere uzunluğu).
- **Çıktı (y):** Önümüzdeki `H` adet mumun kapanış serisi (çok adımlı tahmin).
- **Veri ayrımı (zaman bazlı, karışmasız):**
  - Train: 2009-10 → 2022-12
  - Validation: 2023-01 → 2023-12
  - Test: 2024-01 → günümüz

### 3.2 Öznitelikler (Planlanan)

- Fiyat tabanlı: log returns, rolling mean/std, z-score, price ratios.
- Teknik göstergeler: RSI, MACD, Bollinger Bands, ATR, EMA/SMA çaprazları.
- Zaman özellikleri: saat, gün, haftanın günü, ay sonu, seans (Asya/Avrupa/ABD).
- Volatilite: realized vol, GARCH-türevi.

### 3.3 Model Ailesi

| Aile | Aday Modeller | Rolü |
|------|---------------|------|
| Naive | Son değer, ortalama, drift | Baseline |
| Klasik istatistik | ARIMA, SARIMAX | Baseline |
| Klasik ML | Lineer Regresyon, Ridge, RandomForest, XGBoost, LightGBM | Ana ML karşılaştırması |
| Derin Öğrenme | LSTM, GRU, 1D-CNN, Transformer (opsiyonel) | Nihai model adayı |
| Olasılıksal | Quantile regression, Monte Carlo dropout | Senaryo üretimi |

### 3.4 Değerlendirme

- **Walk-forward validation:** Her periyotta model yeniden eğitilir (expanding veya rolling window).
- **Hata analizi:** Rejim bazlı (düşük/yüksek vol), seans bazlı, olağan dışı gün bazlı (NFP, FOMC, CPI).
- **Açıklanabilirlik:** SHAP feature importance; kararların yorumu.

---

## 4. Demo Senaryosu (Hafta 10)

Streamlit uygulaması şu bileşenleri içerir:

1. **Veri Seçici:** Kullanıcı başlangıç tarihi ve tahmin ufku seçer.
2. **Model Seçici:** Eğitilmiş modellerden birini seçer (baseline, XGBoost, LSTM, ...).
3. **Tahmin Grafiği:** Geçmiş + tahmin + (opsiyonel) çoklu senaryo paths.
4. **Metrik Paneli:** Seçilen pencere için MAE/RMSE/Directional Accuracy.
5. **Forward Test:** Kullanıcı, belirlenen tarihten itibaren modelin nasıl performe ettiğini canlı izler.

---

## 5. Risk Analizi

| Risk | Olasılık | Etki | Önlem |
|------|----------|------|-------|
| Overfitting | Yüksek | Yüksek | Walk-forward + early stopping + düzenlileştirme |
| Rejim değişikliği (2020 COVID, 2022 faiz şoku) | Orta | Yüksek | Rejim farkındalığı + ayrı test grubu |
| Data leakage (özniteliklerde geleceğe bakma) | Orta | Çok yüksek | Tüm feature'lar `shift(1)` ile türetilir, testler yazılır |
| Eğitim süresi | Orta | Orta | Colab GPU veya küçük pencere |
| Demo arızası | Düşük | Orta | Modeli `.pkl`/`.pt` olarak kaydet, gerekirse cache'le |

---

## 6. Etik ve Sorumluluk

- **Finansal tavsiye değildir.** README ve demo arayüzüne uyarı eklenir.
- **Veri lisansı:** Kaggle & MT5 kaynaklarına atıf yapılır.
- **Şeffaflık:** Tüm modellerin kodu ve eğitim scriptleri repo'da açık.
- **Önyargı:** Model belirli piyasa rejimlerinde (örn. trendli dönem) daha iyi olabilir; bu rapor edilir.
