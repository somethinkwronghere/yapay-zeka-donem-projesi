# Önceki Çalışmalar – Model Denemeleri Envanteri

**Proje:** EUR/USD 1H Forex Zaman Serisi Tahmini
**Ders:** Yapay Zekaya Giriş (SUBÜ EEF) – Dönem Projesi
**Doküman Tipi:** Ödev **"Literatür / Önceki Çalışmalar"** bölümü için envanter + karşılaştırma
**Öğrenci:** Bilal – `b200101025`

---

## 1. Amaç

Bu doküman, proje kapsamına girmeden önce EUR/USD paritesi üzerinde farklı yapay zekâ yaklaşımlarını denediğim **13 Colab defterini** tek çatıda özetler. Her defter:

- Ne yapmak için yazıldı,
- Hangi mimariyi/matematiği kullandı,
- Avantaj-dezavantajları neydi,
- Dönem projesinin nihai mimarisine nasıl katkı sağladı

başlıkları altında incelenir. Bölüm 7'de karşılaştırmalı tablo, Bölüm 8'de ise dönem projesine taşınan dersler yer alır.

> **Not:** Bu defterler proje iskeletinden **bağımsız** olarak Colab'da yapılmış pilot çalışmalardır. Final projede doğrudan yeniden kullanılmamakta; ancak mimari tercihlerin **gerekçelendirilmesi** için kritik referanstırlar. Bu nedenle "önceki çalışmalar" bölümüne eklenmiştir.

## 2. Genel Yaklaşım Ailesi

Defterleri üç aileye ayırabiliriz:

| Aile | Amaç | Temsil Ettiği Paradigma |
|------|------|--------------------------|
| **A. Chronos (foundation model)** | Sıfır-örnek (zero-shot) tahmin | Önceden eğitilmiş büyük zaman serisi dil modeli |
| **B. Diffusion + Transformer** | Olasılıksal üretim (DDPM) | Üretken derin öğrenme (Kronos-style) |
| **C. TimeGAN ailesi** | Sentetik OHLCV üretimi | GAN tabanlı zaman serisi augmentation |

## 3. Ortak Zemin: Veri Temsili

Tüm defterlerin çıkış noktası aynıdır:

- **Ham girdi:** EUR/USD 1H mumları (`open, high, low, close, tick_volume, spread`)
- **Türetilmiş temsil:** Log getirisi tabanlı özellikler

  \[r_t = \log(P_t) - \log(P_{t-1})\]

- **Özellik vektörü (minimal):** `log_ret_open_norm, log_ret_high_norm, log_ret_low_norm, log_ret_close_norm` → `[-1, 1]` aralığına sembolik ölçeklenmiş
- **Özellik vektörü (genişletilmiş):** +ATR(14), +RSI(14), +gerçekleşmiş volatilite, +hacim oranı, +saat/gün sin-cos kodlaması

---

## 4. Dosya Haritası (Yeniden Adlandırma ile)

| Yeni İsim | Eski İsim | KB | Aile | İçerik |
|-----------|-----------|----|------|--------|
| `01_chronos_basic.ipynb` | Amazon_Chronos.ipynb | 99 | A | Chronos T5 ile tek yollu (tek sample) tahmin |
| `02_chronos_ohlc_synthetic.ipynb` | Chronos_EURUSD_Synthetic.ipynb | 196 | A | Chronos T5 + OHLC dressing + chunked generation (5000 mum) |
| `03_chronos_draft_notes.ipynb` | Untitled1.ipynb | 9 | A | Chronos pipeline'ının Türkçe notlar eşliğinde taslak hâli |
| `10_diffusion_transformer_v1.ipynb` | fu.ipynb | 4120 | B | DDPM + Transformer (Kronos-style) – ilk tam sürüm |
| `11_diffusion_transformer_v2.ipynb` | fui.ipynb | 4018 | B | Aynı mimarinin iyileştirme turu (OHLC constraint, CFG) |
| `20_timegan_strategy6_eda.ipynb` | strategy_6_analysis.ipynb | 402 | C | TimeGAN öncesi veri hazırlık/EDA (Strategy 6) |
| `30_timegan_baseline_4feat.ipynb` | TimeGAN_Colab.ipynb | 376 | C | Klasik TimeGAN baseline (Yoon 2019) – 4 feature |
| `31_timegan_full_training_a100.ipynb` | timegan_colab_training.ipynb | 3493 | C | TimeGAN tam eğitim notebook'u (mixed precision) |
| `32_timegan_forex_compact.ipynb` | TimeGAN_Forex(1).ipynb | 39 | C | TimeGAN kompakt varyant |
| `33_timegan_forex_11feat.ipynb` | TimeGAN_Forex.ipynb | 387 | C | TimeGAN 11 feature (teknik göstergeler dahil) |
| `34_timegan_forex_run_outputs.ipynb` | TimeGAN_Forex1.ipynb | 884 | C | Önceki defterin uzun run çıktıları (grafikler gömülü) |
| `35_timegan_improved_wgangp.ipynb` | TimeGAN_Forex_Improved (1).ipynb | 30 | C | En gelişmiş TimeGAN: WGAN-GP + TTUR + vol/tail losses |
| `36_timegan_pro_a100.ipynb` | TimeGAN_Pro.ipynb | 144 | C | LayerNorm + LR decay + moment matching |

---

## 5. Aile A: Chronos Denemeleri

### 5.1 `01_chronos_basic.ipynb` — Sıfır-Örnek Tahmin Denemesi

**Ne yapıyor?**
- HuggingFace üzerinden `amazon/chronos-t5-base` modelini yükler.
- CSV'den normalize log getirisi serisi okur (`log_ret_close_norm`).
- Kullanıcının verdiği bir **başlangıç fiyatından** (örn. `1.1675`) başlayarak modelden `FORECAST_LENGTH=48` saat öngörü ister.
- Modelin ürettiği getiri dizisini fiyata geri çevirip (*P_t = P_{t-1} · e^{r_t}*) mum grafiği olarak çizer.
- Open/High/Low, Close etrafında heuristik "wick" (fitil) eklenerek sentezlenir.

**Kritik kod:**
```python
pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-base", ...)
forecast = pipeline.predict(context_tensor, prediction_length=48, num_samples=1)
# Fiyat rekonstrüksiyonu: next_close = current_close * np.exp(ret)
```

**Güçlü yanlar:**
- **Sıfır eğitim maliyeti** – model kutudan çıkar çıkmaz çalışır.
- Kısa vadede makul trend yakalar.

**Zayıf yanlar:**
- `num_samples=1` → belirsizlik görünmez (tek yol).
- OHLC *kurallı* değil; H≥O,C ve L≤O,C kısıtı açık enforce edilmiyor.
- Normalize veriyi ham log-getiri gibi yorumluyor; ölçek tutarsızlığı riski büyük.

### 5.2 `02_chronos_ohlc_synthetic.ipynb` — İyileştirilmiş Chronos Pipeline

**Ne eklenmiş?**
- **GPU/CPU'ya göre otomatik model seçimi** (`chronos-t5-large` vs `small`).
- **Chunked generation:** 24 saatlik bloklar hâlinde toplam 5000 saate kadar (~208 gün) üretim.
- **Sliding-window context:** Her chunk sonunda bağlam yeni tahminlerle güncellenir (uzun üretimde bozulma sınırlı).
- **Stokastik fitil modeli:** Lognormal dağılım ile *dinamik* volatilite (son 500 tikin pct_change std'si baz alınır).
- **Volume simülasyonu:** Mum gövde büyüklüğüne göre orantılı.
- Sonuç `sentetik_gelecek_verisi.csv` olarak dışa aktarılır.

**Güçlü yanlar:**
- Çok daha kullanıma hazır ve anlaşılır bir akış.
- Uzun ufuklu üretim yapabiliyor.
- `np.max/min` ile OHLC sanity enforce ediliyor: `high ≥ max(open,close)`, `low ≤ min(open,close)`.

**Zayıf yanlar:**
- Model yine sadece *close* serisini modeliyor; OHLC korelasyonları heuristik.
- Walk-forward metrik değerlendirmesi yok (görsel inceleme ile yetiniliyor).

### 5.3 `03_chronos_draft_notes.ipynb` — Taslak

`02`'nin Türkçe açıklamalarla hazırlanan bir ön taslağı. Kod blokları ile yorum satırları karışık; prodüksiyon için değil, not tutmak için yazılmış.

---

## 6. Aile B: Diffusion + Transformer Denemeleri

### 6.1 `10_diffusion_transformer_v1.ipynb` / `11_diffusion_transformer_v2.ipynb` — Forex Diffusion (Kronos-Style)

Bu iki defter, projenin **en iddialı ve en karmaşık** denemesidir. Mimari **DDPM (Denoising Diffusion Probabilistic Model)** + Transformer backbone'u üzerine kurulu. Kronos (sadece 2024'te Google tarafından duyurulan finansal zaman serisi foundation modeli) mantığına yakın bir proof-of-concept'tir.

**Mimari Özet:**
- **Girdi:** `(batch, seq_len=200, 5)` – OHLCV yüzde getirisi olarak normalize edilmiş.
- **Diffusion Schedule:** Cosine (Improved DDPM, Nichol & Dhariwal 2021) – `T=1000` step.
- **Backbone:** `ForexDiffusionTransformer`
  - `d_model=256 (v1)` → `384 (v2)`
  - `n_heads=8`, `n_layers=8→12`, `d_ff=1024`
  - Sinusoidal time embedding + öğrenilebilir pozisyon embedding
  - Pre-LayerNorm transformer blokları
- **Loss:** MSE on predicted noise + (v2'de) OHLC constraint loss
- **Sampling:** DDPM reverse process, ayrıca **classifier-free guidance** denenmiş (v2).

**Kritik kod parçası (loss formu):**
```python
# Forward: x_0'a noise ekle
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
# Reverse: modelden noise tahmini al, x_0 geri çıkar
pred_x0 = (x_t - sqrt(1 - alpha_bar_t) * pred_noise) / sqrt(alpha_bar_t)
```

**v1 → v2 arasındaki fark:**
| Konu | v1 | v2 |
|------|----|----|
| Model boyutu | 256 / 8 layer | 384 / 12 layer |
| Volatilite | "Çok güvenli" (düşük) | Volatilite artırıcı loss |
| OHLC kısıtı | Yok | `L ≤ O,C ≤ H` constraint loss |
| Guidance | Yok | Classifier-free guidance |
| Epoch | ~100 | 200+ |

**Güçlü yanlar:**
- Olasılıksal; aynı bağlamdan **yüzlerce farklı senaryo** örneklenebilir.
- Uzun bağımlılıkları self-attention ile yakalar (GRU bazlı TimeGAN'den teorik olarak üstün).
- Modern araştırma trendine uyumlu (Kronos, ForecastPFN, Time-LLM).

**Zayıf yanlar:**
- Eğitim maliyeti **çok** yüksek (A100'de saatler).
- 1000-step DDPM sampling → gecikme büyük; demo için pratik değil.
- Hiperparametre hassasiyeti yüksek; deneyimsiz ayarlarda mode collapse / patlama.

---

## 7. Aile C: TimeGAN Denemeleri

**TimeGAN** (Yoon, Jarrett, van der Schaar, 2019) beş bileşenden oluşur:

1. **Embedder** – gerçek veriyi latent uzaya projekte eder
2. **Recovery** – latent'ten veriyi geri getirir (autoencoder çifti ile)
3. **Generator** – gürültüden sentetik latent üretir
4. **Supervisor** – latent uzayda zaman tutarlılığı sağlar
5. **Discriminator** – gerçek vs sentetik latent ayrımı yapar

Eğitim 3 fazlıdır: (1) Autoencoder, (2) Supervisor, (3) Joint GAN.

Aşağıdaki 7 defter bu çekirdek fikrin farklı varyasyonlarıdır.

### 7.1 `20_timegan_strategy6_eda.ipynb` — TimeGAN İçin Veri Hazırlık

**Rol:** TimeGAN **eğitim** defteri değil; **veri hazırlık** defteridir.

İçerik:
- Zaman farkı analizi → hafta sonu gap'leri, broker tatilleri saptanıyor.
- ADF (Augmented Dickey-Fuller) **durağanlık testi** → log getirilerinin `p < 0.05` ile durağan olduğu gösteriliyor (TimeGAN için ideal).
- Log getirilerinin **yüzde birlik ve çeyrekliklerle simetrik ölçekleme** yapılarak `[-1, 1]`'e getirilmesi.
- Korelasyon analizi, dağılım grafikleri (Q-Q plot, hist).
- Çıktı: `timegan_ready_strategy_6.csv` + `scaling_params_strategy_6.json`.

**Önemi:** TimeGAN'in mode-collapse ve NaN üretme riskini azaltan en kritik adım veridir; bu defter o yüzden var.

### 7.2 `30_timegan_baseline_4feat.ipynb` — TimeGAN Baseline

**En sade uygulama:**
- 4 özellik: `log_ret_open/high/low/close_norm`
- `seq_len=24`, `hidden_dim=24`, `num_layers=3`
- 3 fazlı klasik Yoon 2019 uygulaması
- PCA + t-SNE ile gerçek/sentetik karşılaştırması

### 7.3 `31_timegan_full_training_a100.ipynb` — Tam Eğitim Pipeline

**Ölçek büyütülmüş versiyon:**
- `hidden_dim=128`, `z_dim=64`, `batch=128`
- **Mixed precision (`float16`)** training – A100'de ~2× hızlanma
- Her ağ için ayrı Adam optimizer
- Save/load artifact'lar, daha detaylı plotlar

### 7.4 `32_timegan_forex_compact.ipynb` — Kompakt Varyant

`30` ve `31`'in sadeleştirilmiş orta kademe sürümü. Deney için hızlı iterasyon amaçlı.

### 7.5 `33_timegan_forex_11feat.ipynb` — 11 Feature TimeGAN

**En zengin feature seti:**
```python
FEATURES = [
  'logret', 'hl_range', 'body', 'atr', 'realized_vol',
  'rsi', 'volume_ratio', 'sin_hour', 'cos_hour',
  'sin_dow', 'cos_dow'
]
```
- Teknik göstergeler + takvim dönüşümleri dahil.
- `SEQ_LEN=24, HIDDEN_DIM=128, NUM_LAYERS=3, EPOCHS=500`.
- MinMaxScaler [0,1].
- G kaybında **moment matching** (std ve mean farkı) ekli.

### 7.6 `34_timegan_forex_run_outputs.ipynb` — Tam Çalıştırılmış Sürüm

`33`'ün çıktılarının (grafikler, metrikler) notebook'a gömülü hâli. Boyutu bu yüzden büyüktür (884 KB). Kod mantığı aynı.

### 7.7 `35_timegan_improved_wgangp.ipynb` — En Gelişmiş Varyant

Bu defter **ayrı değerlendirmeyi hak ediyor.** Şu sorunları hedefler:

- **"Too-smooth" sentetik seri** → Volatilite loss'u
- **Fat tail (finans aşk dağılımı) eksikliği** → Tail quantile matching
- **Discriminator collapse** → WGAN-GP gradient penalty
- **Eğitim kararsızlığı** → TTUR (Two Time-Scale Update Rule): `LR_D > LR_G`

**Ek loss bileşenleri:**
```python
g_total = g_adv
        + 100 * sqrt(g_sup)                # Yoon 2019 orijinal
        + VOL_LAMBDA * |E|r_real| - E|r_fake||
        + VOL_LAMBDA * |E[r_real^2] - E[r_fake^2]|
        + TAIL_LAMBDA * (|q01_real - q01_fake| + |q99_real - q99_fake|)

d_total = BCE(real, fake) + GP_LAMBDA * gradient_penalty
```

- **Scaling:** Robust (median/IQR), clip `[-10, 10]`.
- **Early stopping:** GAN loss yerine *kalite metrikleri* üzerinden (fevkalade doğru seçim).
- **Hard OHLC post-processing.**

Literatür etiketi: **SigWGAN / QuantGAN** ailesine en yakın uygulama.

### 7.8 `36_timegan_pro_a100.ipynb` — Pro Sürüm

- **LayerNorm** GRU çıkışında (stabilizasyon).
- **Exponential LR decay** (`decay_rate=0.96 / 1000 steps`).
- Moment matching loss.
- mplfinance ile gerçek candlestick görselleştirme.

---

## 8. Karşılaştırmalı Tablo

| # | Defter | Mimari | Tip | Deterministik? | Eğitim Maliyeti | OHLC Kısıtı | Uygulama Alanı |
|---|--------|--------|-----|----------------|------------------|-------------|-----------------|
| 01 | chronos_basic | T5 (pretrained) | Tahmin | Hayır (sample) | Yok (zero-shot) | Heuristik | Kısa ufuk |
| 02 | chronos_ohlc_synthetic | T5 (pretrained) | Sentetik üretim | Hayır | Yok | Soft enforcement | Uzun senaryo |
| 10 | diffusion_v1 | Transformer + DDPM | Üretken | Hayır (çok sample) | ÇOK yüksek | Yok | Araştırma |
| 11 | diffusion_v2 | Transformer + DDPM | Üretken | Hayır | ÇOK yüksek | Loss ile enforce | Araştırma |
| 20 | timegan_strategy6_eda | — | Veri hazırlık | — | Yok | — | Veri pipeline |
| 30 | timegan_baseline_4feat | TimeGAN (Yoon) | Sentetik | Hayır | Orta | Yok | Baseline |
| 31 | timegan_full_training | TimeGAN + mixed FP | Sentetik | Hayır | Yüksek | Yok | Üretim hazır |
| 32 | timegan_forex_compact | TimeGAN | Sentetik | Hayır | Düşük-orta | Yok | Hızlı deney |
| 33 | timegan_forex_11feat | TimeGAN + TA | Sentetik | Hayır | Yüksek | Yok | Feature-rich |
| 34 | timegan_forex_run_outputs | = 33 | Sentetik | — | — | — | Rapor için run |
| 35 | timegan_improved_wgangp | TimeGAN + WGAN-GP + vol/tail | Sentetik | Hayır | Çok yüksek | Post-process | En iyi TimeGAN |
| 36 | timegan_pro_a100 | TimeGAN + LayerNorm | Sentetik | Hayır | Yüksek | Post-process | A100 stabil |

---

## 9. Dönem Projesine Çıkarılan Dersler

Bu denemeler, dönem projesinin (tahmin + walk-forward + Streamlit demo) **doğru mimari seçimlerinin** gerekçesini oluşturuyor.

### 9.1 Seçilen Yol

Final projede **doğrudan tahmin (forecasting)** hedefleniyor; sentetik üretim değil. Bu nedenle TimeGAN/Diffusion ailelerini **bağımsız modül** olarak dışarıda tutuyorum. Bunun yerine:

1. **Baseline:** Naive / Drift (proje planındaki `src/models/baseline.py`)
2. **Klasik ML:** XGBoost / LightGBM / RandomForest
3. **DL:** LSTM veya 1D-CNN
4. **Bonus / senaryo paneli:** Buradaki `02_chronos_ohlc_synthetic.ipynb`'nin sadeleştirilmiş bir port'u — Streamlit demo'nun "senaryo üretimi" sekmesinde kullanılacak.

### 9.2 Neden TimeGAN Final'de Ana Model Değil?

- **Tahmin ≠ Sentetik üretim:** TimeGAN, "gerçeğe benzer yeni seriler" üretir, "şu andan sonrası ne olur" sorusunu yanıtlamaz. Dersin değerlendirme metrikleri (MAE/RMSE/Directional Accuracy) TimeGAN için doğal tanımlı değil.
- **Demo sınırı:** GAN çıktıları stokastik ve subjektif; 12 dakikalık sunumda net metrik göstermek daha etkili.
- **Augmentation olarak kullanma ihtimali:** Az veriden muzdarip olmadığımız için (100k satır) sentetik veriye **ihtiyaç yok**.

### 9.3 Neden Chronos Yardımcı Modül Olarak Kalıyor?

- Zero-shot, eğitim maliyeti sıfır.
- Olasılıksal çoklu yol üretebildiği için **senaryo paneli** için ideal.
- Streamlit demo'da "Model XGBoost dedi ki X, ancak Chronos 100 yol örnekledi; bant şöyle" gibi zenginleştirici bir içerik sunabilir.

### 9.4 Diffusion Deneyinden Alınan Ders

Büyük model → daha iyi demek değil. Özellikle:

- Walk-forward test'te en basit naive baseline bile çoğu zaman tabanını geçmek gerektirir.
- Diffusion için eğitim süresi ve belirsizlik, bir ödev kapsamında çok yüksek.
- Bu defterler **"denedim, gerekçeyle final mimarisinde yok"** olarak rapora girecek.

---

## 10. Rapora Giriş Önerisi (Kopyalanabilir Paragraf)

> Dönem projesinin nihai mimarisini belirlemeden önce EUR/USD 1H serisi üzerinde üç farklı yapay zekâ paradigması pilot düzeyde denenmiştir: (i) önceden eğitilmiş Amazon Chronos (T5) zaman serisi foundation modeli ile zero-shot tahmin, (ii) DDPM + Transformer omurgalı üretken modelleme (Kronos-tipi) ve (iii) TimeGAN ailesi altında WGAN-GP, gradient penalty, volatilite/kuyruk kayıpları gibi iyileştirmelerle toplam 7 varyant. Yapılan 13 Colab defterinin tamamı `docs/prior_work.md` altında belgelenmiştir. Bu denemeler, projenin sınıflandırıcı/regresör tabanlı tahmin yönünü **tahmin tarafında klasik ML + derin öğrenme**, **senaryo paneli tarafında ise hafif bir Chronos bağlaması** olarak şekillendirmiştir. Üretken yaklaşımlar (TimeGAN, Diffusion) veri kıtlığı olmaması ve dersin `MAE/RMSE/Directional Accuracy` gibi noktasal metrikleriyle doğal uyum sağlamaması nedeniyle final mimarinin **ana hattı dışında** bırakılmıştır.

---

## 11. Değişiklik Günlüğü

| Tarih | Değişiklik |
|-------|------------|
| 2026-04-20 | 13 Colab defteri incelenip açıklayıcı isimlerle yeniden adlandırıldı, 4 boş stub silindi. |
| 2026-04-20 | Bu `MODELS_OVERVIEW.md` hazırlandı; ana projenin `docs/prior_work.md`'sine de kopyalandı. |
