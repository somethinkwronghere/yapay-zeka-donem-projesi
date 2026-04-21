# Prior Experiments — Colab Model Denemeleri

Bu klasör, dönem projesinin **resmî kapsamı dışındaki** 13 Colab defterini barındırır. Hepsi EUR/USD 1H serisi üzerinde yapılan erken denemelerdir ve final projenin "Önceki Çalışmalar" bölümünün delil dosyalarıdır.

> Bu defterleri birebir çalıştırmanız gerekmez. Hepsinin toplu analizi, karşılaştırması ve her birinin neden final mimaride yer almadığının gerekçesi için [`docs/prior_work.md`](../../../docs/prior_work.md) dosyasına bakınız.

## Aileler

| Önek | Aile | Açıklama |
|------|------|----------|
| `01-09` | **Chronos** | Amazon Chronos T5 foundation modeli ile zero-shot tahmin |
| `10-19` | **Diffusion + Transformer** | Kronos-style DDPM + self-attention üretken modelleme |
| `20-29` | **TimeGAN Veri Hazırlık** | TimeGAN için EDA ve ölçekleme |
| `30-39` | **TimeGAN Eğitim Varyantları** | Baseline → WGAN-GP iyileştirmelerine 7 varyant |

## Defter Listesi

| Defter | Aile | Rol |
|--------|------|-----|
| `01_chronos_basic.ipynb` | A | Chronos zero-shot minimal uygulama |
| `02_chronos_ohlc_synthetic.ipynb` | A | OHLC dressing + chunked generation (5000 mum) |
| `03_chronos_draft_notes.ipynb` | A | Chronos taslak notlar |
| `10_diffusion_transformer_v1.ipynb` | B | Kronos-style DDPM, d_model=256 |
| `11_diffusion_transformer_v2.ipynb` | B | İyileştirilmiş: d_model=384, OHLC constraint loss |
| `20_timegan_strategy6_eda.ipynb` | C | ADF test + [-1,1] simetrik ölçekleme |
| `30_timegan_baseline_4feat.ipynb` | C | Klasik TimeGAN (Yoon 2019) |
| `31_timegan_full_training_a100.ipynb` | C | Mixed-precision tam pipeline |
| `32_timegan_forex_compact.ipynb` | C | Hızlı deney için kompakt sürüm |
| `33_timegan_forex_11feat.ipynb` | C | 11 feature (RSI, ATR, takvim) |
| `34_timegan_forex_run_outputs.ipynb` | C | `33`'ün gömülü grafiklerle çalıştırılmış hâli |
| `35_timegan_improved_wgangp.ipynb` | C | En gelişmiş: WGAN-GP + TTUR + vol/tail loss |
| `36_timegan_pro_a100.ipynb` | C | LayerNorm + moment matching |

## Yan Veri

`20_timegan_strategy6_eda.ipynb`'nin ürettiği `train_sequences.csv` (~162 MB) GitHub'ın 100 MB per-file limitini aştığı için repo'ya commit edilmemiştir. Notebook yeniden çalıştırılınca Colab'da otomatik oluşur. Gerekirse Git LFS ile eklenebilir, ancak proje bağımsız olarak çalışabildiği için gerekli değil.

## Commit Kararı

Bu defterler repo'ya sadece **tarihsel kayıt** ve **rapor delili** amacıyla eklenmiştir. Final projedeki kodlama akışı tamamen `src/` + `notebooks/01_eda.ipynb` vb. üzerinden ilerler.
