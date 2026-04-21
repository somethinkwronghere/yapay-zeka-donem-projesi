# Data Klasörü

Bu klasör projenin tüm verilerini barındırır.

## Alt Klasörler

| Klasör | Amaç |
|--------|------|
| `raw/` | Hiç dokunulmamış ham veri. MT5/Kaggle export'ları, orijinal CSV dosyaları. **Salt okunur** olarak kabul edilir. |
| `processed/` | EDA + feature engineering sonrası üretilen temizlenmiş / öznitelikli tablolar (parquet/feather). Git tarafından **ignore** edilir; notebook çalıştırılınca yeniden üretilir. |
| `external/` | Dış kaynaklardan (ör. FRED, haber API'leri, ekonomik takvim) çekilen yardımcı veriler. Git ignore. |

## Ana Veri Seti

- **Dosya:** `raw/eurusd_h1.csv`
- **Sembol:** EURUSD
- **Zaman Dilimi:** 1 saat (H1)
- **Kaynak:** Kaggle – *"EURUSD - 1H - 2020-2024 September FOREX"* + MetaTrader 5 export ile genişletilmiş.
- **Dönem:** 2009-10-16 00:00 UTC → (MT5 brokerinden son çekilen saate kadar)
- **Satır sayısı:** ~100.000 mum

### Veri Sözlüğü

| Sütun | Tip | Açıklama |
|-------|-----|----------|
| `time` | datetime64[ns] | Mumun başlangıç zamanı (UTC, broker saatine göre). |
| `open` | float64 | Mumun açılış fiyatı (EUR/USD). |
| `high` | float64 | Mum içindeki en yüksek fiyat. |
| `low` | float64 | Mum içindeki en düşük fiyat. |
| `close` | float64 | Mumun kapanış fiyatı. **Temel hedef değişken.** |
| `tick_volume` | int64 | Mum süresince oluşan fiyat tik sayısı (forex'te gerçek hacim yerine kullanılır). |
| `spread` | int64 | Alış/satış farkı (puan cinsinden). |
| `real_volume` | int64 | Broker tarafından raporlanan hacim. Forex majors için genelde 0. |

## Lisans ve Kullanım

- Kaggle veri setinin lisansı ve kullanım koşulları geçerlidir.
- MetaTrader 5 üzerinden indirilen veriler, broker'ın kullanım koşullarına tabidir.
- Bu proje yalnızca akademik amaçlıdır; ticari kullanım için ek lisans gerekebilir.

## Yeniden Üretilebilirlik

```python
from src.data_loader import load_raw_eurusd
df = load_raw_eurusd("data/raw/eurusd_h1.csv")
```
