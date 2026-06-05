# API / Demo

Bu klasör, final sunumunda gösterilecek **Streamlit** arayüzünü barındırır. Demo; EUR/USD verisini, model skorlarını, tahmin/senaryo panelini, rolling forward-test grafiğini ve EDA görsellerini tek ekranda toplar.

## Çalıştırma

```powershell
# Proje kökünde, venv aktif:
streamlit run api/app.py
```

Varsayılan adres: `http://localhost:8501`

## Ekranlar

| Sekme | İçerik |
|-------|--------|
| Canlı karşılaştırma | Yahoo EURUSD=X saatlik mumları, checkpoint tahminleri, diğer model yolları ve canlı rolling karşılaştırma |
| Tahmin paneli | Geçmiş kapanış, seçilen model tahmini, senaryo bandı ve CSV indirme |
| Forward test | Rolling RMSE, MAE, yön isabeti ve gerçek/tahmin grafiği |
| Model karşılaştırma | Test skorlarının aile bazlı dağılımı ve detaylı skor tablosu |
| Veri ve EDA | Ham veri özeti, son veri kesiti ve rapor görselleri |

## Not

Demo hızlı ve stabil sunum için yerel veri ve kayıtlı skor tablolarıyla çalışır. `LSTM` veya `Transformer` seçildiğinde `data/processed/checkpoints/*.pt` dosyaları yüklenir; ilk tahmin adımı gerçek PyTorch checkpoint inference ile üretilir. Model tek-adım (`t+1`) eğitildiği için daha uzun ufuklar bu sinyalin sönümlü senaryo yolu olarak gösterilir.

Bu uygulama **akademik** bir projenin parçasıdır. Ürettiği tahminler **yatırım tavsiyesi değildir** ve gerçek işlem kararları için kullanılamaz.
