# Notebooks

Projeye ait Jupyter defterleri. Sıra numarası ile proje yaşam döngüsünü takip eder.

| Defter | Hafta | İçerik |
|--------|-------|--------|
| `01_eda.ipynb` | 3 | Keşifsel veri analizi, veri kalitesi, görselleştirmeler |
| `02_baselines.ipynb` | 4 | Naive, Drift, Seasonal-24H/168H, MA(24), AR(p) — RMSE/dir_acc tablosu |
| `03_ml_comparison.ipynb` | 5 | RF / XGBoost / LightGBM + 25 feature + TimeSeriesCV + feature importance |
| `04_deep_learning.ipynb` | 6 | LSTM (64 hidden) + küçük Transformer encoder; eğitim eğrileri + checkpoint kaydı |
| `05_evaluation.ipynb` | 7 | Walk-forward test ve tüm metrikler (eklenecek) |
| `archive/` | - | Proje kapsamı dışında kalan erken denemeler |

## Çalıştırma

```powershell
jupyter lab
```

Her defter, proje kökünü `sys.path`'e ekleyerek `src/` altındaki modülleri kullanır. Bu sayede kod tekrarı olmaz.
