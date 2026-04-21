# Notebooks

Projeye ait Jupyter defterleri. Sıra numarası ile proje yaşam döngüsünü takip eder.

| Defter | Hafta | İçerik |
|--------|-------|--------|
| `01_eda.ipynb` | 3 | Keşifsel veri analizi, veri kalitesi, görselleştirmeler |
| `02_baseline.ipynb` | 4 | Naive + lineer baseline modeller (eklenecek) |
| `03_ml_comparison.ipynb` | 5 | RF / XGBoost / LightGBM karşılaştırması (eklenecek) |
| `04_deep_learning.ipynb` | 6 | LSTM / GRU / 1D-CNN (eklenecek) |
| `05_evaluation.ipynb` | 7 | Walk-forward test ve tüm metrikler (eklenecek) |
| `archive/` | - | Proje kapsamı dışında kalan erken denemeler |

## Çalıştırma

```powershell
jupyter lab
```

Her defter, proje kökünü `sys.path`'e ekleyerek `src/` altındaki modülleri kullanır. Bu sayede kod tekrarı olmaz.
