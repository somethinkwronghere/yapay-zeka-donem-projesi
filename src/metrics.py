"""Zaman serisi tahmini için ortak metrikler.

Tüm fonksiyonlar numpy array / pandas Series kabul eder ve skaler döndürür.
Modelleme defterleri ve ileri testler arasında tutarlılığı sağlamak için
burada tek noktada tanımlanmıştır.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ArrayLike = np.ndarray | pd.Series


def _as_arr(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true, y_pred = _as_arr(y_true), _as_arr(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true, y_pred = _as_arr(y_true), _as_arr(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-12) -> float:
    y_true, y_pred = _as_arr(y_true), _as_arr(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)


def direction_accuracy(y_true_ret: ArrayLike, y_pred_ret: ArrayLike) -> float:
    """Getirinin yönünü (yukarı/aşağı) doğru bilme oranı.

    Birebir 0'a eşit getiriler ayıklanır; kalan gözlemlerin yüzdesi döner.
    """
    y_true, y_pred = _as_arr(y_true_ret), _as_arr(y_pred_ret)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])) * 100.0)


def pip_rmse(y_true_close: ArrayLike, y_pred_close: ArrayLike) -> float:
    """EUR/USD için RMSE'yi pip cinsinden (1 pip = 0.0001) verir."""
    return rmse(y_true_close, y_pred_close) * 1e4


def score_all(
    y_true_ret: ArrayLike,
    y_pred_ret: ArrayLike,
    y_true_close: ArrayLike,
    y_pred_close: ArrayLike,
) -> dict[str, float]:
    """Tüm metrikleri tek bir sözlük olarak döndürür."""
    return {
        "rmse_ret_bp": rmse(y_true_ret, y_pred_ret) * 1e4,
        "mae_ret_bp": mae(y_true_ret, y_pred_ret) * 1e4,
        "rmse_close_pip": pip_rmse(y_true_close, y_pred_close),
        "mae_close_pip": mae(y_true_close, y_pred_close) * 1e4,
        "mape_close": mape(y_true_close, y_pred_close),
        "dir_acc_pct": direction_accuracy(y_true_ret, y_pred_ret),
    }
