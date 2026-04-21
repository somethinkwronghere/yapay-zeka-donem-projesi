"""Baseline tahmin modelleri (Hafta 4 teslimi için iskelet)."""
from __future__ import annotations

import numpy as np
import pandas as pd


class NaiveLastValue:
    """Basit baseline: bir sonraki H saat için son kapanışı tekrarla.

    Kulağa aptalca gelse de forex gibi rassal yürüyüş benzeri serilerde kırılması
    şaşırtıcı derecede zor bir baseline'dır ve literatürde referans olarak kullanılır.
    """

    def __init__(self, horizon: int = 1) -> None:
        self.horizon = horizon
        self._last_value: float | None = None

    def fit(self, y: pd.Series) -> "NaiveLastValue":
        self._last_value = float(y.iloc[-1])
        return self

    def predict(self, n_steps: int | None = None) -> np.ndarray:
        if self._last_value is None:
            raise RuntimeError("Önce fit() çağırılmalı.")
        n = n_steps or self.horizon
        return np.full(n, self._last_value, dtype=np.float64)


class DriftModel:
    """Son gözleme ek olarak, eğitim serisindeki ortalama log-getiriyi doğrusal olarak ekler."""

    def __init__(self, horizon: int = 1) -> None:
        self.horizon = horizon
        self._last_value: float | None = None
        self._drift: float | None = None

    def fit(self, y: pd.Series) -> "DriftModel":
        y = y.astype(float)
        log_ret = np.log(y).diff().dropna()
        self._drift = float(log_ret.mean())
        self._last_value = float(y.iloc[-1])
        return self

    def predict(self, n_steps: int | None = None) -> np.ndarray:
        if self._last_value is None or self._drift is None:
            raise RuntimeError("Önce fit() çağırılmalı.")
        n = n_steps or self.horizon
        steps = np.arange(1, n + 1)
        return self._last_value * np.exp(self._drift * steps)
