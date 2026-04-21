"""Öznitelik mühendisliği yardımcıları.

Hepsi `shift(1)` disiplinine uyar: modele verilen öznitelikler yalnızca `t` zamanındaki
bilgiyle bilinebilir; hedef `t+1..t+H` ise kaçak (leakage) olmaz.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_returns(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    out = df.copy()
    out["ret_1"] = out[price_col].pct_change()
    out["log_ret_1"] = np.log(out[price_col]).diff()
    return out


def add_rolling_stats(
    df: pd.DataFrame,
    price_col: str = "close",
    windows: tuple[int, ...] = (5, 20, 60, 168),
) -> pd.DataFrame:
    """5H, 20H, 60H, 168H (1 hafta) rolling mean ve std."""
    out = df.copy()
    for w in windows:
        out[f"roll_mean_{w}"] = out[price_col].rolling(w).mean()
        out[f"roll_std_{w}"] = out[price_col].rolling(w).std()
    return out


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["month"] = idx.month
    out["is_asia_session"] = ((idx.hour >= 0) & (idx.hour < 8)).astype("int8")
    out["is_eu_session"] = ((idx.hour >= 7) & (idx.hour < 16)).astype("int8")
    out["is_us_session"] = ((idx.hour >= 13) & (idx.hour < 21)).astype("int8")
    return out


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Tüm öznitelikleri tek çağrıda üretir (ileride TA-Lib / pandas-ta eklenecek)."""
    out = (
        df.pipe(add_returns)
        .pipe(add_rolling_stats)
        .pipe(add_calendar)
    )
    return out
