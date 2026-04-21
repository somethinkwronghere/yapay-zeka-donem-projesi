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


def add_lagged_returns(
    df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 3, 6, 12, 24, 168),
    col: str = "log_ret_1",
) -> pd.DataFrame:
    """`col` sütununun lag'lerini `col_lag_k` adıyla ekler (shift ile geriye)."""
    out = df.copy()
    for k in lags:
        out[f"{col}_lag_{k}"] = out[col].shift(k)
    return out


def add_volatility(
    df: pd.DataFrame,
    col: str = "log_ret_1",
    windows: tuple[int, ...] = (5, 20, 60, 168),
) -> pd.DataFrame:
    """Log-getirinin rolling standart sapması (volatilite)."""
    out = df.copy()
    for w in windows:
        out[f"vol_{w}"] = out[col].rolling(w).std()
    return out


def add_rsi(df: pd.DataFrame, price_col: str = "close", period: int = 14) -> pd.DataFrame:
    """Klasik Wilder RSI (0-100)."""
    out = df.copy()
    delta = out[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    return out


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range — OHLC tabanlı volatilite göstergesi."""
    out = df.copy()
    high, low, close = out["high"], out["low"], out["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    out[f"atr_{period}"] = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return out


def add_price_range(df: pd.DataFrame) -> pd.DataFrame:
    """Mum içi yüksek-düşük aralığı (close'a oranla)."""
    out = df.copy()
    out["hl_range"] = (out["high"] - out["low"]) / out["close"]
    out["oc_change"] = (out["close"] - out["open"]) / out["open"]
    return out


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Tüm öznitelikleri tek çağrıda üretir; ML/DL defterleri bu fonksiyonu çağırır."""
    out = (
        df.pipe(add_returns)
        .pipe(add_rolling_stats)
        .pipe(add_calendar)
        .pipe(add_lagged_returns)
        .pipe(add_volatility)
        .pipe(add_rsi)
        .pipe(add_atr)
        .pipe(add_price_range)
    )
    return out
