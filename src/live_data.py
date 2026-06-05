"""Canli/son EUR/USD verisi cekme yardimcilari."""
from __future__ import annotations

import pandas as pd


def fetch_live_eurusd(period: str = "60d", interval: str = "1h") -> pd.DataFrame:
    """Yahoo Finance uzerinden EUR/USD saatlik OHLC verisini getirir.

    Streamlit demosunda internet yoksa cagiran taraf hatayi yakalayip
    kullaniciya bilgi verir.
    """
    import yfinance as yf

    periods = [period, "30d", "5d", "1mo"]
    raw = pd.DataFrame()
    last_error: Exception | None = None
    for candidate in dict.fromkeys(periods):
        try:
            raw = yf.download(
                "EURUSD=X",
                period=candidate,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,
                timeout=20,
            )
            if not raw.empty:
                break
        except Exception as exc:
            last_error = exc
    if raw.empty:
        if last_error is not None:
            raise RuntimeError(f"Yahoo Finance EURUSD=X verisi alinamadi: {last_error}")
        raise RuntimeError("Yahoo Finance EURUSD=X icin bos veri dondu.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    out = raw.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "tick_volume",
        }
    )
    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise RuntimeError(f"Canli veri eksik kolon dondu: {missing}")

    out = out[["open", "high", "low", "close", "tick_volume"]].copy()
    out["spread"] = 0
    out["real_volume"] = 0
    out["tick_volume"] = out["tick_volume"].fillna(0).astype("int64")
    out["spread"] = out["spread"].astype("int64")
    out["real_volume"] = out["real_volume"].astype("int64")
    out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_convert("Europe/Istanbul").tz_localize(None)
    out = out.dropna().sort_index()
    return out
