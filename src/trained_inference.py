"""Egitilmis PyTorch checkpoint'leri ile inference yardimcilari."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.features import build_feature_frame
from src.models.deep import LSTMForecaster, TransformerForecaster

TRAINED_DEEP_MODELS = {"lstm", "transformer"}
RAW_COLUMNS = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
PIP = 0.0001


def _raw_view(df: pd.DataFrame) -> pd.DataFrame:
    return df[RAW_COLUMNS].copy()


@lru_cache(maxsize=4)
def load_trained_model(model_name: str, checkpoint_dir: str):
    """Checkpoint'i yukler ve modeli eval modunda dondurur."""
    if model_name not in TRAINED_DEEP_MODELS:
        raise ValueError(f"Desteklenmeyen egitilmis model: {model_name}")

    ckpt_path = Path(checkpoint_dir) / f"{model_name}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = dict(ckpt["config"])
    if model_name == "lstm":
        model = LSTMForecaster(**config)
    else:
        model = TransformerForecaster(**config)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def _scaled_feature_matrix(feature_frame: pd.DataFrame, ckpt: dict) -> np.ndarray:
    cols = ckpt["feature_cols"]
    mu = pd.Series(ckpt["feature_mu"])[cols]
    sd = pd.Series(ckpt["feature_sd"])[cols].replace(0, 1e-9)
    return ((feature_frame[cols] - mu) / sd).values.astype(np.float32)


def predict_next_return(raw_df: pd.DataFrame, model_name: str, checkpoint_dir: Path) -> float:
    """Verilen son mumdan bir sonraki saat icin log-getiri tahmini uretir."""
    model, ckpt = load_trained_model(model_name, str(checkpoint_dir))
    seq_len = int(ckpt["seq_len"])
    features = build_feature_frame(_raw_view(raw_df)).dropna()
    if len(features) < seq_len:
        raise ValueError(f"Model icin en az {seq_len} temiz feature satiri gerekir.")

    x = _scaled_feature_matrix(features, ckpt)[-seq_len:][None, :, :]
    with torch.no_grad():
        return float(model(torch.from_numpy(x)).cpu().numpy()[0])


def forecast_trained_path(raw_df: pd.DataFrame, model_name: str, horizon: int, checkpoint_dir: Path) -> pd.Series:
    """Tek-adim checkpoint tahminini cok-adimli senaryo yoluna cevirir.

    Checkpoint t+1 log-getiri icin egitildi. Bu nedenle ilk nokta dogrudan
    model ciktisidir; daha uzun ufuklar ise bu sinyalin sonumlu uzantisidir.
    Gercek performans yorumu icin `rolling_backtest_trained` kullanilir.
    """
    work = _raw_view(raw_df).copy()
    first_ret = predict_next_return(work, model_name, checkpoint_dir)
    recent_ret_vol = float(np.log(work["close"]).diff().tail(168).std())
    if not np.isfinite(recent_ret_vol) or recent_ret_vol <= 0:
        recent_ret_vol = 0.0006
    first_ret = float(np.clip(first_ret, -2.5 * recent_ret_vol, 2.5 * recent_ret_vol))

    steps = np.arange(horizon, dtype=np.float64)
    pred_rets = first_ret * np.exp(-steps / 12.0)
    last_close = float(work["close"].iloc[-1])
    preds = last_close * np.exp(np.cumsum(pred_rets))
    idx = pd.date_range(work.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    return pd.Series(preds, index=idx, name="forecast")


def rolling_backtest_trained(raw_df: pd.DataFrame, model_name: str, end_ts: pd.Timestamp, hours: int, checkpoint_dir: Path) -> pd.DataFrame:
    """Son saatler icin egitilmis checkpoint ile hizli tek-adim rolling test."""
    model, ckpt = load_trained_model(model_name, str(checkpoint_dir))
    seq_len = int(ckpt["seq_len"])
    features = build_feature_frame(_raw_view(raw_df.loc[:end_ts])).dropna()
    supervised = features.copy()
    supervised["target"] = supervised["log_ret_1"].shift(-1)
    supervised["actual_close_next"] = supervised["close"].shift(-1)
    supervised = supervised.dropna()
    if len(supervised) < seq_len:
        raise ValueError(f"Rolling test icin en az {seq_len} satir gerekir.")

    x_all = _scaled_feature_matrix(supervised, ckpt)
    windows = np.stack([x_all[i - seq_len + 1 : i + 1] for i in range(seq_len - 1, len(supervised))])
    rows = supervised.iloc[seq_len - 1 :].copy()
    preds = []
    batch = 1024
    with torch.no_grad():
        for start in range(0, len(windows), batch):
            xb = torch.from_numpy(windows[start : start + batch])
            preds.append(model(xb).cpu().numpy())
    pred_ret = np.concatenate(preds)
    pred_close = rows["close"].values * np.exp(pred_ret)

    out = pd.DataFrame(
        {
            "actual": rows["actual_close_next"].values,
            "predicted": pred_close,
            "error_pip": (pred_close - rows["actual_close_next"].values) / PIP,
            "dir_hit": np.sign(pred_close - rows["close"].values) == np.sign(rows["actual_close_next"].values - rows["close"].values),
        },
        index=rows.index + pd.Timedelta(hours=1),
    )
    return out.tail(hours)
