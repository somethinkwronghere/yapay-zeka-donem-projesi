"""Ham EUR/USD verisini yükleme ve temel sağlık kontrolleri."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]

EXPECTED_COLUMNS = [
    "time",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
    "real_volume",
]


def load_raw_eurusd(csv_path: PathLike) -> pd.DataFrame:
    """MT5/Kaggle kaynaklı ham EUR/USD 1H CSV'sini yükler.

    - `time` sütununu parse edip DatetimeIndex yapar.
    - Kronolojik sıraya getirir.
    - Tekrarlayan zaman damgalarını tek satıra indirger (son kayıt kalır).
    - Temel tip dönüşümlerini (float) uygular.

    Parameters
    ----------
    csv_path:
        `data/raw/eurusd_h1.csv` dosyasının yolu.

    Returns
    -------
    pandas.DataFrame
        `time` index; `open, high, low, close, tick_volume, spread, real_volume` sütunlu DataFrame.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Ham veri bulunamadı: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV beklenen şemayla uyumsuz. Eksik sütun(lar): {sorted(missing)}"
        )

    df["time"] = pd.to_datetime(df["time"], utc=False, errors="raise")
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype("float64")
    for col in ("tick_volume", "spread", "real_volume"):
        df[col] = df[col].astype("int64")

    return df


def train_val_test_split(
    df: pd.DataFrame,
    train_end: str = "2022-12-31",
    val_end: str = "2023-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Zaman bazlı, karışmasız üçlü ayırma.

    Varsayılan kesim noktaları proje planıyla uyumludur.
    """
    train = df.loc[:train_end]
    val = df.loc[train_end:val_end].iloc[1:]
    test = df.loc[val_end:].iloc[1:]
    return train, val, test
