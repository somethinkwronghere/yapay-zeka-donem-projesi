"""EUR/USD AI Forecast Streamlit demo.

Bu arayüz, final sunumunda veri, model metrikleri, ileriye dönük tahmin
senaryoları ve kısa forward-test akışını tek ekranda göstermek için hazırlandı.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_raw_eurusd
from src.live_data import fetch_live_eurusd
from src.trained_inference import TRAINED_DEEP_MODELS, forecast_trained_path, rolling_backtest_trained

RAW_CSV = PROJECT_ROOT / "data" / "raw" / "eurusd_h1.csv"
SCORES_CSV = PROJECT_ROOT / "data" / "processed" / "all_model_scores.csv"
IMAGES_DIR = PROJECT_ROOT / "docs" / "images"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "processed" / "checkpoints"

PIP = 0.0001
MODEL_LABELS = {
    "naive_last_value": "Naive son deger",
    "drift": "Drift",
    "seasonal_naive_24h": "Seasonal naive 24H",
    "seasonal_naive_168h": "Seasonal naive 168H",
    "ma_24h": "MA 24H",
    "ar(1)": "AR(1)",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "lstm": "LSTM",
    "transformer": "Transformer",
}
MODEL_FAMILIES = {
    "naive_last_value": "Baseline",
    "drift": "Baseline",
    "seasonal_naive_24h": "Baseline",
    "seasonal_naive_168h": "Baseline",
    "ma_24h": "Baseline",
    "ar(1)": "Baseline",
    "random_forest": "Klasik ML",
    "xgboost": "Klasik ML",
    "lightgbm": "Klasik ML",
    "lstm": "Derin öğrenme",
    "transformer": "Derin öğrenme",
}
MODEL_COLORS = {
    "naive_last_value": "#64748b",
    "drift": "#0f766e",
    "seasonal_naive_24h": "#f97316",
    "seasonal_naive_168h": "#d97706",
    "ma_24h": "#0891b2",
    "ar(1)": "#475569",
    "random_forest": "#16a34a",
    "xgboost": "#e11d48",
    "lightgbm": "#7c3aed",
    "lstm": "#2563eb",
    "transformer": "#9333ea",
}


@dataclass(frozen=True)
class MarketState:
    last_close: float
    change_24h_pip: float
    realized_vol_bp: float
    sample_start: pd.Timestamp
    sample_end: pd.Timestamp


st.set_page_config(
    page_title="EUR/USD AI Forecast",
    page_icon="💶",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #111827;
            --muted: #64748b;
            --line: #d9e1ea;
            --panel: #ffffff;
            --accent: #0f766e;
            --accent-soft: #dff7f3;
            --warn: #f97316;
        }
        [data-testid="stAppViewContainer"] {
            background: #f5f7fb;
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--line);
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1360px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--ink);
        }
        .demo-header {
            background: var(--panel);
            border: 1px solid var(--line);
            border-left: 6px solid var(--accent);
            border-radius: 8px;
            padding: 1.05rem 1.2rem;
            margin-bottom: 1rem;
        }
        .demo-header h1 {
            margin: 0 0 .25rem 0;
            font-size: 2rem;
            line-height: 1.15;
        }
        .demo-header p {
            margin: 0;
            color: var(--muted);
            font-size: .98rem;
        }
        .small-note {
            color: var(--muted);
            font-size: .86rem;
            line-height: 1.45;
        }
        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: .8rem .9rem;
        }
        div[data-testid="stMetricValue"] {
            color: var(--ink);
            font-size: 1.35rem;
        }
        div[data-testid="stMetricLabel"] {
            color: var(--muted);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: .25rem;
            border-bottom: 1px solid var(--line);
        }
        .stTabs [data-baseweb="tab"] {
            height: 2.75rem;
            border-radius: 6px 6px 0 0;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .stTabs [aria-selected="true"] {
            background: var(--accent-soft);
            color: var(--accent);
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_market_data() -> pd.DataFrame:
    df = load_raw_eurusd(RAW_CSV)
    df["log_ret_1"] = np.log(df["close"]).diff()
    df["pip_change"] = df["close"].diff() / PIP
    df["ret_bp"] = df["log_ret_1"] * 10_000
    df["rolling_vol_24h_bp"] = df["ret_bp"].rolling(24).std()
    df["rolling_vol_168h_bp"] = df["ret_bp"].rolling(168).std()
    df["ma_24h"] = df["close"].rolling(24).mean()
    df["ma_168h"] = df["close"].rolling(168).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    return df


@st.cache_data(show_spinner=False)
def load_scores() -> pd.DataFrame:
    if not SCORES_CSV.exists():
        return pd.DataFrame()
    scores = pd.read_csv(SCORES_CSV)
    scores["model_label"] = scores["model"].map(MODEL_LABELS).fillna(scores["model"])
    scores["family"] = scores["model"].map(MODEL_FAMILIES).fillna("Diğer")
    return scores


def pct_change_label(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f} pip"


def nearest_index(index: pd.DatetimeIndex, requested: pd.Timestamp) -> pd.Timestamp:
    eligible = index[index <= requested]
    if len(eligible) == 0:
        return index.min()
    return eligible.max()


def market_state(df: pd.DataFrame, end_ts: pd.Timestamp) -> MarketState:
    history = df.loc[:end_ts]
    last = float(history["close"].iloc[-1])
    close_24h = float(history["close"].iloc[-25]) if len(history) > 24 else last
    vol = float(history["ret_bp"].tail(168).std())
    return MarketState(
        last_close=last,
        change_24h_pip=(last - close_24h) / PIP,
        realized_vol_bp=vol,
        sample_start=df.index.min(),
        sample_end=df.index.max(),
    )


def model_options(scores: pd.DataFrame) -> list[str]:
    if scores.empty:
        return list(MODEL_LABELS)
    ordered = (
        scores.loc[scores["split"].eq("test")]
        .sort_values(["dir_acc_pct", "rmse_close_pip"], ascending=[False, True])
        ["model"]
        .tolist()
    )
    rest = [m for m in MODEL_LABELS if m not in ordered]
    return ordered + rest


def display_score(scores: pd.DataFrame, model: str) -> pd.Series:
    fallback = pd.Series(
        {
            "rmse_close_pip": np.nan,
            "mae_close_pip": np.nan,
            "dir_acc_pct": np.nan,
            "mape_close": np.nan,
        }
    )
    if scores.empty:
        return fallback
    subset = scores[(scores["split"] == "test") & (scores["model"] == model)]
    return subset.iloc[0] if not subset.empty else fallback


def _safe_autocorr(series: pd.Series) -> float:
    value = series.autocorr(1) if len(series) > 3 else 0.0
    if pd.isna(value):
        return 0.0
    return float(np.clip(value, -0.6, 0.6))


def _seasonal_returns(returns: pd.Series, horizon: int, lag: int) -> np.ndarray:
    if len(returns) <= lag + 2:
        return np.zeros(horizon)
    start = max(0, len(returns) - lag)
    pattern = returns.iloc[start : start + horizon].to_numpy()
    if len(pattern) == 0:
        pattern = returns.tail(min(lag, len(returns))).to_numpy()
    if len(pattern) < horizon:
        pattern = np.resize(pattern, horizon)
    return pattern[:horizon]


def forecast_path(history: pd.DataFrame, model: str, horizon: int) -> pd.Series:
    close = history["close"].astype(float)
    returns = np.log(close).diff().dropna()
    if len(returns) < 10:
        idx = pd.date_range(close.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
        return pd.Series(np.full(horizon, close.iloc[-1]), index=idx, name="forecast")

    last_close = float(close.iloc[-1])
    last_ret = float(returns.iloc[-1])
    mean_24 = float(returns.tail(24).mean())
    mean_168 = float(returns.tail(168).mean())
    vol = float(returns.tail(168).std())
    phi = _safe_autocorr(returns.tail(720))
    steps = np.arange(1, horizon + 1)

    if model == "naive_last_value":
        pred_rets = np.zeros(horizon)
    elif model == "drift":
        pred_rets = np.full(horizon, mean_168)
    elif model == "seasonal_naive_24h":
        pred_rets = _seasonal_returns(returns, horizon, 24)
    elif model == "seasonal_naive_168h":
        pred_rets = _seasonal_returns(returns, horizon, 168)
    elif model == "ma_24h":
        target = float(close.tail(24).mean())
        total_ret = np.log(max(target, 1e-9) / last_close)
        pred_rets = np.linspace(total_ret / horizon, mean_24, horizon) * 0.5
    elif model == "ar(1)":
        pred_rets = mean_168 + (phi ** steps) * (last_ret - mean_168)
    elif model == "random_forest":
        seasonal = _seasonal_returns(returns, horizon, 24)
        pred_rets = 0.35 * mean_24 + 0.25 * seasonal + 0.40 * np.exp(-steps / 18) * last_ret
    elif model == "xgboost":
        seasonal = _seasonal_returns(returns, horizon, 168)
        pred_rets = 0.30 * mean_24 + 0.30 * seasonal + 0.40 * np.exp(-steps / 14) * last_ret
    elif model == "lightgbm":
        seasonal = 0.55 * _seasonal_returns(returns, horizon, 24) + 0.45 * _seasonal_returns(returns, horizon, 168)
        pred_rets = 0.25 * mean_168 + 0.45 * seasonal + 0.30 * np.exp(-steps / 20) * last_ret
    elif model == "lstm":
        pred_rets = 0.45 * np.exp(-steps / 10) * last_ret + 0.35 * mean_24 + 0.20 * mean_168
    elif model == "transformer":
        seasonal = 0.5 * _seasonal_returns(returns, horizon, 24) + 0.5 * _seasonal_returns(returns, horizon, 168)
        pred_rets = 0.35 * seasonal + 0.35 * mean_168 + 0.30 * np.exp(-steps / 24) * last_ret
    else:
        pred_rets = np.full(horizon, mean_168)

    pred_rets = np.clip(pred_rets, -4 * vol, 4 * vol)
    values = last_close * np.exp(np.cumsum(pred_rets))
    idx = pd.date_range(close.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    return pd.Series(values, index=idx, name="forecast")


def scenario_frame(history: pd.DataFrame, forecast: pd.Series, n_paths: int, seed: int) -> pd.DataFrame:
    returns = np.log(history["close"]).diff().dropna()
    vol = float(returns.tail(168).std()) if len(returns) else 0.0005
    vol = max(vol, 0.00005)
    base_rets = np.diff(np.log(np.r_[history["close"].iloc[-1], forecast.to_numpy()]))
    rng = np.random.default_rng(seed)
    paths = {}
    for i in range(n_paths):
        scale = rng.uniform(0.65, 1.25)
        shocks = rng.normal(0.0, vol * scale, size=len(base_rets))
        paths[f"senaryo_{i + 1:02d}"] = history["close"].iloc[-1] * np.exp(np.cumsum(base_rets + shocks))
    return pd.DataFrame(paths, index=forecast.index)


def forecast_figure(history: pd.DataFrame, forecast: pd.Series, scenarios: pd.DataFrame, model: str) -> go.Figure:
    recent = history.tail(min(len(history), 24 * 45))
    q10 = scenarios.quantile(0.10, axis=1)
    q90 = scenarios.quantile(0.90, axis=1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["close"],
            mode="lines",
            name="Gerçek kapanış",
            line=dict(color="#111827", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=q90.index,
            y=q90,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=q10.index,
            y=q10,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(15, 118, 110, 0.14)",
            name="%10-%90 senaryo bandı",
            hoverinfo="skip",
        )
    )
    for col in scenarios.columns[: min(16, len(scenarios.columns))]:
        fig.add_trace(
            go.Scatter(
                x=scenarios.index,
                y=scenarios[col],
                mode="lines",
                name=col,
                line=dict(color="rgba(100, 116, 139, 0.26)", width=1),
                showlegend=False,
                hovertemplate="%{x}<br>%{y:.5f}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast,
            mode="lines+markers",
            name=MODEL_LABELS.get(model, model),
            line=dict(color=MODEL_COLORS.get(model, "#0f766e"), width=3),
            marker=dict(size=5),
            hovertemplate="%{x}<br>%{y:.5f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=35, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.04, x=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="EUR/USD", gridcolor="#e5e7eb"),
    )
    return fig


def rolling_backtest(df: pd.DataFrame, end_ts: pd.Timestamp, model: str, hours: int) -> pd.DataFrame:
    window = df.loc[:end_ts].tail(hours + 240)
    rows = []
    start = max(180, len(window) - hours - 1)
    for i in range(start, len(window) - 1):
        hist = window.iloc[: i + 1]
        actual_ts = window.index[i + 1]
        pred = float(forecast_path(hist, model, 1).iloc[0])
        actual = float(window["close"].iloc[i + 1])
        prev = float(window["close"].iloc[i])
        rows.append(
            {
                "time": actual_ts,
                "actual": actual,
                "predicted": pred,
                "error_pip": (pred - actual) / PIP,
                "dir_hit": np.sign(pred - prev) == np.sign(actual - prev),
            }
        )
    return pd.DataFrame(rows).set_index("time")


def backtest_figure(backtest: pd.DataFrame, model: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=backtest.index,
            y=backtest["actual"],
            mode="lines",
            name="Gerçek",
            line=dict(color="#111827", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=backtest.index,
            y=backtest["predicted"],
            mode="lines",
            name=MODEL_LABELS.get(model, model),
            line=dict(color=MODEL_COLORS.get(model, "#0f766e"), width=2),
        )
    )
    fig.update_layout(
        height=390,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05, x=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="EUR/USD", gridcolor="#e5e7eb"),
    )
    return fig


def score_scatter(scores: pd.DataFrame) -> go.Figure:
    test = scores[scores["split"] == "test"].copy()
    fig = px.scatter(
        test,
        x="rmse_close_pip",
        y="dir_acc_pct",
        color="family",
        hover_name="model_label",
        size="mae_close_pip",
        color_discrete_map={
            "Baseline": "#64748b",
            "Klasik ML": "#0f766e",
            "Derin öğrenme": "#f97316",
        },
        labels={
            "rmse_close_pip": "RMSE (pip, düşük daha iyi)",
            "dir_acc_pct": "Yön doğruluğu (%)",
            "family": "Aile",
            "mae_close_pip": "MAE pip",
        },
    )
    fig.update_traces(marker=dict(line=dict(color="white", width=1)))
    fig.update_layout(
        height=440,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#e5e7eb"),
        yaxis=dict(gridcolor="#e5e7eb"),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return fig


def scores_table(scores: pd.DataFrame) -> pd.DataFrame:
    test = scores[scores["split"] == "test"].copy()
    cols = ["model_label", "family", "rmse_close_pip", "mae_close_pip", "mape_close", "dir_acc_pct"]
    out = test[cols].sort_values(["dir_acc_pct", "rmse_close_pip"], ascending=[False, True])
    return out.rename(
        columns={
            "model_label": "Model",
            "family": "Aile",
            "rmse_close_pip": "RMSE pip",
            "mae_close_pip": "MAE pip",
            "mape_close": "MAPE %",
            "dir_acc_pct": "Yön doğruluğu %",
        }
    )


def image_gallery() -> list[Path]:
    names = [
        "eda_price_logret_vol.png",
        "ml_vs_baselines.png",
        "dl_vs_all.png",
        "ml_feature_importance.png",
        "dl_training_curves.png",
        "eda_heatmap_hour_dow.png",
    ]
    return [IMAGES_DIR / name for name in names if (IMAGES_DIR / name).exists()]


def make_forecast(df: pd.DataFrame, end_ts: pd.Timestamp, model: str, horizon: int) -> tuple[pd.Series, str]:
    history = df.loc[:end_ts]
    if model in TRAINED_DEEP_MODELS:
        return forecast_trained_path(history, model, horizon, CHECKPOINT_DIR), "Checkpoint t+1 inference + sönümlü yol"
    return forecast_path(history, model, horizon), "Demo/proxy tahmin"


def make_backtest(df: pd.DataFrame, end_ts: pd.Timestamp, model: str, hours: int) -> pd.DataFrame:
    if model in TRAINED_DEEP_MODELS:
        return rolling_backtest_trained(df, model, end_ts, hours, CHECKPOINT_DIR)
    return rolling_backtest(df, end_ts, model, hours)


@st.cache_data(show_spinner=False, ttl=900)
def load_live_data(period: str = "60d") -> pd.DataFrame:
    return fetch_live_eurusd(period=period, interval="1h")


def candlestick_with_forecasts(live_df: pd.DataFrame, forecasts: dict[str, pd.Series]) -> go.Figure:
    recent = live_df.tail(96)
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=recent.index,
            open=recent["open"],
            high=recent["high"],
            low=recent["low"],
            close=recent["close"],
            name="Canlı EUR/USD mumları",
            increasing_line_color="#0f766e",
            decreasing_line_color="#e11d48",
            increasing_fillcolor="rgba(15,118,110,0.45)",
            decreasing_fillcolor="rgba(225,29,72,0.45)",
        )
    )
    for model, forecast in forecasts.items():
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast.values,
                mode="lines+markers",
                name=MODEL_LABELS.get(model, model),
                line=dict(color=MODEL_COLORS.get(model, "#111827"), width=2.4),
                marker=dict(size=4),
                hovertemplate="%{x}<br>%{y:.5f}<extra></extra>",
            )
        )
    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, rangeslider=dict(visible=False)),
        yaxis=dict(title="EUR/USD", gridcolor="#e5e7eb"),
        legend=dict(orientation="h", y=1.08, x=0),
        hovermode="x unified",
    )
    return fig


def all_model_forecast_table(live_df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    models = ["lstm", "transformer", "xgboost", "random_forest", "lightgbm", "ar(1)", "naive_last_value"]
    forecasts: dict[str, pd.Series] = {}
    rows: list[dict[str, object]] = []
    last_close = float(live_df["close"].iloc[-1])
    for model in models:
        if model in TRAINED_DEEP_MODELS:
            forecast = forecast_trained_path(live_df, model, horizon, CHECKPOINT_DIR)
            source = "Gerçek checkpoint"
        else:
            forecast = forecast_path(live_df, model, horizon)
            source = "Demo/proxy"
        forecasts[model] = forecast
        rows.append(
            {
                "Model": MODEL_LABELS.get(model, model),
                "Kaynak": source,
                "İlk tahmin": round(float(forecast.iloc[0]), 5),
                "Ufuk sonu": round(float(forecast.iloc[-1]), 5),
                "Değişim pip": round((float(forecast.iloc[-1]) - last_close) / PIP, 2),
                "Yön": "Yukarı" if forecast.iloc[-1] > last_close else "Aşağı",
            }
        )
    return pd.DataFrame(rows), forecasts


def live_backtest_comparison(live_df: pd.DataFrame, hours: int) -> pd.DataFrame:
    models = ["lstm", "transformer", "xgboost", "random_forest", "lightgbm", "ar(1)", "naive_last_value"]
    rows = []
    end_ts = live_df.index.max()
    for model in models:
        try:
            bt = make_backtest(live_df, end_ts, model, hours)
            rows.append(
                {
                    "Model": MODEL_LABELS.get(model, model),
                    "Kaynak": "Gerçek checkpoint" if model in TRAINED_DEEP_MODELS else "Demo/proxy",
                    "Canlı RMSE pip": round(float(np.sqrt(np.mean(np.square(bt["error_pip"])))), 2),
                    "Canlı MAE pip": round(float(np.mean(np.abs(bt["error_pip"]))), 2),
                    "Canlı yön %": round(float(bt["dir_hit"].mean() * 100), 2),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "Model": MODEL_LABELS.get(model, model),
                    "Kaynak": "Hata",
                    "Canlı RMSE pip": np.nan,
                    "Canlı MAE pip": np.nan,
                    "Canlı yön %": np.nan,
                    "Not": str(exc)[:90],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    inject_css()

    if not RAW_CSV.exists():
        st.error(f"Ham veri bulunamadı: {RAW_CSV}")
        st.stop()

    df = load_market_data()
    scores = load_scores()
    available_models = model_options(scores)

    with st.sidebar:
        st.header("Demo kontrolleri")
        selected_model = st.selectbox(
            "Model",
            available_models,
            format_func=lambda value: MODEL_LABELS.get(value, value),
        )
        horizon = st.slider("Tahmin ufku", 6, 168, 48, step=6, format="%d saat")
        lookback_days = st.slider("Geçmiş pencere", 14, 120, 45, step=7, format="%d gün")
        scenario_count = st.slider("Senaryo sayısı", 20, 160, 80, step=20)
        backtest_hours = st.slider("Forward test", 48, 720, 240, step=24, format="%d saat")
        default_date = df.index.max().date()
        picked_date = st.date_input(
            "Bitiş tarihi",
            value=default_date,
            min_value=df.index.min().date(),
            max_value=df.index.max().date(),
        )
        picked_hour = st.slider("Saat", 0, 23, int(df.index.max().hour))
        end_ts = nearest_index(df.index, pd.Timestamp.combine(picked_date, time(picked_hour, 0)))
        st.caption(
            "Tahmin paneli yerel veri ve kayıtlı metriklerle çalışır. Finansal tavsiye değildir."
        )

    state = market_state(df, end_ts)
    history = df.loc[:end_ts].tail(24 * lookback_days)
    forecast, forecast_mode = make_forecast(df, end_ts, selected_model, horizon)
    scenarios = scenario_frame(df.loc[:end_ts], forecast, scenario_count, seed=42 + horizon)
    score = display_score(scores, selected_model)

    st.markdown(
        """
        <div class="demo-header">
            <h1>EUR/USD 1H AI Forecast Demo</h1>
            <p>Yapay Zekaya Giriş dönem projesi için Streamlit arayüzü: veri kesiti, tahmin, senaryo, metrik ve forward-test paneli.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Son kapanış", f"{state.last_close:.5f}", pct_change_label(state.change_24h_pip))
    col2.metric("168H volatilite", f"{state.realized_vol_bp:.2f} bp")
    col3.metric("Seçilen model RMSE", f"{score['rmse_close_pip']:.2f} pip" if pd.notna(score["rmse_close_pip"]) else "-")
    col4.metric("Test yön doğruluğu", f"{score['dir_acc_pct']:.2f}%" if pd.notna(score["dir_acc_pct"]) else "-")

    tab_live, tab_forecast, tab_backtest, tab_models, tab_data = st.tabs(
        ["Canlı karşılaştırma", "Tahmin paneli", "Forward test", "Model karşılaştırma", "Veri ve EDA"]
    )

    with tab_forecast:
        st.plotly_chart(forecast_figure(history, forecast, scenarios, selected_model), use_container_width=True)
        if selected_model in TRAINED_DEEP_MODELS:
            st.success(
                f"{MODEL_LABELS.get(selected_model, selected_model)} checkpoint'i yüklendi; "
                "ilk adım gerçek PyTorch inference ile, uzun ufuk ise sönümlü senaryo yolu olarak üretiliyor."
            )
        left, right = st.columns([1.2, 1])
        with left:
            endpoint = float(forecast.iloc[-1])
            change_pip = (endpoint - state.last_close) / PIP
            band = scenarios.quantile([0.1, 0.5, 0.9], axis=1).T
            st.subheader("Tahmin özeti")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Alan": ["Model", "Tahmin türü", "Ufuk", "Son kapanış", "Tahmin sonu", "Beklenen değişim", "Senaryo medyan sonu"],
                        "Değer": [
                            MODEL_LABELS.get(selected_model, selected_model),
                            forecast_mode,
                            f"{horizon} saat",
                            f"{state.last_close:.5f}",
                            f"{endpoint:.5f}",
                            pct_change_label(change_pip),
                            f"{float(band[0.5].iloc[-1]):.5f}",
                        ],
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
        with right:
            st.subheader("Senaryo aralığı")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Saat": band.index,
                        "P10": band[0.1].round(5),
                        "P50": band[0.5].round(5),
                        "P90": band[0.9].round(5),
                    }
                ).tail(12),
                hide_index=True,
                use_container_width=True,
            )
        scenario_export = scenarios.copy()
        scenario_export.insert(0, "forecast", forecast)
        st.download_button(
            "Senaryo CSV indir",
            data=scenario_export.to_csv(index_label="time").encode("utf-8"),
            file_name="eurusd_forecast_scenarios.csv",
            mime="text/csv",
        )

    with tab_live:
        st.subheader("Canlı EUR/USD mumları ve model çıktıları")
        try:
            live_df = load_live_data("60d")
            live_last = float(live_df["close"].iloc[-1])
            live_prev = float(live_df["close"].iloc[-25]) if len(live_df) > 24 else live_last
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Canlı son kapanış", f"{live_last:.5f}", pct_change_label((live_last - live_prev) / PIP))
            c2.metric("Canlı son mum", live_df.index.max().strftime("%Y-%m-%d %H:%M"))
            c3.metric("Canlı mum sayısı", f"{len(live_df):,}")
            c4.metric("Kaynak", "Yahoo EURUSD=X")

            compare_table, compare_forecasts = all_model_forecast_table(live_df, horizon)
            st.plotly_chart(candlestick_with_forecasts(live_df, compare_forecasts), use_container_width=True)
            st.caption(
                "LSTM ve Transformer satırları kayıtlı .pt checkpoint ile üretilir. "
                "Diğer modeller demo/proxy karşılaştırma çizgisidir."
            )
            left, right = st.columns([1.05, 1])
            with left:
                st.subheader("Canlı veri üstünde ileri tahmin")
                st.dataframe(compare_table, hide_index=True, use_container_width=True)
            with right:
                st.subheader("Canlı rolling karşılaştırma")
                live_cmp = live_backtest_comparison(live_df, min(backtest_hours, 240))
                st.dataframe(live_cmp, hide_index=True, use_container_width=True)
        except Exception as exc:
            st.warning(
                "Canlı veri şu anda alınamadı. İnternet/Yahoo Finance erişimi gelince bu panel otomatik çalışır."
            )
            st.code(str(exc))

    with tab_backtest:
        backtest = make_backtest(df, end_ts, selected_model, backtest_hours)
        rmse = float(np.sqrt(np.mean(np.square(backtest["error_pip"]))))
        mae = float(np.mean(np.abs(backtest["error_pip"])))
        hit = float(backtest["dir_hit"].mean() * 100)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rolling RMSE", f"{rmse:.2f} pip")
        c2.metric("Rolling MAE", f"{mae:.2f} pip")
        c3.metric("Yön isabeti", f"{hit:.2f}%")
        st.plotly_chart(backtest_figure(backtest, selected_model), use_container_width=True)
        st.dataframe(
            backtest.tail(18).assign(error_pip=lambda x: x["error_pip"].round(2)),
            use_container_width=True,
        )

    with tab_models:
        if scores.empty:
            st.warning("Skor tablosu bulunamadı.")
        else:
            st.plotly_chart(score_scatter(scores), use_container_width=True)
            st.dataframe(scores_table(scores), hide_index=True, use_container_width=True)

    with tab_data:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Toplam mum", f"{len(df):,}")
        c2.metric("Başlangıç", state.sample_start.strftime("%Y-%m-%d"))
        c3.metric("Bitiş", state.sample_end.strftime("%Y-%m-%d"))
        c4.metric("Ham sütun", "8")
        st.subheader("Son veri kesiti")
        st.dataframe(df.tail(20).round(6), use_container_width=True)
        st.subheader("Rapor görselleri")
        gallery = image_gallery()
        if gallery:
            cols = st.columns(3)
            for i, image_path in enumerate(gallery):
                cols[i % 3].image(str(image_path), caption=image_path.stem.replace("_", " "))
        else:
            st.info("docs/images klasöründe görsel bulunamadı.")

    st.markdown(
        '<p class="small-note">Bu demo akademik amaçlıdır. Forex piyasası yüksek risklidir; model çıktısı gerçek işlem kararı için kullanılmamalıdır.</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
