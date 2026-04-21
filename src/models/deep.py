"""Küçük LSTM ve Transformer encoder'ları, EUR/USD 1H log-getiri tahmini için.

Iki mimari de scalar regresyon yapar:
  y_{t+1} = f( x_{t-L+1 : t} )     # L = seq_len

Mimariler bilinçli olarak küçük tutulmuştur; CPU üzerinde 5 epoch ~dakikalar
mertebesinde biter. Transformer bölümü positional encoding öğrenir.
"""
from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """Iki katmanlı LSTM + son adımın çıktısı üzerinde lineer regresyon başı."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, F)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)  # (B,)


class _LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int) -> None:
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerForecaster(nn.Module):
    """Hafif bir transformer encoder.

    Regresyon için iki kritik ayar:
    - Giriş projeksiyonu sonrası LayerNorm (ölçek kontrolü).
    - Çıkış başı sıfır ağırlıklarla başlatılır → başlangıçta y_hat = 0 (log-getiri için iyi prior).
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos = _LearnablePositionalEncoding(seq_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, F)
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = self.pos(h)
        h = self.encoder(h)
        return self.head(h[:, -1, :]).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
