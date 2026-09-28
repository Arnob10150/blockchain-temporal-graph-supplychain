"""Core: data loading, temporal-graph construction, models, metrics.

Next-period product-level production forecasting on the SupplyGraph benchmark.
Everything is deterministic (fixed seeds) and uses a strictly chronological
train/val/test split. Node features are the four daily signals
(production, sales order, factory issue, delivery), log1p-transformed and
standardized on TRAIN statistics only; the target is next-day production.
"""
from __future__ import annotations
import os, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "SupplyGraph" / "Raw Dataset" / "Homogenoeus"
LOOKBACK = 14
SIGNALS = {
    "production": "Temporal Data/Unit/Production .csv",
    "sales":      "Temporal Data/Unit/Sales Order.csv",
    "issue":      "Temporal Data/Unit/Factory Issue.csv",
    "delivery":   "Temporal Data/Unit/Delivery To distributor.csv",
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_signals():
    """Return dates (T,), product list (N,), and raw tensor X_raw (T,N,F).

    Rows in the source CSVs are NOT stored in date order, so every signal is
    re-indexed by date and sorted chronologically before use, which is required
    for a valid chronological forecasting split.
    """
    frames = {k: pd.read_csv(RAW / v) for k, v in SIGNALS.items()}
    prod = frames["production"]
    products = list(prod.columns[1:])
    F = len(SIGNALS)
    # canonical chronological order from the production dates
    d0 = pd.to_datetime(prod["Date"], errors="coerce")
    order = np.argsort(d0.values, kind="stable")
    dates = d0.values[order]
    T, N = len(prod), len(products)
    X = np.zeros((T, N, F), dtype=np.float64)
    for fi, key in enumerate(SIGNALS):
        df = frames[key]
        di = pd.to_datetime(df["Date"], errors="coerce")
        o = np.argsort(di.values, kind="stable")          # sort each signal by its own date
        X[:, :, fi] = df[products].values[o].astype(np.float64)
    return dates, products, X


def build_adjacency(products):
    """Product-product graph from shared Plant / Group / Sub-Group relations.

    Storage-Location co-location is deliberately excluded: it connects almost
    every pair (3046 edges) and makes the graph near-complete, which causes
    graph-convolution over-smoothing. Group/Sub-Group/Plant give a sparser,
    business-meaningful similarity graph.
    """
    idx = {p: i for i, p in enumerate(products)}
    N = len(products)
    A = np.zeros((N, N), dtype=np.float64)
    edge_files = [
        ("Edges/Edges (Product Group).csv", "node1", "node2"),
        ("Edges/Edges (Product Sub-Group).csv", "node1", "node2"),
    ]
    raw_edges = 0
    for f, a, b in edge_files:
        d = pd.read_csv(RAW / f)
        for u, v in zip(d[a], d[b]):
            if u in idx and v in idx and u != v:
                A[idx[u], idx[v]] = 1.0
                A[idx[v], idx[u]] = 1.0
                raw_edges += 1
    n_edges = int(A.sum() // 2)
    # symmetric normalization with self-loops:  D^-1/2 (A+I) D^-1/2
    A_hat = A + np.eye(N)
    deg = A_hat.sum(1)
    dinv = np.diag(1.0 / np.sqrt(deg))
    A_norm = dinv @ A_hat @ dinv
    return A.astype(np.float32), A_norm.astype(np.float32), n_edges


def chronological_split(T, lookback=LOOKBACK, ratios=(0.70, 0.15, 0.15)):
    """Prediction indices t (predict production at t from window [t-L, t-1])."""
    idx = np.arange(lookback, T)          # target time indices
    n = len(idx)
    n_tr = int(round(ratios[0] * n)); n_va = int(round(ratios[1] * n))
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


class Standardizer:
    """log1p then per-feature standardization, fit on train targets/inputs."""
    def __init__(self): self.mu = None; self.sd = None
    def fit(self, X):                      # X: (samples, ..., F)
        Z = np.log1p(np.clip(X, 0, None))
        self.mu = Z.reshape(-1, Z.shape[-1]).mean(0)
        self.sd = Z.reshape(-1, Z.shape[-1]).std(0) + 1e-6
        return self
    def tf(self, X):  return (np.log1p(np.clip(X, 0, None)) - self.mu) / self.sd
    def inv_prod(self, z):                 # invert production channel (feature 0)
        return np.expm1(z * self.sd[0] + self.mu[0])
    def tf_prod(self, y):                  # transform a raw production target
        return (np.log1p(np.clip(y, 0, None)) - self.mu[0]) / self.sd[0]


def make_windows(X, target_idx, lookback=LOOKBACK):
    """Return W (S, L, N, F) inputs and Y (S, N) raw production targets."""
    W, Y = [], []
    for t in target_idx:
        W.append(X[t - lookback:t])        # (L, N, F)
        Y.append(X[t, :, 0])               # production at t (raw)
    return np.stack(W), np.stack(Y)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def persistence(X, target_idx):
    """Predict production at t == production at t-1 (last observed)."""
    return np.stack([X[t - 1, :, 0] for t in target_idx])   # (S, N) raw units


class LSTMForecaster(nn.Module):
    """Temporal baseline; shared weights across nodes, no graph.

    Persistence-anchored: predicts a correction added to the last observed
    (standardized) production, so it can never do worse than persistence by much
    and learns only the residual dynamics.
    """
    def __init__(self, in_dim=4, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, w):                  # w: (B, L, F)
        h, _ = self.lstm(w)
        return w[:, -1, 0] + self.head(h[:, -1]).squeeze(-1)


class GConvLSTMCell(nn.Module):
    """LSTM cell whose gates ingest graph-convolved features; batched over B."""
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.hidden = hidden
        self.gc_x = nn.Linear(in_dim, 4 * hidden)
        self.gc_h = nn.Linear(hidden, 4 * hidden)
    def forward(self, x, h, c, A):         # x:(B,N,in) h,c:(B,N,hid) A:(N,N)
        gx = torch.einsum("nm,bmf->bnf", A, self.gc_x(x))
        gh = torch.einsum("nm,bmf->bnf", A, self.gc_h(h))
        i, f, g, o = (gx + gh).chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c = f * c + i * torch.tanh(g)
        return o * torch.tanh(c), c


class GConvLSTM(nn.Module):
    """Temporal graph model: graph-convolutional LSTM, persistence-anchored."""
    def __init__(self, in_dim=4, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.cell = GConvLSTMCell(in_dim, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, w, A):               # w:(B,L,N,F) A:(N,N)
        B, L, N, F = w.shape
        h = torch.zeros(B, N, self.hidden, device=w.device)
        c = torch.zeros(B, N, self.hidden, device=w.device)
        for t in range(L):
            h, c = self.cell(w[:, t], h, c, A)
        corr = self.head(h).squeeze(-1)    # (B, N)
        return w[:, -1, :, 0] + corr       # persistence anchor + graph correction


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metrics(y_true, y_pred):
    yt = np.asarray(y_true, float).ravel(); yp = np.asarray(y_pred, float).ravel()
    mae = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    ss_res = np.sum((yt - yp) ** 2); ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    denom = (np.abs(yt) + np.abs(yp)) + 1e-6
    smape = np.mean(2 * np.abs(yt - yp) / denom) * 100
    return dict(MAE=float(mae), RMSE=float(rmse), R2=float(r2), sMAPE=float(smape))


# --------------------------------------------------------------------------- #
# Provenance (SHA-256, off-chain data / on-chain hash)
# --------------------------------------------------------------------------- #
def snapshot_hash(date, production_vector, products):
    """Deterministic canonical serialization -> SHA-256 hex digest."""
    payload = {"date": str(date)[:10],
               "production": {p: int(v) for p, v in zip(products, np.asarray(production_vector).tolist())}}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()
