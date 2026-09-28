"""Tampering, SHA-256 verification, and recovery experiment.

Registers clean snapshot hashes, then at 0/1/5/10% tampering of the test-period
production data measures: blockchain detection rate, number of corrupted
snapshots, and forecasting error under clean / corrupted / blockchain-recovered
conditions using the trained GConvLSTM. Recovery replaces any snapshot that
fails verification with the last verified snapshot (carry-forward).
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import core as C
from clean import predict, train_torch

TAB = ROOT / "results" / "tables"; CACHE = ROOT / "results" / "cache"
np.random.seed(C.SEED); torch.manual_seed(C.SEED)
LEVELS = [0.0, 0.01, 0.05, 0.10]


def tamper(Xprod, dates_range, frac, rng):
    """Alter `frac` of (date,node) production entries in dates_range. Returns
    tampered production matrix and the set of tampered date indices."""
    Xt = Xprod.copy()
    N = Xt.shape[1]
    cells = [(d, n) for d in dates_range for n in range(N)]
    k = int(round(frac * len(cells)))
    picked = rng.choice(len(cells), size=k, replace=False) if k > 0 else []
    tampered_dates = set()
    for pi in picked:
        d, n = cells[pi]
        orig = Xt[d, n]
        # realistic corruption: scale by a random factor far from 1, or inject spike
        factor = rng.choice([0.0, 0.3, 3.0, 5.0])
        newv = orig * factor if orig > 0 else rng.integers(500, 5000)
        if newv == orig:
            newv = orig + rng.integers(100, 1000)
        Xt[d, n] = newv
        tampered_dates.add(d)
    return Xt, tampered_dates


def main():
    dates, products, X = C.load_signals()
    A_bin, A_norm, n_edges = C.build_adjacency(products)
    tr_idx, va_idx, te_idx = C.chronological_split(len(dates))
    std = C.Standardizer().fit(X[:tr_idx[-1] + 1])
    Wtr, Ytr = C.make_windows(X, tr_idx); Wva, Yva = C.make_windows(X, va_idx)
    Wte, Yte = C.make_windows(X, te_idx)

    # trained GConvLSTM (reuse cache if present, else train)
    g = C.GConvLSTM()
    cache = CACHE / "models.pt"
    if cache.exists():
        g.load_state_dict(torch.load(cache)["gclstm"])
    else:
        g, _ = train_torch(g, True, Wtr, Ytr, Wva, Yva, A_norm, std)

    # register clean hashes for every date (on-chain hash of production vector)
    ledger = {str(dates[t])[:10]: C.snapshot_hash(dates[t], X[t, :, 0], products)
              for t in range(len(dates))}

    # test input region = dates feeding the test windows
    in_lo = int(te_idx[0]) - C.LOOKBACK
    in_range = list(range(in_lo, int(te_idx[-1])))

    rng = np.random.default_rng(C.SEED)
    rows = []
    for frac in LEVELS:
        Xp = X.copy()
        Xt_prod, tampered = tamper(X[:, :, 0], in_range, frac, rng)
        Xp[:, :, 0] = Xt_prod

        # ---- blockchain verification over the test input region ----
        detected = []
        for d in in_range:
            h_now = C.snapshot_hash(dates[d], Xp[d, :, 0], products)
            if h_now != ledger[str(dates[d])[:10]]:
                detected.append(d)
        detected = set(detected)
        det_rate = (len(detected & tampered) / len(tampered)) if tampered else 1.0

        # ---- corrupted forecast (tampered inputs) ----
        Wc, _ = C.make_windows(Xp, te_idx)
        p_corr = predict(g, True, Wc, A_norm, std)
        m_corr = C.metrics(Yte, p_corr)

        # ---- recovery: replace verified-failed snapshots with last good ----
        Xr = Xp.copy()
        last_good = None
        for d in range(len(dates)):
            if d in detected:                      # verification failed -> reject
                if last_good is not None:
                    Xr[d, :, 0] = X[last_good, :, 0]
            else:
                last_good = d                      # trusted snapshot
        Wr, _ = C.make_windows(Xr, te_idx)
        p_rec = predict(g, True, Wr, A_norm, std)
        m_rec = C.metrics(Yte, p_rec)

        rows.append({
            "tamper_pct": int(frac * 100),
            "corrupted_snapshots": len(tampered),
            "detection_rate": round(det_rate, 4),
            "RMSE_corrupted": round(m_corr["RMSE"], 2),
            "R2_corrupted": round(m_corr["R2"], 4),
            "RMSE_recovered": round(m_rec["RMSE"], 2),
            "R2_recovered": round(m_rec["R2"], 4),
        })
        print(rows[-1])

    # clean reference (frac 0 corrupted == clean)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "tampering_results.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
