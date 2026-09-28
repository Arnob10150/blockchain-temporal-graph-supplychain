"""Clean-data experiment: persistence, LSTM, GConvLSTM on chronological split.

Saves: results/tables/model_comparison.csv, predictions.csv, training_history.csv
and a fitted-model cache for the tampering experiment.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import core as C

TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
LOG = ROOT / "results" / "logs"; LOG.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "results" / "cache"; CACHE.mkdir(parents=True, exist_ok=True)
torch.manual_seed(C.SEED); np.random.seed(C.SEED)
DEV = "cpu"


def train_torch(model, is_graph, Wtr, Ytr, Wva, Yva, A, std, epochs=120, lr=2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    lossf = torch.nn.SmoothL1Loss()
    A_t = torch.tensor(A, device=DEV)
    ytr = torch.tensor(std.tf_prod(Ytr), dtype=torch.float32, device=DEV)
    yva = torch.tensor(std.tf_prod(Yva), dtype=torch.float32, device=DEV)
    hist = []
    best = (1e18, None)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        if is_graph:
            xb = torch.tensor(std.tf(Wtr), dtype=torch.float32, device=DEV)   # (S,L,N,F)
            pred = model(xb, A_t)
            loss = lossf(pred, ytr)
        else:
            # flatten nodes into the batch: (S*N, L, F)
            S, L, N, F = Wtr.shape
            xb = torch.tensor(std.tf(Wtr).transpose(0, 2, 1, 3).reshape(S * N, L, F),
                              dtype=torch.float32, device=DEV)
            pred = model(xb).reshape(S, N)
            loss = lossf(pred, ytr)
        loss.backward(); opt.step()
        # validation
        model.eval()
        with torch.no_grad():
            if is_graph:
                xv = torch.tensor(std.tf(Wva), dtype=torch.float32, device=DEV)
                pv = model(xv, A_t)
            else:
                S, L, N, F = Wva.shape
                xv = torch.tensor(std.tf(Wva).transpose(0, 2, 1, 3).reshape(S * N, L, F),
                                  dtype=torch.float32, device=DEV)
                pv = model(xv).reshape(S, N)
            vloss = float(lossf(pv, yva))
        hist.append({"epoch": ep, "train_loss": float(loss), "val_loss": vloss})
        if vloss < best[0]:
            best = (vloss, {k: v.detach().clone() for k, v in model.state_dict().items()})
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model, hist


def predict(model, is_graph, W, A, std):
    model.eval(); A_t = torch.tensor(A, device=DEV)
    with torch.no_grad():
        if is_graph:
            x = torch.tensor(std.tf(W), dtype=torch.float32, device=DEV)
            p = model(x, A_t).cpu().numpy()
        else:
            S, L, N, F = W.shape
            x = torch.tensor(std.tf(W).transpose(0, 2, 1, 3).reshape(S * N, L, F),
                             dtype=torch.float32, device=DEV)
            p = model(x).reshape(S, N).cpu().numpy()
    return np.clip(std.inv_prod(p), 0, None)     # back to raw units, non-negative


def main():
    dates, products, X = C.load_signals()
    A_bin, A_norm, n_edges = C.build_adjacency(products)
    tr_idx, va_idx, te_idx = C.chronological_split(len(dates))
    std = C.Standardizer().fit(X[:tr_idx[-1] + 1])          # train-only stats

    Wtr, Ytr = C.make_windows(X, tr_idx); Wva, Yva = C.make_windows(X, va_idx)
    Wte, Yte = C.make_windows(X, te_idx)

    rows, preds, hists = [], {}, {}

    # 1) persistence
    p_per = C.persistence(X, te_idx)
    rows.append({"Model": "Persistence", **C.metrics(Yte, p_per)})
    preds["persistence"] = p_per

    # 2) LSTM
    t0 = time.time()
    lstm = C.LSTMForecaster().to(DEV)
    lstm, h1 = train_torch(lstm, False, Wtr, Ytr, Wva, Yva, A_norm, std, epochs=200)
    p_lstm = predict(lstm, False, Wte, A_norm, std)
    rows.append({"Model": "LSTM", **C.metrics(Yte, p_lstm), "train_s": round(time.time() - t0, 1)})
    preds["lstm"] = p_lstm; hists["lstm"] = h1

    # 3) GConvLSTM
    t0 = time.time()
    g = C.GConvLSTM().to(DEV)
    g, h2 = train_torch(g, True, Wtr, Ytr, Wva, Yva, A_norm, std, epochs=200)
    p_g = predict(g, True, Wte, A_norm, std)
    rows.append({"Model": "GConvLSTM", **C.metrics(Yte, p_g), "train_s": round(time.time() - t0, 1)})
    preds["gclstm"] = p_g; hists["gclstm"] = h2

    # save
    comp = pd.DataFrame(rows)
    comp.to_csv(TAB / "model_comparison.csv", index=False)
    print(comp.to_string(index=False))

    # predictions long-form (node,date,actual,pred per model) for plotting
    te_dates = [str(dates[t])[:10] for t in te_idx]
    pr = []
    for si, t in enumerate(te_idx):
        for ni, prod in enumerate(products):
            pr.append({"date": te_dates[si], "product": prod, "actual": float(Yte[si, ni]),
                       "persistence": float(p_per[si, ni]), "lstm": float(p_lstm[si, ni]),
                       "gclstm": float(p_g[si, ni])})
    pd.DataFrame(pr).to_csv(TAB / "predictions.csv", index=False)
    pd.concat([pd.DataFrame(h).assign(model=m) for m, h in hists.items()]).to_csv(
        TAB / "training_history.csv", index=False)

    # cache for tampering experiment
    torch.save({"gclstm": g.state_dict(), "lstm": lstm.state_dict()}, CACHE / "models.pt")
    meta = {"n_nodes": len(products), "n_edges": int(n_edges), "T": int(len(dates)),
            "lookback": C.LOOKBACK, "features": list(C.SIGNALS.keys()),
            "n_train": len(tr_idx), "n_val": len(va_idx), "n_test": len(te_idx),
            "date_start": str(dates[0])[:10], "date_end": str(dates[-1])[:10]}
    json.dump(meta, open(TAB / "dataset_summary.json", "w"), indent=2)
    print("meta:", meta)


if __name__ == "__main__":
    main()
