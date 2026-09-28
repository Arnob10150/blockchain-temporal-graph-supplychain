"""Blockchain / provenance overhead benchmark (real measurements).

Measures SHA-256 snapshot-hash generation, ledger registration, and
verification latency over many snapshots. These are wall-clock measurements of
the off-chain hashing and on-ledger operations; on-chain EVM gas is measured
separately by the Hardhat script (blockchain/scripts/measure_gas.js) when a
local node is available.
"""
from __future__ import annotations
import sys, time, json, statistics
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import core as C

TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)


class Ledger:
    """Minimal append-only, write-once hash ledger (mirrors the Solidity map)."""
    def __init__(self): self.store = {}
    def register(self, date, h):
        if date in self.store:
            raise ValueError("already registered")
        self.store[date] = h
    def get(self, date): return self.store.get(date)
    def verify(self, date, h): return self.store.get(date) == h


def main():
    dates, products, X = C.load_signals()
    n = len(dates)

    # hash-generation latency
    t = []
    for i in range(n):
        t0 = time.perf_counter(); C.snapshot_hash(dates[i], X[i, :, 0], products)
        t.append((time.perf_counter() - t0) * 1e3)
    hash_ms = statistics.mean(t)

    # registration latency
    led = Ledger(); rt = []
    for i in range(n):
        h = C.snapshot_hash(dates[i], X[i, :, 0], products)
        t0 = time.perf_counter(); led.register(str(dates[i])[:10], h)
        rt.append((time.perf_counter() - t0) * 1e3)
    reg_ms = statistics.mean(rt)

    # verification latency (recompute hash + compare) over repeated passes
    vt = []
    for _ in range(5):
        for i in range(n):
            t0 = time.perf_counter()
            h = C.snapshot_hash(dates[i], X[i, :, 0], products)
            led.verify(str(dates[i])[:10], h)
            vt.append((time.perf_counter() - t0) * 1e3)
    ver_ms = statistics.mean(vt)

    rows = [
        {"operation": "SHA-256 snapshot hash", "mean_latency_ms": round(hash_ms, 4),
         "count": n, "note": "41-product snapshot, canonical JSON"},
        {"operation": "Ledger register (off-chain map)", "mean_latency_ms": round(reg_ms, 4),
         "count": n, "note": "write-once insert"},
        {"operation": "Verify (rehash + compare)", "mean_latency_ms": round(ver_ms, 4),
         "count": len(vt), "note": "detects any content change"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "provenance_overhead.csv", index=False)
    print(df.to_string(index=False))

    # merge on-chain gas if the Hardhat measurement produced it
    gas_path = ROOT / "results" / "tables" / "onchain_gas.json"
    if gas_path.exists():
        gas = json.load(open(gas_path))
        print("on-chain gas:", gas)
    else:
        print("on-chain gas: (run blockchain/scripts/measure_gas.js via Hardhat)")


if __name__ == "__main__":
    main()
