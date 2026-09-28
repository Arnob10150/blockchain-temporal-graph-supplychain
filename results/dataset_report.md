# SupplyGraph dataset audit

Source: CIOL-SUST/SupplyGraph (cloned into `data/raw/SupplyGraph`). Homogeneous
product graph of a Bangladeshi FMCG manufacturer. Used here strictly as an
academic benchmark; no claim is made that it represents Bangladesh exports or
any specific company deployment.

## Nodes and edges
- **Nodes:** 41 product SKUs (from the temporal signal columns; `Nodes.csv`
  lists 41 with a 40-name overlap, so the node set is taken from the temporal
  files and edges are intersected against it).
- **Product--product edge types** (undirected, homogeneous):
  - Shared **Plant**: 1,647 edges
  - Shared **Product Group**: 188 edges
  - Shared **Product Sub-Group**: 52 edges
  - Shared **Storage Location**: 3,046 edges
- The forecasting graph uses the **union** of these relationships (an edge exists
  if two SKUs share any of the four), symmetrically normalized with self-loops.

## Temporal signals
Four daily signals, each a `Date x 41-product` matrix:

| Signal | Rows (dates) | Products | Missing | Zeros | min / median / max |
|---|---|---|---|---|---|
| Production | 221 | 41 | 0.0% | 62.0% | 0 / 0 / 18,859 |
| Sales Order | 221 | 41 | 0.0% | 46.1% | 0 / 15 / 29,251 |
| Factory Issue | 221 | 41 | 0.0% | 43.9% | 0 / 45 / 19,617 |
| Delivery to distributor | 221 | 41 | 0.0% | 35.8% | 0 / 90 / 18,517 |

- **Date range:** 2023-01-01 to 2023-08-09 (221 timestamps, 220 usable steps).
- **No missing values.** Production is **intermittent** (62% zeros), which makes
  next-period production a genuinely hard, sparse regression target and makes a
  persistence baseline non-trivial to beat.

## Prediction task (frozen)
- **Single task: next-period product-level production forecasting.**
- **Inputs:** per node, a lookback window (L = 14 days) of the four signals
  (production, sales order, factory issue, delivery), `log1p`-transformed and
  standardized using **train-split statistics only**.
- **Target:** next-day production per node (inverse-transformed to original units
  for reporting).
- **Split:** strictly **chronological** — earliest 70% train, next 15%
  validation, latest 15% test. No shuffling.
- **Metrics:** MAE, RMSE, R^2, sMAPE (reported in original production units).

## Provenance / blockchain layer (design)
- **Snapshot = one date's 41-product production vector** (plus the three
  auxiliary signals), canonically serialized and hashed with **SHA-256**
  (off-chain data, on-chain hash). Hashes are registered in a `DataProvenance`
  ledger keyed by date.
- Tampering experiments alter 1/5/10% of test production values **without**
  updating the registered hash; verification recomputes each snapshot hash and
  flags mismatches; recovery substitutes the last verified snapshot.
