# Blockchain-Assisted Temporal Graph Learning for Trustworthy Supply Chain Forecasting

Code for *Blockchain-Assisted Temporal Graph Learning for Trustworthy Supply Chain Forecasting:
A Bangladesh Case Study* (IEEE CSDE 2026).

Arnob Aich Anurag, Shamiul Islam, MD.Nasif Rafidi, MD. Sadman Hossain, Nuren Tasnim Risha, Trisha Sarker, Samiha Islam Abrita

## What is here

```
src/core.py                     data loading, product graph, Persistence / LSTM / GConvLSTM models
experiments/clean.py            forecasting on clean data (chronological 145/31/31 split)
experiments/tampering.py        1/5/10% tampering, SHA-256 detection, last-verified recovery
experiments/blockchain_benchmark.py   hashing / registration / verification overhead
blockchain/contracts/DataProvenance.sol   write-once on-chain snapshot-hash registry
blockchain/scripts/measure_gas.js         Hardhat gas measurement
results/tables/                 measured results reported in the paper
```

## Key results

- Clean data: LSTM and GConvLSTM reach R² 0.57, against 0.52 for persistence.
- At 5% tampering R² falls to 0.33. Hash verification detects 100% of tampered snapshots, and last-verified recovery restores R² to 0.43.
- Hashing or verifying one snapshot takes about 0.015 ms.

## Data (not included)

The SupplyGraph benchmark (Wasi et al., 2024) is used as published:

```bash
git clone https://github.com/CIOL-SUST/SupplyGraph data/raw/SupplyGraph
```

## Reproduce

```bash
pip install -r requirements.txt
python experiments/clean.py
python experiments/tampering.py
python experiments/blockchain_benchmark.py
cd blockchain && npm install && npx hardhat compile && npx hardhat run scripts/measure_gas.js
```

Seed 42 throughout.

## License

Code is released for academic use. Third-party datasets remain under their original licenses.
