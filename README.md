# UTXO — AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

**SIH 2026 · Problem Statement ID: SIH26146 · Organization: NTRO · Theme: Blockchain & Cybersecurity**

## The Problem

Bitcoin transactions are publicly visible on the blockchain (wallet addresses, TXIDs, amounts, fees), but that data alone rarely tells you *who* actually sent a transaction or *where it came from*. Separately, the Bitcoin P2P network carries a second layer of signal — which IP/port relayed a transaction first, and when — that most existing analysis tools completely ignore.

NTRO's ask: build a system that **fuses both layers** — the blockchain ledger and the P2P network traffic — to detect potentially illicit Bitcoin activity, and present it in a way an analyst can actually act on: a ranked, explainable list of alerts, not just a black-box anomaly score.

Existing tools (GraphSense, BlockSci, Chainalysis) only look at the blockchain layer. None of them properly correlate network-layer IP/timing data with wallet/TXID data — that fusion is our core differentiator.

## What We're Building

An offline, Linux-compatible pipeline + dashboard that:

1. **Fuses** blockchain data (wallets, TXIDs, amounts, script types) with network data (src/dst IP, port, timestamp) into a single **entity graph** — wallets, IPs, and TXIDs as nodes, with typed edges (co-spend, first-broadcast, transaction flow).
2. **Detects anomalies** on that graph using unsupervised ML (Isolation Forest) over features like transaction burst frequency, amount deviation, co-spend cluster size, geo/ASN mismatch, and Tor-exit-node usage.
3. **Explains every alert** using SHAP — so an analyst sees *why* something was flagged, not just a score.
4. **Visualizes** everything in an interactive Streamlit + pyvis dashboard, with a ranked alert table and a clickable link-analysis graph.

Since no real dataset was provided, we generate a **realistic synthetic dataset** (with injected ground-truth illicit clusters) to build, test, and demo against.

**Important honest limitation we lead with, not hide:** identifying the "initiator" of a transaction via IP is *probabilistic, not guaranteed* — Bitcoin's gossip-relay network means the first-seen IP isn't always the true sender, and VPN/Tor can hide it entirely. So we report a **confidence score**, and when the network-layer signal is weak (VPN/Tor in use), we fall back on IP-independent blockchain heuristics (co-spend clustering, peeling-chain detection, behavioral fingerprinting) — and treat anonymization behavior itself as an anomaly signal.

## Docs (read in this order)

1. [`PRD.md`](./PRD.md) — full scope, functional requirements, data schema, success metrics, limitations.
2. [`ARCHITECTURE.md`](./ARCHITECTURE.md) — system design: data flow, module breakdown, tech stack, folder structure, module interfaces for parallel development.
3. `requirements.txt` — Python dependencies.

## Tech Stack (at a glance)

Synthetic data (Faker) → Ingestion (pandas) → Entity graph (NetworkX) → Anomaly detection (scikit-learn Isolation Forest) → Explainability (SHAP) → Dashboard (Streamlit + pyvis) → Deployment (Docker, offline, Linux).

## Quickstart (once code exists)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. generate synthetic data
python -m src.data_gen.generate --out data/synthetic/transactions.csv

# 2. run full pipeline (graph -> features -> detection -> explain)
python -m src.pipeline --input data/synthetic/transactions.csv --out data/alerts.csv

# 3. launch dashboard
streamlit run src/dashboard/app.py
```

## Docker (offline, Linux target)

```bash
docker build -t bitsentinel -f docker/Dockerfile .
docker run --network none -p 8501:8501 bitsentinel
```

`--network none` proves the offline requirement during the demo.

## Status

Prototype build in progress for SIH 2026. See `ARCHITECTURE.md` §6 for the current build order / sprint plan.
