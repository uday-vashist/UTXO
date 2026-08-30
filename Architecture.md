# Architecture — UTXO (SIH26146)


## 1. High-Level Data Flow

```
[Synthetic Data Generator]
        |
        v
   data/synthetic/*.csv  (blockchain + network fields, unified rows)
        |
        v
[Ingestion Layer] --------> validates schema, loads into pandas DataFrame
        |
        v
[Graph Builder] -----------> NetworkX MultiDiGraph
        |   nodes: wallet, ip, txid
        |   edges: co-spend (wallet-wallet), first-broadcast (ip-wallet),
        |          input->output (wallet-wallet via txid)
        v
[Feature Engineering] -----> per-node/per-txn feature table
        |   degree centrality, burst frequency, amount z-score,
        |   geo/ASN mismatch, tor-exit flag, peeling-chain flag, co-spend size
        v
[Anomaly Detection] -------> scikit-learn IsolationForest -> anomaly_score
        |
        v
[Explainability] -----------> SHAP -> top-3 contributing features per alert
        |
        v
[Alert Ranking] ------------> sorted table: entity, score, confidence, reasons, links
        |
        v
[Dashboard: Streamlit + pyvis] --> analyst-facing UI
```

Everything runs as a batch pipeline — no live network calls, fully reproducible on Linux inside Docker.

## 2. Component Breakdown

### 2.1 Data Generator (`src/data_gen/`)
- Uses `Faker` for identities/IPs, custom logic for Bitcoin-shaped data.
- Simulates: normal user wallets, mixing services (many-in/many-out), peeling chains (one large input -> small change repeatedly), a small set of injected "ground-truth illicit" clusters for self-evaluation.
- Also emits a `geoip_lookup.csv` (or static mapping) so we don't need live GeoIP API calls — fully offline.
- Also emits a static `tor_exit_nodes.csv` (snapshot list, bundled — not fetched live).
- Output: single unified CSV per §4 schema below.

### 2.2 Ingestion (`src/ingestion/`)
- Reads CSV (extensible to JSON/XML later) via pandas.
- Schema validation (required columns, types, no nulls in key fields).
- Normalizes timestamps to UTC.

### 2.3 Graph Builder (`src/graph/`)
- Builds a `networkx.MultiDiGraph`.
- Edge type `co-spend`: two input addresses appearing together in the same TXID's inputs → wallet-wallet edge (classic clustering heuristic).
- Edge type `first-broadcast`: the (src_ip, dst_ip) pair with the earliest timestamp seen for a given TXID → ip-wallet edge, weighted by confidence (see §3).
- Edge type `flow`: input address(es) → output address(es) for a TXID.

### 2.4 Feature Engineering (`src/detection/features.py`)
Computed per wallet-node and per transaction (join both into one feature table before scoring):
- degree / centrality in wallet-wallet subgraph
- txn burst frequency (rolling window count)
- amount z-score vs. that wallet's historical amounts
- co-spend cluster size
- peeling-chain flag (heuristic: repeated small-change outputs)
- geo/ASN mismatch flag
- tor-exit flag
- ip-switching frequency for the wallet's associated IPs

### 2.5 Anomaly Detection (`src/detection/model.py`)
- `sklearn.ensemble.IsolationForest` or **Extended Isolation Forest (EIF)**, unsupervised, contamination as a tunable param. EIF is preferred to avoid axis-parallel artifacts.
- Output: `anomaly_score` per entity, normalized to 0–1, plus a bucketed `confidence` (Low/Med/High) derived from score distribution.

### 2.6 Explainability (`src/explain/`)
- **TreeSHAP** via `shap.TreeExplainer` (works natively with tree-based models like IsolationForest and EIF).
- For each flagged entity: top 3 features by |SHAP value| → converted to a human-readable reason string, e.g. *"Unusual amount pattern (3.2σ from wallet norm) + Tor-exit IP + high co-spend cluster size."*

### 2.7 Dashboard (`src/dashboard/app.py`)
- **Streamlit** for layout: alert table (sortable/filterable by score, confidence, date range), detail panel per alert.
- **pyvis** for the interactive graph — click an alert → renders its local subgraph (wallet ↔ IP ↔ TXID neighborhood) embedded in the Streamlit page via `st.components.v1.html`.
- Must clearly display "Probable initiator (confidence: X%)" — never assert certainty (ties back to PRD §7/§9).

## 3. Confidence Scoring Logic (important — don't skip)

Two distinct confidence numbers, don't conflate them:
1. **Anomaly confidence** — how unusual is this entity (from Isolation Forest score distribution).
2. **Attribution confidence** — how sure are we the flagged IP is the true initiator (from §2.3's first-broadcast weighting: penalize if Tor/VPN-flagged, penalize if multiple IPs broadcast the same TXID within a small time delta, boost if a single IP is consistently first-seen across that wallet's history).

Both should show up separately in the alert detail view.

## 4. Data Schema (synthetic CSV)

| Column | Type | Notes |
|---|---|---|
| timestamp | ISO8601 | UTC |
| src_ip | string | IPv4 |
| src_port | int | |
| dst_ip | string | IPv4 |
| dst_port | int | |
| txid | string | hex, unique per transaction |
| input_addresses | string | semicolon-separated list |
| output_addresses | string | semicolon-separated list |
| output_amounts | string | semicolon-separated list of float values matching output_addresses 1-to-1 |
| amount_btc | float | total transaction amount (sum of outputs or primary transfer amount) |
| fee_btc | float | |
| script_type | string | e.g. P2PKH, P2WPKH, P2SH |
| geo_country | string | from static GeoIP mapping |
| asn | string | from static GeoIP mapping |
| is_tor_exit | bool | from bundled static Tor exit-node list |

## 5. Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Synthetic data | Python + Faker | No GPU, fast, realistic-enough identifiers |
| Ingestion | pandas / json / xml.etree | standard, extensible |
| Entity graph | NetworkX | pure-Python, no server dependency, fine at hackathon scale |
| Anomaly detection | scikit-learn (IsolationForest) / EIF | unsupervised, no labels needed, fast, explainable via SHAP |
| Explainability | SHAP (TreeSHAP) | exact Shapley values for tree models, fast, judge-friendly |
| Dashboard | Streamlit + pyvis | fastest way to a polished interactive demo |
| Deployment | Docker | guarantees "offline + Linux" requirement is met |

## 6. Build Order / Sprint Plan

1. **Day 1:** Repo skeleton + PRD/architecture finalized + synthetic data generator (M1). Everyone can start once CSV schema is fixed.
2. **Day 2:** Graph builder + feature engineering (M2), in parallel with dashboard skeleton (static/dummy data first).
3. **Day 3:** IsolationForest + SHAP wired into pipeline (M3); dashboard connects to real pipeline output.
4. **Day 4:** Polish dashboard, tune contamination/thresholds, inject ground-truth illicit clusters and measure precision/recall (M5/§9 metrics).
5. **Day 5:** Dockerize, test fully offline on Linux, freeze demo dataset, rehearse pitch with limitation section (PRD §7).

## 7. Repo Structure

```
sih26146-bitcoin-monitor/
├── PRD.md
├── ARCHITECTURE.md
├── README.md
├── requirements.txt
├── data/
│   ├── synthetic/          # generated CSVs, geoip + tor lists
│   └── ground_truth/       # injected illicit cluster labels (for our own eval)
├── src/
│   ├── data_gen/
│   ├── ingestion/
│   ├── graph/
│   ├── detection/
│   │   ├── features.py
│   │   └── model.py
│   ├── explain/
│   └── dashboard/
│       └── app.py
├── notebooks/               # exploration only, not shipped logic
├── docker/
│   └── Dockerfile
└── tests/
```

## 8. Module Interfaces (contracts between teammates)

- `data_gen.generate(n_wallets, n_txns, illicit_ratio, seed) -> pd.DataFrame` — saves CSV, returns DataFrame + ground-truth labels DataFrame.
- `graph.build_graph(df: pd.DataFrame) -> networkx.MultiDiGraph`
- `detection.features.compute(graph, df) -> pd.DataFrame` (one row per entity)
- `detection.model.score(feature_df) -> pd.DataFrame` (adds `anomaly_score`, `confidence`)
- `explain.explain(model, feature_df) -> pd.DataFrame` (adds `top_reasons: List[str]`)
- `dashboard.app` consumes the final ranked alert DataFrame + graph object only — no re-computation in the UI layer.

Keep these signatures stable — that's what lets four people build in parallel without merge hell.
