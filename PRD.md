# PRD — AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic
**SIH 2026 · Problem Statement ID: SIH26146 · Organization: NTRO · Theme: Blockchain & Cybersecurity**

Codename (working title): **BitSentinel** — rename freely, used only as a project identifier across docs/repo.

---

## 1. Problem Statement (restated)

NTRO needs a system that monitors Bitcoin transaction traffic by fusing two normally-separate data layers to flag potentially illicit activity:

- **Blockchain layer** (public ledger): wallet addresses, TXIDs, amounts, fees, script type.
- **Network layer** (P2P propagation): source/destination IP, port, timestamp of when a transaction was first seen/relayed.

No existing tool (GraphSense, BlockSci, Chainalysis) correlates both layers together — they only analyze the blockchain layer. Our differentiator is doing that fusion.

## 2. Objectives

1. Fuse blockchain-layer and network-layer data into a single **entity graph** (IP ↔ wallet ↔ TXID).
2. Run **unsupervised anomaly/clustering detection** over that graph + transaction features to surface suspicious activity.
3. Produce a **ranked, explainable alert list** with confidence scores (not black-box scores).
4. Provide a **visual dashboard** with interactive link-analysis graph for an analyst to investigate alerts.
5. Work **fully offline**, on **Linux**, with **no live blockchain/internet dependency** during the demo.

## 3. Target User

An NTRO cyber-intel analyst who needs to triage a batch of transaction + network capture data and get a prioritized list of "investigate these first" entities, with reasons attached — not raw ML scores.

## 4. Scope

### In scope (MVP for hackathon)
- Synthetic dataset generator producing realistic blockchain + network-layer data (since no real dataset is provided).
- Batch ingestion of that data (CSV, extensible to JSON/XML).
- Graph construction: wallet–wallet edges (co-spend heuristic), IP–wallet edges (first-broadcast correlation).
- Feature engineering: degree centrality, burst frequency, amount z-score, geo/ASN mismatch, Tor-exit-node flag, peeling-chain / co-spend indicators.
- Isolation Forest anomaly scoring.
- SHAP-based explanation attached to every alert.
- Streamlit dashboard: alert table (ranked, filterable) + pyvis interactive graph view.
- Dockerized, runs on Linux without internet access.

### Explicitly out of scope (state this clearly in the pitch/demo, don't over-promise)
- Real-time / live blockchain or mempool monitoring.
- Definitive identity resolution (KYC linking, on-chain-to-real-world identity).
- Guaranteed "true initiator" detection — this is **probabilistic only** (see §7).
- Training on real labeled illicit-transaction data (none provided; synthetic only).
- Full Tor de-anonymization (we only *flag* Tor/VPN usage as a feature, we don't break it).

## 5. Functional Requirements

| ID | Requirement |
|----|-------------|
| FR1 | System shall generate a configurable-size synthetic dataset with realistic wallet reuse, layering/mixing patterns, and amount distributions. |
| FR2 | System shall ingest CSV (minimum) with schema in §8, validating required fields. |
| FR3 | System shall build a graph with nodes = {wallets, IPs, TXIDs} and typed edges = {co-spend, first-broadcast, input→output}. |
| FR4 | System shall compute per-entity features listed in §6. |
| FR5 | System shall run Isolation Forest (or comparable) to produce an anomaly score per entity/transaction. |
| FR6 | System shall generate a SHAP explanation (top contributing features) per flagged entity. |
| FR7 | System shall output a ranked alert table: entity, anomaly score, confidence, top 3 reasons, linked wallets/IPs. |
| FR8 | Dashboard shall let an analyst filter/sort alerts and click into an interactive graph view centered on a flagged entity. |
| FR9 | Dashboard shall visibly label initiator-attribution confidence as "probable," never "confirmed." |
| FR10 | Entire pipeline shall run via a single Docker command with no external network calls. |

## 6. Feature Set for Anomaly Detection (initial list — refine during build)

- Node degree / centrality in wallet-wallet graph
- Transaction burst frequency (txns/time window)
- Amount z-score vs. wallet's own history
- Peeling-chain pattern flag
- Co-spend cluster size
- Geo/ASN mismatch (claimed vs. GeoIP-derived location)
- Known Tor-exit-node / VPN-range flag
- IP-switching frequency per wallet cluster
- Time-of-day / timing regularity (behavioral fingerprint)

## 7. Key Assumption & Honest Limitation (must appear in pitch + report)

> Identifying the "probable initiator" via IP is **probabilistic, not definitive**. Bitcoin's P2P gossip-relay model means the first-seen IP is not always the true sender, and VPN/Tor usage can hide it entirely. The system always reports a **confidence score**, never a certain identity.

**Mitigation strategy (layered):**
1. Strong network-layer signal → use IP/timing correlation for initiator probability.
2. VPN/Tor hides network layer → fall back to IP-independent blockchain heuristics (co-spend clustering, peeling-chain detection, behavioral fingerprinting).
3. VPN/Tor/IP-switching usage itself becomes a positive anomaly feature.

## 8. Data Requirements (synthetic, since no real dataset given)

Minimum CSV fields:
`timestamp, src_ip, src_port, dst_ip, dst_port, txid, input_addresses, output_addresses, output_amounts, amount_btc, fee_btc, script_type, geo_country, asn, is_tor_exit`

Generator must simulate: normal wallets, mixing/layering wallets, peeling chains, a handful of "ground truth" illicit clusters (so we can self-evaluate precision/recall even without real labels).

## 9. Success Metrics (for our own evaluation, since there's no real ground truth)

- Precision/recall of the pipeline against the **synthetic ground-truth illicit clusters** we ourselves inject.
- % of alerts with a human-readable explanation (target: 100%).
- End-to-end pipeline run time on a moderate synthetic dataset (target: <5 min on a laptop, no GPU).
- Dashboard usability — can a non-technical judge understand an alert in <30 seconds?

## 10. Milestones (see ARCHITECTURE.md §6 for detailed build order)

| Phase | Deliverable |
|-------|-------------|
| M1 | Synthetic data generator producing valid CSV per §8 schema |
| M2 | Graph builder + feature engineering pipeline |
| M3 | Isolation Forest scoring + SHAP explanations |
| M4 | Streamlit + pyvis dashboard wired to pipeline output |
| M5 | Dockerized, offline, end-to-end demo run |
| M6 | Pitch deck + report with limitations section (§7) |

---
*This PRD is the source of truth for scope. Any agent (e.g. in Antigravity) or teammate implementing code should treat §4 (Scope) and §7 (Limitation) as hard constraints — do not silently promise "definitive attribution."*
