"""Anomaly detection model module training and scoring wallet entities."""

from typing import Tuple
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def calculate_wallet_attribution_confidence(G: nx.MultiDiGraph, wallet: str) -> float:
    """Computes heuristic attribution evidence confidence based on network telemetry edges.

    In accordance with PRD §7/§9, this metric represents a domain heuristic evidence score,
    not a calibrated frequentist probability.

    Attribution evidence is calculated by:
      1. Gathering all incoming 'first_broadcast' edges (from IP nodes).
      2. If no broadcast telemetry exists (e.g. passive receiver-only wallet), returns np.nan (Finding N3).
      3. Averaging broadcast edge weights (0.85 for normal direct broadcast, 0.35 for Tor exit node).
      4. Applying penalties for high unique IP counts (IP hopping reduces attribution confidence).
      5. Applying consistency boosts if a single IP is consistently observed across broadcasts.

    Args:
        G: The NetworkX MultiDiGraph.
        wallet: The wallet address node ID.

    Returns:
        float: Computed attribution evidence score in [0.10, 0.95], or np.nan if no telemetry exists.
    """
    if not G.has_node(wallet):
        return np.nan

    broadcasts = []
    for ip, _, edge_data in G.in_edges(wallet, data=True):
        if edge_data.get("edge_type") == "first_broadcast":
            broadcasts.append({
                "ip": ip,
                "confidence": edge_data.get("attribution_confidence", 0.85)
            })

    if not broadcasts:
        # Passive recipient wallet: no broadcast telemetry observed (Finding N3)
        return np.nan

    # Baseline: average of broadcast confidence weights
    base_conf = np.mean([b["confidence"] for b in broadcasts])

    # Count unique IPs
    unique_ips = len({b["ip"] for b in broadcasts})
    total_broadcasts = len(broadcasts)

    # Penalty for IP switching (more IPs = harder to attribute true initiator)
    ip_hopping_penalty = 0.0
    if unique_ips > 1:
        ip_hopping_penalty = 0.05 * (unique_ips - 1)

    # Consistency boost if a single IP is consistently broadcasting the wallet's transactions
    consistency_boost = 0.0
    if unique_ips == 1 and total_broadcasts > 1:
        consistency_boost = min(0.10, 0.02 * (total_broadcasts - 1))

    # Calculate final score and clamp to [0.10, 0.95]
    final_conf = base_conf - ip_hopping_penalty + consistency_boost
    return float(np.clip(final_conf, 0.10, 0.95))


def get_attribution_evidence_label(conf: float) -> str:
    """Returns a descriptive forensic label for attribution evidence (PRD §7)."""
    if np.isnan(conf):
        return "No Telemetry (Receiver Only)"
    if conf >= 0.70:
        return "Strong Evidence (Direct IP)"
    if conf >= 0.40:
        return "Moderate Evidence (Multi-IP)"
    return "Low Evidence (Tor/VPN Relay)"


from src.detection.eif import ExtendedIsolationForest


def train_and_score(
    feature_df: pd.DataFrame,
    G: nx.MultiDiGraph,
    contamination: float = 0.05,
    model_type: str = "eif",
    random_seed: int = 42,
) -> Tuple[object, pd.DataFrame]:
    """Trains an Isolation Forest or Extended Isolation Forest model and scores each wallet entity.

    Adds the following columns to the output DataFrame:
      - anomaly_score: Float in [0.0, 1.0] (higher is more anomalous).
      - anomaly_confidence: String ['High', 'Medium', 'Low'].
      - attribution_confidence: Float in [0.0, 1.0].

    Args:
        feature_df: Feature DataFrame indexed by wallet address.
        G: The NetworkX MultiDiGraph containing the network telemetry.
        contamination: Proportion of outliers in the data (IsolationForest parameter).
        model_type: The model algorithm to use ('standard' or 'eif').
        random_seed: Seed for reproducibility.

    Returns:
        Tuple[object, pd.DataFrame]: The trained model (IsolationForest or ExtendedIsolationForest)
                                    and scored DataFrame.
    """
    if feature_df.empty:
        df_out = feature_df.copy()
        df_out["anomaly_score"] = []
        df_out["anomaly_confidence"] = []
        df_out["attribution_confidence"] = []
        return IsolationForest(), df_out

    # Fill NaNs/Infs just in case (features are typically clean, but good practice)
    X = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Train selected model
    if model_type.lower() == "eif":
        clf = ExtendedIsolationForest(
            n_estimators=100,
            random_state=random_seed,
        )
    else:
        clf = IsolationForest(
            contamination=contamination,
            random_state=random_seed,
            n_estimators=100,
        )
    clf.fit(X)

    # Get raw decision scores (lower means more anomalous)
    raw_scores = clf.decision_function(X)

    # Normalize to [0.0, 1.0] where 1.0 is most anomalous and 0.0 is most normal
    min_score = raw_scores.min()
    max_score = raw_scores.max()
    if max_score > min_score:
        anomaly_scores = (max_score - raw_scores) / (max_score - min_score)
    else:
        anomaly_scores = np.zeros_like(raw_scores)

    # Create scored DataFrame
    scored_df = feature_df.copy()
    scored_df["anomaly_score"] = anomaly_scores.astype(float)

    # Categorize anomaly confidence
    def get_confidence_bucket(score: float) -> str:
        if score >= 0.70:
            return "High"
        if score >= 0.45:
            return "Medium"
        return "Low"

    scored_df["anomaly_confidence"] = scored_df["anomaly_score"].apply(get_confidence_bucket)

    # Calculate network layer attribution confidence and evidence level for each wallet
    attribution_confs = []
    attribution_labels = []
    for wallet in scored_df.index:
        conf = calculate_wallet_attribution_confidence(G, wallet)
        attribution_confs.append(conf)
        attribution_labels.append(get_attribution_evidence_label(conf))
    scored_df["attribution_confidence"] = attribution_confs
    scored_df["attribution_evidence_level"] = attribution_labels

    return clf, scored_df
