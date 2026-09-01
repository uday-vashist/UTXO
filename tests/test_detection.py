"""Unit tests for the anomaly detection model module."""

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.detection.model import (
    calculate_wallet_attribution_confidence,
    train_and_score,
)


def test_calculate_wallet_attribution_confidence():
    """Tests the attribution confidence calculation logic for wallets."""
    G = nx.MultiDiGraph()

    wallet_normal = "1NormalWallet"
    wallet_tor = "1TorWallet"
    wallet_mixed = "1MixedWallet"
    wallet_none = "1NoTelemetryWallet"

    G.add_node(wallet_normal, node_type="wallet")
    G.add_node(wallet_tor, node_type="wallet")
    G.add_node(wallet_mixed, node_type="wallet")
    G.add_node(wallet_none, node_type="wallet")

    # IPs
    ip_home = "73.1.1.1"
    ip_tor = "185.220.101.5"

    G.add_node(ip_home, node_type="ip", is_tor_exit=False)
    G.add_node(ip_tor, node_type="ip", is_tor_exit=True)

    # Add edges
    # Normal: 2 normal broadcasts from 1 IP
    G.add_edge(ip_home, wallet_normal, edge_type="first_broadcast", attribution_confidence=0.85)
    G.add_edge(ip_home, wallet_normal, edge_type="first_broadcast", attribution_confidence=0.85)

    # Tor: 1 Tor broadcast
    G.add_edge(ip_tor, wallet_tor, edge_type="first_broadcast", attribution_confidence=0.35)

    # Mixed: 1 normal, 1 Tor (unique IPs)
    G.add_edge(ip_home, wallet_mixed, edge_type="first_broadcast", attribution_confidence=0.85)
    G.add_edge(ip_tor, wallet_mixed, edge_type="first_broadcast", attribution_confidence=0.35)

    from src.detection.model import get_attribution_evidence_label

    # 1. No telemetry (Finding N3: returns np.nan, labeled as receiver only)
    conf_none = calculate_wallet_attribution_confidence(G, wallet_none)
    assert np.isnan(conf_none)
    assert get_attribution_evidence_label(conf_none) == "No Telemetry (Receiver Only)"

    # 2. Normal (expected high confidence due to single IP consistency boost)
    conf_normal = calculate_wallet_attribution_confidence(G, wallet_normal)
    assert conf_normal > 0.85
    assert get_attribution_evidence_label(conf_normal) == "Strong Evidence (Direct IP)"

    # 3. Tor (expected low confidence due to anonymization penalty)
    conf_tor = calculate_wallet_attribution_confidence(G, wallet_tor)
    assert conf_tor == 0.35
    assert get_attribution_evidence_label(conf_tor) == "Low Evidence (Tor/VPN Relay)"

    # 4. Mixed (expected intermediate confidence with IP hopping penalty)
    conf_mixed = calculate_wallet_attribution_confidence(G, wallet_mixed)
    assert np.isclose(conf_mixed, 0.55)
    assert get_attribution_evidence_label(conf_mixed) == "Moderate Evidence (Multi-IP)"


def test_train_and_score():
    """Tests model training, scoring, scaling, and confidence bucketing."""
    # Create fake feature dataset
    np.random.seed(42)
    data = {
        "degree_centrality": [1.0, 1.0, 2.0, 1.0, 10.0],  # node 4 is outlier
        "co_spend_cluster_size": [1.0, 1.0, 1.0, 1.0, 5.0],
        "unique_ips": [1.0, 1.0, 1.0, 1.0, 4.0],
        "tor_broadcast_ratio": [0.0, 0.0, 0.0, 0.0, 1.0],
        "ip_switching_frequency": [0.0, 0.0, 0.0, 0.0, 0.8],
        "tx_count": [2.0, 3.0, 2.0, 2.0, 15.0],
        "total_volume_btc": [0.5, 0.8, 1.2, 0.4, 85.0],
        "avg_tx_amount": [0.25, 0.26, 0.6, 0.2, 5.66],
        "max_tx_amount": [0.4, 0.5, 0.8, 0.3, 20.0],
        "std_tx_amount": [0.1, 0.1, 0.2, 0.1, 4.2],
        "burst_ratio": [0.0, 0.0, 0.0, 0.0, 0.6],
        "peeling_chain_score": [0.0, 0.0, 0.0, 0.0, 0.9],
    }
    wallets = ["wallet0", "wallet1", "wallet2", "wallet3", "wallet4"]
    df = pd.DataFrame(data, index=wallets)

    G = nx.MultiDiGraph()
    for w in wallets:
        G.add_node(w, node_type="wallet")
    # Add normal telemetry for wallet0-3
    ip_normal = "73.1.1.1"
    G.add_node(ip_normal, node_type="ip", is_tor_exit=False)
    for w in wallets[:-1]:
        G.add_edge(ip_normal, w, edge_type="first_broadcast", attribution_confidence=0.85)
    # Add Tor telemetry for wallet4
    ip_tor = "185.220.101.5"
    G.add_node(ip_tor, node_type="ip", is_tor_exit=True)
    G.add_edge(ip_tor, "wallet4", edge_type="first_broadcast", attribution_confidence=0.35)

    clf, scored_df = train_and_score(df, G, contamination=0.2, model_type="standard", random_seed=42)

    # 1. Assert return types
    assert isinstance(clf, IsolationForest)
    assert isinstance(scored_df, pd.DataFrame)

    # 2. Check added columns
    assert "anomaly_score" in scored_df.columns
    assert "anomaly_confidence" in scored_df.columns
    assert "attribution_confidence" in scored_df.columns

    # 3. Check values
    assert scored_df["anomaly_score"].min() == 0.0
    assert scored_df["anomaly_score"].max() == 1.0

    # 4. Outlier (wallet4) should have highest anomaly score
    assert scored_df.loc["wallet4", "anomaly_score"] == 1.0
    assert scored_df.loc["wallet4", "anomaly_confidence"] == "High"

    # 5. Outlier should have lower attribution confidence
    assert scored_df.loc["wallet4", "attribution_confidence"] == 0.35
    assert scored_df.loc["wallet0", "attribution_confidence"] == 0.85


def test_train_and_score_eif():
    """Tests model training and scoring using the custom Extended Isolation Forest."""
    from src.detection.eif import ExtendedIsolationForest

    # Create fake feature dataset
    np.random.seed(42)
    data = {
        "degree_centrality": [1.0, 1.0, 2.0, 1.0, 10.0],  # node 4 is outlier
        "co_spend_cluster_size": [1.0, 1.0, 1.0, 1.0, 5.0],
        "unique_ips": [1.0, 1.0, 1.0, 1.0, 4.0],
        "tor_broadcast_ratio": [0.0, 0.0, 0.0, 0.0, 1.0],
        "ip_switching_frequency": [0.0, 0.0, 0.0, 0.0, 0.8],
        "tx_count": [2.0, 3.0, 2.0, 2.0, 15.0],
        "total_volume_btc": [0.5, 0.8, 1.2, 0.4, 85.0],
        "avg_tx_amount": [0.25, 0.26, 0.6, 0.2, 5.66],
        "max_tx_amount": [0.4, 0.5, 0.8, 0.3, 20.0],
        "std_tx_amount": [0.1, 0.1, 0.2, 0.1, 4.2],
        "burst_ratio": [0.0, 0.0, 0.0, 0.0, 0.6],
        "peeling_chain_score": [0.0, 0.0, 0.0, 0.0, 0.9],
    }
    wallets = ["wallet0", "wallet1", "wallet2", "wallet3", "wallet4"]
    df = pd.DataFrame(data, index=wallets)

    G = nx.MultiDiGraph()
    for w in wallets:
        G.add_node(w, node_type="wallet")
    ip_normal = "73.1.1.1"
    G.add_node(ip_normal, node_type="ip")
    for w in wallets[:-1]:
        G.add_edge(ip_normal, w, edge_type="first_broadcast", attribution_confidence=0.85)

    clf, scored_df = train_and_score(df, G, contamination=0.2, model_type="eif", random_seed=42)

    # 1. Assert return types
    assert isinstance(clf, ExtendedIsolationForest)
    assert isinstance(scored_df, pd.DataFrame)

    # 2. Check added columns
    assert "anomaly_score" in scored_df.columns
    assert "anomaly_confidence" in scored_df.columns
    assert "attribution_confidence" in scored_df.columns

    # 3. Check values
    assert scored_df["anomaly_score"].min() == 0.0
    assert scored_df["anomaly_score"].max() == 1.0

    # 4. Outlier (wallet4) should have highest anomaly score
    assert scored_df.loc["wallet4", "anomaly_score"] == 1.0
    assert scored_df.loc["wallet4", "anomaly_confidence"] == "High"

