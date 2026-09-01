"""Unit tests for the explainability explainer module."""

import pandas as pd
from sklearn.ensemble import IsolationForest

from src.explain.explainer import format_reason, explain_anomalies


def test_format_reason():
    """Tests the format_reason conversion helper for features with SHAP directionality."""
    # 1. Feature increased anomaly score (negative decision SHAP = positive anomaly driver)
    reason_tor = format_reason("tor_broadcast_ratio", 0.85, shap_val=-0.25)
    assert "Tor exit node broadcast ratio" in reason_tor
    assert "elevated anomaly score" in reason_tor
    assert "85.0%" in reason_tor

    reason_vol = format_reason("total_volume_btc", 12.3456, shap_val=-0.50)
    assert "Elevated total transaction volume" in reason_vol
    assert "increased anomaly score" in reason_vol
    assert "12.35 BTC" in reason_vol

    reason_ips = format_reason("unique_ips", 5.0, shap_val=-0.15)
    assert "Elevated unique IP count" in reason_ips
    assert "5" in reason_ips

    # 2. Feature mitigated anomaly score (positive decision SHAP = inlier/normalizing signal)
    reason_mitigated = format_reason("total_volume_btc", 0.25, shap_val=0.30)
    assert "Typical total transaction volume" in reason_mitigated
    assert "mitigated anomaly score" in reason_mitigated


def test_explain_anomalies():
    """Tests explain_anomalies correctly integrates TreeSHAP and attaches reasons."""
    # Simple feature dataset
    data = {
        "degree_centrality": [1.0, 1.0, 10.0],
        "co_spend_cluster_size": [1.0, 1.0, 5.0],
        "unique_ips": [1.0, 1.0, 4.0],
        "tor_broadcast_ratio": [0.0, 0.0, 1.0],
        "ip_switching_frequency": [0.0, 0.0, 0.8],
        "tx_count": [2.0, 3.0, 15.0],
        "total_volume_btc": [0.5, 0.8, 85.0],
        "avg_tx_amount": [0.25, 0.26, 5.66],
        "max_tx_amount": [0.4, 0.5, 20.0],
        "std_tx_amount": [0.1, 0.1, 4.2],
        "burst_ratio": [0.0, 0.0, 0.6],
        "peeling_chain_score": [0.0, 0.0, 0.9],
    }
    wallets = ["wallet0", "wallet1", "wallet2"]
    df = pd.DataFrame(data, index=wallets)

    # Train Isolation Forest
    clf = IsolationForest(contamination=0.3, random_state=42)
    clf.fit(df)

    # Run explanations
    explained_df = explain_anomalies(clf, df, top_n=3)

    # 1. Assert returns DataFrame
    assert isinstance(explained_df, pd.DataFrame)

    # 2. Check top_reasons exists
    assert "top_reasons" in explained_df.columns

    # 3. wallet2 is clear outlier, should have reasons
    reasons_wallet2 = explained_df.loc["wallet2", "top_reasons"]
    assert isinstance(reasons_wallet2, str)
    assert len(reasons_wallet2) > 0

    # Ensure reasons are semicolon-separated list of 3 items
    parts = [p.strip() for p in reasons_wallet2.split(";")]
    assert len(parts) == 3
