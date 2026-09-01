"""SHAP explainability explainer module for anomaly detection features."""

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest


def format_reason(feature_name: str, val: float, shap_val: float) -> str:
    """Formats a feature name, feature value, and SHAP value into a directionally accurate explanation.

    In TreeSHAP/KernelSHAP for Isolation Forests, lower decision function values correspond
    to higher anomaly scores. Therefore:
      - Negative SHAP value (anomaly_contribution > 0): Feature increased anomaly score.
      - Positive SHAP value (anomaly_contribution < 0): Feature reduced anomaly score (mitigating factor).
    """
    labels = {
        "degree_centrality": "degree centrality",
        "co_spend_cluster_size": "co-spend cluster size",
        "unique_ips": "unique IP count",
        "tor_broadcast_ratio": "Tor exit broadcast ratio",
        "ip_switching_frequency": "IP switching frequency",
        "tx_count": "transaction count",
        "total_volume_btc": "total transaction volume",
        "avg_tx_amount": "average transaction size",
        "max_tx_amount": "maximum transaction size",
        "std_tx_amount": "transaction amount volatility",
        "burst_ratio": "burst transaction ratio",
        "peeling_chain_score": "peeling chain score",
    }
    label = labels.get(feature_name, feature_name.replace("_", " "))
    anomaly_contribution = -shap_val  # Invert: negative decision SHAP = positive anomaly driver

    if anomaly_contribution > 0:
        # Feature drove entity towards anomaly
        if feature_name == "tor_broadcast_ratio":
            return f"Tor exit node broadcast ratio ({val:.1%}) elevated anomaly score"
        elif feature_name in ("total_volume_btc", "max_tx_amount", "avg_tx_amount", "std_tx_amount"):
            return f"Elevated {label} ({val:.2f} BTC) increased anomaly score"
        elif feature_name in ("burst_ratio", "peeling_chain_score"):
            return f"High {label} ({val:.1%}) increased anomaly score"
        elif feature_name == "ip_switching_frequency":
            return f"Rapid IP switching ({val:.2f}/tx) elevated anomaly score"
        elif feature_name in ("co_spend_cluster_size", "unique_ips", "tx_count", "degree_centrality"):
            return f"Elevated {label} ({int(round(val))}) elevated anomaly score"
        else:
            return f"Unusual {label} ({val:.2f}) elevated anomaly score"
    else:
        # Feature mitigated anomaly score (acted as a normal inlier signal)
        if feature_name in ("total_volume_btc", "max_tx_amount", "avg_tx_amount"):
            return f"Typical {label} ({val:.2f} BTC) mitigated anomaly score"
        elif feature_name in ("burst_ratio", "peeling_chain_score", "tor_broadcast_ratio"):
            return f"Low {label} ({val:.1%}) mitigated anomaly score"
        else:
            return f"Standard {label} ({val:.2f}) mitigated anomaly score"


from src.detection.eif import ExtendedIsolationForest


def explain_anomalies(
    model: object,
    feature_df: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """Computes SHAP values and attaches directionally accurate human-readable reasons to each wallet.

    For each entity, it runs either TreeSHAP (for standard models) or KernelSHAP
    (for Extended Isolation Forest models), selects the top_n features contributing
    towards the anomaly score (negative decision SHAP / positive anomaly contribution),
    and constructs a semicolon-separated string of explanations.

    Args:
        model: Trained Isolation Forest or Extended Isolation Forest model.
        feature_df: Feature DataFrame indexed by wallet address (used for training).
        top_n: Number of top contributing features to include in explanations.

    Returns:
        pd.DataFrame: Scored DataFrame with an additional 'top_reasons' column.
    """
    if feature_df.empty:
        df_out = feature_df.copy()
        df_out["top_reasons"] = []
        return df_out

    # Make copy
    df_out = feature_df.copy()

    # Pre-clean inputs for SHAP (drop output metadata columns added during scoring)
    extra_cols = ["anomaly_score", "anomaly_confidence", "attribution_confidence", "attribution_evidence_level", "top_reasons"]
    X = feature_df.drop(columns=[col for col in extra_cols if col in feature_df.columns])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Compute SHAP values based on model type
    if isinstance(model, ExtendedIsolationForest):
        # EIH uses KernelSHAP for model-agnostic explanation.
        # Summarize feature space with 5 representative k-means clusters.
        background = shap.kmeans(X, 5)
        explainer = shap.KernelExplainer(model.decision_function, background)
        
        # Optimization: Only explain the top 100 highest scored anomalies (minimum score of 0.45)
        # to ensure fast execution times under offline demo conditions.
        if "anomaly_score" in feature_df.columns:
            scores_sorted = feature_df["anomaly_score"].sort_values(ascending=False)
            if len(scores_sorted) > 100:
                threshold = max(0.45, scores_sorted.iloc[99])
            else:
                threshold = 0.45
            explain_mask = feature_df["anomaly_score"] >= threshold
        else:
            explain_mask = pd.Series(True, index=feature_df.index)
            
        X_explain = X[explain_mask]
        
        if not X_explain.empty:
            shap_values = explainer.shap_values(X_explain, nsamples=100, silent=True)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            elif len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 0]
        else:
            shap_values = np.zeros((0, X.shape[1]))
    else:
        # Standard Isolation Forest uses native TreeSHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        elif len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 0]
        explain_mask = pd.Series(True, index=feature_df.index)

    reasons = []
    shap_idx = 0
    for idx in range(len(X)):
        if explain_mask.iloc[idx]:
            row_features = X.iloc[idx]
            row_shap = shap_values[shap_idx]
            shap_idx += 1

            # Compute anomaly contributions (invert decision SHAP)
            anomaly_contribs = -row_shap

            # Prioritize features that actively increased the anomaly score (anomaly_contribs > 0)
            # Sort by anomaly contribution descending
            sorted_indices = np.argsort(anomaly_contribs)[::-1]
            
            # Select top_n positive contributors first, or top absolute if none positive
            positive_indices = [i for i in sorted_indices if anomaly_contribs[i] > 0.001]
            if positive_indices:
                chosen_indices = positive_indices[:top_n]
            else:
                chosen_indices = sorted_indices[:top_n]

            row_reasons = []
            for f_idx in chosen_indices:
                feature_name = X.columns[f_idx]
                val = float(row_features.iloc[f_idx])
                shap_v = float(row_shap[f_idx])
                row_reasons.append(format_reason(feature_name, val, shap_v))

            reasons.append("; ".join(row_reasons))
        else:
            reasons.append("Baseline activity consistent with normal ledger behavior")

    df_out["top_reasons"] = reasons
    return df_out
