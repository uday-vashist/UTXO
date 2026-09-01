"""Integration test suite for the end-to-end BitSentinel pipeline runner."""

import os
import pickle
import networkx as nx
import pandas as pd
import pytest
from src.pipeline import run_pipeline


def test_run_pipeline_eif(tmp_path):
    """Tests end-to-end pipeline execution using Extended Isolation Forest."""
    input_csv = "data/synthetic/transactions.csv"
    if not os.path.exists(input_csv):
        pytest.skip(f"Synthetic dataset {input_csv} not found.")

    out_csv = str(tmp_path / "alerts.csv")
    out_pkl = str(tmp_path / "entity_graph.pkl")

    run_pipeline(
        input_path=input_csv,
        out_path=out_csv,
        graph_out_path=out_pkl,
        contamination=0.05,
        model_type="eif",
        seed=42,
    )

    # 1. Assert alerts CSV generated
    assert os.path.exists(out_csv)
    alerts_df = pd.read_csv(out_csv, index_col=0)
    assert not alerts_df.empty
    assert "anomaly_score" in alerts_df.columns
    assert "anomaly_confidence" in alerts_df.columns
    assert "attribution_confidence" in alerts_df.columns
    assert "attribution_evidence_level" in alerts_df.columns
    assert "top_reasons" in alerts_df.columns

    # 2. Assert score ordering
    assert alerts_df["anomaly_score"].is_monotonic_decreasing

    # 3. Assert graph pickle generated
    assert os.path.exists(out_pkl)
    with open(out_pkl, "rb") as f:
        G = pickle.load(f)
    assert isinstance(G, nx.MultiDiGraph)
    assert G.number_of_nodes() > 0


def test_run_pipeline_standard(tmp_path):
    """Tests end-to-end pipeline execution using standard Isolation Forest."""
    input_csv = "data/synthetic/transactions.csv"
    if not os.path.exists(input_csv):
        pytest.skip(f"Synthetic dataset {input_csv} not found.")

    out_csv = str(tmp_path / "alerts_std.csv")
    out_pkl = str(tmp_path / "entity_graph_std.pkl")

    run_pipeline(
        input_path=input_csv,
        out_path=out_csv,
        graph_out_path=out_pkl,
        contamination=0.05,
        model_type="standard",
        seed=42,
    )

    assert os.path.exists(out_csv)
    alerts_df = pd.read_csv(out_csv, index_col=0)
    assert not alerts_df.empty
    assert "anomaly_score" in alerts_df.columns
    assert "top_reasons" in alerts_df.columns
