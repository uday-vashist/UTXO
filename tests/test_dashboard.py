"""Unit tests for the dashboard visualization and subgraph extraction module."""

import networkx as nx
import pandas as pd
from src.dashboard.graph_viz import generate_subgraph_html, get_ego_subgraph


def test_get_ego_subgraph():
    """Tests ego-network subgraph extraction and filtering."""
    G = nx.MultiDiGraph()

    w1 = "1WalletAlpha"
    w2 = "1WalletBeta"
    w3 = "1WalletGamma"
    ip1 = "73.1.1.1"
    tx1 = "tx_abc123"

    G.add_node(w1, node_type="wallet")
    G.add_node(w2, node_type="wallet")
    G.add_node(w3, node_type="wallet")
    G.add_node(ip1, node_type="ip", is_tor_exit=False)
    G.add_node(tx1, node_type="tx", amount_btc=1.5)

    G.add_edge(w1, tx1, edge_type="flow", amount_btc=1.5)
    G.add_edge(tx1, w2, edge_type="flow", amount_btc=1.5)
    G.add_edge(w1, w3, edge_type="co_spend")
    G.add_edge(ip1, w1, edge_type="first_broadcast", attribution_confidence=0.85)

    # 1. 1-hop neighborhood of w1
    sub_1hop = get_ego_subgraph(G, w1, max_hops=1)
    assert w1 in sub_1hop
    assert tx1 in sub_1hop
    assert w3 in sub_1hop
    assert ip1 in sub_1hop
    assert w2 not in sub_1hop  # w2 is 2 hops away (via tx1)

    # 2. 2-hop neighborhood of w1
    sub_2hop = get_ego_subgraph(G, w1, max_hops=2)
    assert w2 in sub_2hop

    # 3. Edge type filter (only 'flow')
    sub_flow = get_ego_subgraph(G, w1, max_hops=2, allowed_edge_types=["flow"])
    assert w1 in sub_flow
    assert tx1 in sub_flow
    assert w2 in sub_flow
    # No co-spend or first_broadcast edges
    for _, _, d in sub_flow.edges(data=True):
        assert d.get("edge_type") == "flow"

    # 4. Non-existent node handles gracefully
    sub_missing = get_ego_subgraph(G, "NonExistentWallet", max_hops=2)
    assert len(sub_missing) <= 1


def test_generate_subgraph_html():
    """Tests Pyvis HTML generation with node colors and tooltips."""
    G = nx.MultiDiGraph()
    w1 = "1WalletAlpha"
    ip1 = "185.220.101.5"  # Tor IP
    tx1 = "tx_999"

    G.add_node(w1, node_type="wallet")
    G.add_node(ip1, node_type="ip", is_tor_exit=True, geo_country="DE", asn="AS1234")
    G.add_node(tx1, node_type="tx", amount_btc=10.0, timestamp="2026-08-31T12:00:00Z")

    G.add_edge(ip1, w1, edge_type="first_broadcast", attribution_confidence=0.35)
    G.add_edge(w1, tx1, edge_type="flow", amount_btc=10.0)

    alerts_df = pd.DataFrame(
        {
            "anomaly_score": [0.95],
            "anomaly_confidence": ["High"],
            "top_reasons": ["Unusual volume"],
        },
        index=[w1],
    )

    html_content = generate_subgraph_html(G, center_node=w1, alerts_df=alerts_df, height="400px")

    assert isinstance(html_content, str)
    assert len(html_content) > 500
    assert "<html>" in html_content
    assert w1[:6] in html_content  # Label is present
