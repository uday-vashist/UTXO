"""Unit and integration tests for entity graph builder module."""

import networkx as nx
import pandas as pd

from src.graph.builder import REQUIRED_COLUMNS, build_graph
from src.data_gen.generate import generate


def _create_sample_row(
    txid: str = "tx1",
    timestamp: str = "2026-03-01T00:00:00Z",
    src_ip: str = "73.15.1.1",
    input_addresses: str = "1WalletIn1",
    output_addresses: str = "1WalletOut1;1WalletOut2",
    output_amounts: str = "1.5;0.5",
    amount_btc: float = 2.0,
    fee_btc: float = 0.0001,
    script_type: str = "P2PKH",
    geo_country: str = "US",
    asn: str = "AS7922 Comcast",
    is_tor_exit: bool = False,
) -> dict:
    """Helper to construct a single valid row dictionary."""
    return {
        "timestamp": timestamp,
        "src_ip": src_ip,
        "src_port": 12345,
        "dst_ip": "188.40.1.1",
        "dst_port": 8333,
        "txid": txid,
        "input_addresses": input_addresses,
        "output_addresses": output_addresses,
        "output_amounts": output_amounts,
        "amount_btc": amount_btc,
        "fee_btc": fee_btc,
        "script_type": script_type,
        "geo_country": geo_country,
        "asn": asn,
        "is_tor_exit": is_tor_exit,
    }


def test_build_graph_returns_multidigraph():
    """Test 1: build_graph returns a networkx.MultiDiGraph."""
    df = pd.DataFrame([_create_sample_row()])
    G = build_graph(df)
    assert isinstance(G, nx.MultiDiGraph)


def test_node_types_and_attributes():
    """Test 2: Wallet, IP, and TXID nodes created correctly with required attributes."""
    row = _create_sample_row(
        txid="tx_test_nodes",
        src_ip="73.15.2.2",
        input_addresses="1InAddr",
        output_addresses="1OutAddr",
        output_amounts="1.0",
        amount_btc=1.0,
        fee_btc=0.0002,
        script_type="P2WPKH",
        geo_country="DE",
        asn="AS3320 Deutsche Telekom",
        is_tor_exit=False,
    )
    df = pd.DataFrame([row])
    G = build_graph(df)

    # 1. TXID node
    assert "tx_test_nodes" in G
    tx_node = G.nodes["tx_test_nodes"]
    assert tx_node["node_type"] == "txid"
    assert tx_node["amount_btc"] == 1.0
    assert tx_node["fee_btc"] == 0.0002
    assert tx_node["script_type"] == "P2WPKH"
    assert tx_node["timestamp"] == "2026-03-01T00:00:00Z"

    # 2. IP node
    assert "73.15.2.2" in G
    ip_node = G.nodes["73.15.2.2"]
    assert ip_node["node_type"] == "ip"
    assert ip_node["geo_country"] == "DE"
    assert ip_node["asn"] == "AS3320 Deutsche Telekom"
    assert ip_node["is_tor_exit"] is False

    # 3. Wallet nodes
    assert "1InAddr" in G
    assert G.nodes["1InAddr"]["node_type"] == "wallet"
    assert "1OutAddr" in G
    assert G.nodes["1OutAddr"]["node_type"] == "wallet"


def test_flow_edge_directions_and_pairing():
    """Test 3 & 4: Flow edges have correct directions and output amounts are paired correctly."""
    row = _create_sample_row(
        txid="tx_flow",
        input_addresses="1SenderA",
        output_addresses="1ReceiverB;1ChangeC",
        output_amounts="2.5;0.15",
    )
    df = pd.DataFrame([row])
    G = build_graph(df)

    # Inflow: 1SenderA -> tx_flow
    assert G.has_edge("1SenderA", "tx_flow")
    inflow_data = G.get_edge_data("1SenderA", "tx_flow")[0]
    assert inflow_data["edge_type"] == "flow"
    assert inflow_data["txid"] == "tx_flow"

    # Outflows: tx_flow -> 1ReceiverB and tx_flow -> 1ChangeC
    assert G.has_edge("tx_flow", "1ReceiverB")
    outflow_b = G.get_edge_data("tx_flow", "1ReceiverB")[0]
    assert outflow_b["edge_type"] == "flow"
    assert outflow_b["amount_btc"] == 2.5

    assert G.has_edge("tx_flow", "1ChangeC")
    outflow_c = G.get_edge_data("tx_flow", "1ChangeC")[0]
    assert outflow_c["edge_type"] == "flow"
    assert outflow_c["amount_btc"] == 0.15


def test_co_spend_edges_multi_input():
    """Test 5: Multi-input transactions create pairwise bidirectional co-spend edges."""
    row = _create_sample_row(
        txid="tx_multi_in",
        input_addresses="1Addr1;1Addr2;1Addr3",
        output_addresses="1Out",
        output_amounts="3.0",
    )
    df = pd.DataFrame([row])
    G = build_graph(df)

    pairs = [
        ("1Addr1", "1Addr2"),
        ("1Addr2", "1Addr1"),
        ("1Addr1", "1Addr3"),
        ("1Addr3", "1Addr1"),
        ("1Addr2", "1Addr3"),
        ("1Addr3", "1Addr2"),
    ]
    for u, v in pairs:
        assert G.has_edge(u, v)
        edge_data = G.get_edge_data(u, v)[0]
        assert edge_data["edge_type"] == "co_spend"
        assert edge_data["txid"] == "tx_multi_in"


def test_co_spend_edges_single_input():
    """Test 6: Single-input transactions do not create co-spend edges."""
    row = _create_sample_row(
        txid="tx_single_in",
        input_addresses="1OnlyInput",
        output_addresses="1Out",
        output_amounts="1.0",
    )
    df = pd.DataFrame([row])
    G = build_graph(df)

    co_spend_edges = [
        (u, v)
        for u, v, k, d in G.edges(keys=True, data=True)
        if d.get("edge_type") == "co_spend"
    ]
    assert len(co_spend_edges) == 0


def test_first_broadcast_edges_and_confidence():
    """Test 7, 8, & 9: First-broadcast connects src_ip -> input_wallet with attribution_confidence."""
    # Standard non-Tor broadcast
    row_clean = _create_sample_row(
        txid="tx_clean",
        src_ip="73.15.1.1",
        input_addresses="1CleanWallet",
        output_addresses="1Out",
        output_amounts="1.0",
        is_tor_exit=False,
    )
    # Tor exit node broadcast
    row_tor = _create_sample_row(
        txid="tx_tor",
        src_ip="185.220.101.5",
        input_addresses="1TorWallet",
        output_addresses="1Out",
        output_amounts="1.0",
        is_tor_exit=True,
    )
    df = pd.DataFrame([row_clean, row_tor])
    G = build_graph(df)

    # 1. Clean broadcast edge: 73.15.1.1 -> 1CleanWallet
    assert G.has_edge("73.15.1.1", "1CleanWallet")
    clean_edge = G.get_edge_data("73.15.1.1", "1CleanWallet")[0]
    assert clean_edge["edge_type"] == "first_broadcast"
    assert clean_edge["txid"] == "tx_clean"
    conf_clean = clean_edge["attribution_confidence"]
    assert 0.0 <= conf_clean <= 1.0

    # 2. Tor broadcast edge: 185.220.101.5 -> 1TorWallet
    assert G.has_edge("185.220.101.5", "1TorWallet")
    tor_edge = G.get_edge_data("185.220.101.5", "1TorWallet")[0]
    assert tor_edge["edge_type"] == "first_broadcast"
    assert tor_edge["txid"] == "tx_tor"
    conf_tor = tor_edge["attribution_confidence"]
    assert 0.0 <= conf_tor <= 1.0

    # 3. Tor confidence must be strictly lower than clean confidence
    assert conf_tor < conf_clean


def test_mismatched_output_lengths_raises_value_error():
    """Test 10: Mismatched output_addresses and output_amounts raise ValueError."""
    # 2 addresses but only 1 amount
    row_invalid = _create_sample_row(
        output_addresses="1OutA;1OutB",
        output_amounts="1.5",
    )
    df = pd.DataFrame([row_invalid])
    try:
        build_graph(df)
        assert False, "Expected ValueError for mismatched output addresses and amounts."
    except ValueError as e:
        assert "Count mismatch" in str(e)


def test_missing_required_columns_raises_error():
    """Test 11: Missing required columns raise ValueError."""
    df_missing = pd.DataFrame([{"txid": "tx1", "input_addresses": "1In"}])
    try:
        build_graph(df_missing)
        assert False, "Expected ValueError for missing columns."
    except ValueError as e:
        assert "Missing required columns" in str(e)


def test_integration_with_synthetic_generator():
    """Test 12: Integration test building graph from synthetic data generator."""
    out_csv = "data/synthetic/test_graph_txns.csv"
    gt_csv = "data/ground_truth/test_graph_gt.csv"

    df = generate(
        n_wallets=50,
        n_txns=100,
        illicit_ratio=0.1,
        seed=42,
        out_path=out_csv,
        ground_truth_path=gt_csv,
    )

    G = build_graph(df)
    assert isinstance(G, nx.MultiDiGraph)
    assert G.number_of_nodes() > 0
    assert G.number_of_edges() > 0

    # Verify presence of all three node types
    node_types = {d.get("node_type") for _, d in G.nodes(data=True)}
    assert node_types == {"wallet", "txid", "ip"}

    # Verify presence of all three edge types
    edge_types = {d.get("edge_type") for _, _, _, d in G.edges(keys=True, data=True)}
    assert edge_types == {"flow", "co_spend", "first_broadcast"}

    # Clean up test output files
    import os
    if os.path.exists(out_csv):
        os.remove(out_csv)
def test_coinjoin_skips_co_spend_edges():
    """Test 13: CoinJoin transactions with equal outputs do not create false co-spend edges (B1)."""
    # 3 inputs, 3 equal outputs (0.1 BTC each) plus change
    row = _create_sample_row(
        txid="tx_coinjoin_test",
        input_addresses="1MixerInA;1MixerInB;1MixerInC",
        output_addresses="1MixerOutA;1MixerOutB;1MixerOutC;1ChangeA",
        output_amounts="0.1;0.1;0.1;0.05",
        amount_btc=0.35,
    )
    df = pd.DataFrame([row])
    G = build_graph(df)

    # Assert tx node has is_coinjoin=True
    assert G.nodes["tx_coinjoin_test"]["is_coinjoin"] is True

    # Assert NO co-spend edges exist between mixing inputs
    assert not G.has_edge("1MixerInA", "1MixerInB")
    assert not G.has_edge("1MixerInB", "1MixerInC")
    assert not G.has_edge("1MixerInA", "1MixerInC")


if __name__ == "__main__":
    test_build_graph_returns_multidigraph()
    test_node_types_and_attributes()
    test_flow_edge_directions_and_pairing()
    test_co_spend_edges_multi_input()
    test_co_spend_edges_single_input()
    test_first_broadcast_edges_and_confidence()
    test_mismatched_output_lengths_raises_value_error()
    test_missing_required_columns_raises_error()
    test_integration_with_synthetic_generator()
    print("[OK] All graph builder unit and integration tests passed successfully!")
