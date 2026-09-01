"""Feature engineering module for Bitcoin transaction and network telemetry graph."""

from typing import Dict, List
import networkx as nx
import numpy as np
import pandas as pd
from datetime import datetime


def _get_flow_degree(G: nx.MultiDiGraph, node: str) -> int:
    """Calculates strictly the transaction flow degree, ignoring IP broadcast and co-spend edges (Finding B4)."""
    if not G.has_node(node):
        return 0
    flow_in = sum(1 for _, _, d in G.in_edges(node, data=True) if d.get("edge_type") == "flow")
    flow_out = sum(1 for _, _, d in G.out_edges(node, data=True) if d.get("edge_type") == "flow")
    return flow_in + flow_out


def compute_features(G: nx.MultiDiGraph) -> pd.DataFrame:
    """Computes a rich feature set for each wallet node in the MultiDiGraph.

    Args:
        G: The heterogeneous MultiDiGraph built by graph builder.

    Returns:
        pd.DataFrame: A DataFrame indexed by wallet address containing computed features.
    """
    # Extract all wallet nodes
    wallet_nodes = [node for node, data in G.nodes(data=True) if data.get("node_type") == "wallet"]

    features_list = []

    for wallet in wallet_nodes:
        # --- 1. Graph Structural Features ---
        # Degree in general graph
        degree = G.degree(wallet)

        # Co-spend cluster features
        co_spend_neighbors = set()
        # Look at out-edges and in-edges of type 'co_spend'
        if G.has_node(wallet):
            for _, neighbor, edge_data in G.out_edges(wallet, data=True):
                if edge_data.get("edge_type") == "co_spend":
                    co_spend_neighbors.add(neighbor)
            for neighbor, _, edge_data in G.in_edges(wallet, data=True):
                if edge_data.get("edge_type") == "co_spend":
                    co_spend_neighbors.add(neighbor)
        co_spend_cluster_size = len(co_spend_neighbors)

        # --- 2. Network Layer Features ---
        broadcast_ips = []
        is_tor_flags = []
        # Find first_broadcast edges where src_ip -> wallet (this wallet is the input)
        # Note: in builder.py, first_broadcast edge is src_ip -> input_wallet
        if G.has_node(wallet):
            for src_ip, _, edge_data in G.in_edges(wallet, data=True):
                if edge_data.get("edge_type") == "first_broadcast":
                    broadcast_ips.append(src_ip)
                    # Check IP node properties
                    ip_node_data = G.nodes.get(src_ip, {})
                    is_tor_flags.append(bool(ip_node_data.get("is_tor_exit", False)))

        unique_ips = len(set(broadcast_ips))
        total_broadcasts = len(broadcast_ips)
        
        tor_ratio = 0.0
        if total_broadcasts > 0:
            tor_ratio = sum(is_tor_flags) / total_broadcasts

        ip_switching_freq = 0.0
        if total_broadcasts > 1:
            ip_switching_freq = unique_ips / total_broadcasts

        # --- 3. Ledger Behavior Features ---
        tx_amounts = []
        wallet_tx_timestamps = {}
        inflow_count = 0
        outflow_count = 0

        # Outflow edges: txid -> wallet (means this wallet received funds)
        if G.has_node(wallet):
            for txid, _, edge_data in G.in_edges(wallet, data=True):
                if edge_data.get("edge_type") == "flow":
                    outflow_count += 1
                    amt = edge_data.get("amount_btc")
                    if amt is not None:
                        tx_amounts.append(float(amt))
                    ts = edge_data.get("timestamp")
                    if ts and txid not in wallet_tx_timestamps:
                        wallet_tx_timestamps[txid] = ts

            # Inflow edges: wallet -> txid (means this wallet spent funds)
            for _, txid, edge_data in G.out_edges(wallet, data=True):
                if edge_data.get("edge_type") == "flow":
                    inflow_count += 1
                    # Use the wallet's specific/proportional input share (Finding B2)
                    amt = edge_data.get("amount_btc")
                    if amt is None:
                        # Fallback: divide gross TX amount by number of input addresses
                        tx_node_data = G.nodes.get(txid, {})
                        gross_amt = tx_node_data.get("amount_btc")
                        if gross_amt is not None:
                            num_inputs = max(1, sum(1 for _, _, d in G.in_edges(txid, data=True) if d.get("edge_type") == "flow"))
                            amt = float(gross_amt) / num_inputs
                    if amt is not None:
                        tx_amounts.append(float(amt))
                    ts = edge_data.get("timestamp")
                    if ts and txid not in wallet_tx_timestamps:
                        wallet_tx_timestamps[txid] = ts

        tx_count = inflow_count + outflow_count
        total_volume = sum(tx_amounts) if tx_amounts else 0.0
        avg_tx_amount = np.mean(tx_amounts) if tx_amounts else 0.0
        max_tx_amount = np.max(tx_amounts) if tx_amounts else 0.0
        std_tx_amount = np.std(tx_amounts) if tx_amounts else 0.0

        # --- 4. Time/Sequence Features (Burstiness) ---
        # Deduplicate timestamps per unique transaction to eliminate 0.0s multi-output artifacts (Finding S3)
        burst_ratio = 0.0
        tx_timestamps = list(wallet_tx_timestamps.values())
        if len(tx_timestamps) > 1:
            parsed_times = []
            for ts in tx_timestamps:
                try:
                    ts_clean = ts.replace("Z", "").split(".")[0]
                    parsed_times.append(datetime.fromisoformat(ts_clean))
                except Exception:
                    pass
            
            if len(parsed_times) > 1:
                parsed_times.sort()
                intervals = [(parsed_times[i] - parsed_times[i-1]).total_seconds() for i in range(1, len(parsed_times))]
                burst_txs = sum(1 for interval in intervals if 0.0 < interval < 10.0 or interval == 0.0)  # genuine rapid successive txs
                burst_ratio = burst_txs / len(intervals) if intervals else 0.0

        # --- 5. Peeling Chain Indicator ---
        # Strictly check flow degree instead of total graph degree to avoid network IP pollution (Finding B4)
        peeling_count = 0
        total_sent_txs = 0
        
        if G.has_node(wallet):
            for _, txid, edge_data in G.out_edges(wallet, data=True):
                if edge_data.get("edge_type") == "flow":
                    total_sent_txs += 1
                    # Get all outflows of this txid
                    outflows = []
                    for _, out_wallet, flow_data in G.out_edges(txid, data=True):
                        if flow_data.get("edge_type") == "flow":
                            outflows.append(out_wallet)
                    
                    if len(outflows) == 2:
                        deg_out1 = _get_flow_degree(G, outflows[0])
                        deg_out2 = _get_flow_degree(G, outflows[1])
                        # If one of the destination wallets has flow degree <= 2, it's a one-time change output
                        if deg_out1 <= 2 or deg_out2 <= 2:
                            peeling_count += 1
                            
        peeling_chain_score = 0.0
        if total_sent_txs > 0:
            peeling_chain_score = peeling_count / total_sent_txs

        features_list.append({
            "address": wallet,
            "degree_centrality": float(degree),
            "co_spend_cluster_size": float(co_spend_cluster_size),
            "unique_ips": float(unique_ips),
            "tor_broadcast_ratio": float(tor_ratio),
            "ip_switching_frequency": float(ip_switching_freq),
            "tx_count": float(tx_count),
            "total_volume_btc": float(total_volume),
            "avg_tx_amount": float(avg_tx_amount),
            "max_tx_amount": float(max_tx_amount),
            "std_tx_amount": float(std_tx_amount),
            "burst_ratio": float(burst_ratio),
            "peeling_chain_score": float(peeling_chain_score),
        })

    if not features_list:
        return pd.DataFrame(columns=[
            "degree_centrality", "co_spend_cluster_size", "unique_ips",
            "tor_broadcast_ratio", "ip_switching_frequency", "tx_count",
            "total_volume_btc", "avg_tx_amount", "max_tx_amount",
            "std_tx_amount", "burst_ratio", "peeling_chain_score"
        ])

    df_feat = pd.DataFrame(features_list)
    df_feat.set_index("address", inplace=True)
    return df_feat
