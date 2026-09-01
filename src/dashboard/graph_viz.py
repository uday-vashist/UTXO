"""Graph visualization module for BitSentinel Streamlit dashboard.

Extracts ego-network subgraphs from NetworkX MultiDiGraph and renders interactive
Pyvis HTML components for deep-dive investigation of anomalous entities.
"""

from typing import Dict, List, Optional, Set
import html
import networkx as nx
import pandas as pd
from pyvis.network import Network


def get_ego_subgraph(
    G: nx.MultiDiGraph,
    center_node: str,
    max_hops: int = 2,
    max_nodes: int = 75,
    allowed_edge_types: Optional[List[str]] = None,
) -> nx.MultiDiGraph:
    """Extracts a localized ego-network subgraph centered around a target entity.

    Args:
        G: Full heterogeneous MultiDiGraph.
        center_node: Target wallet, IP, or TXID node identifier.
        max_hops: BFS search radius (1, 2, or 3).
        max_nodes: Maximum node cap to prevent browser lag.
        allowed_edge_types: Optional filter for edge types ('flow', 'co_spend', 'first_broadcast').

    Returns:
        nx.MultiDiGraph: Localized subgraph containing relevant nodes and edges.
    """
    if center_node not in G:
        subG = nx.MultiDiGraph()
        if center_node:
            subG.add_node(center_node, node_type="wallet")
        return subG

    # BFS traversal using undirected view to capture both incoming and outgoing connections
    visited: Set[str] = {center_node}
    current_level: Set[str] = {center_node}

    for _ in range(max_hops):
        next_level: Set[str] = set()
        for node in current_level:
            # Get all neighbors (predecessors and successors in directed graph)
            neighbors = set(G.predecessors(node)) | set(G.successors(node))
            for nbr in neighbors:
                if nbr not in visited:
                    # Check if connection satisfies edge type filter
                    if allowed_edge_types is not None:
                        # Check edges between node and nbr
                        edges_out = G.get_edge_data(node, nbr, default={})
                        edges_in = G.get_edge_data(nbr, node, default={})
                        edge_types = [d.get("edge_type") for d in list(edges_out.values()) + list(edges_in.values())]
                        if not any(et in allowed_edge_types for et in edge_types):
                            continue
                    next_level.add(nbr)
                    visited.add(nbr)
                    if len(visited) >= max_nodes:
                        break
            if len(visited) >= max_nodes:
                break
        current_level = next_level
        if not current_level or len(visited) >= max_nodes:
            break

    # Build induced subgraph
    subG = G.subgraph(visited).copy()

    # Filter edges in subG if allowed_edge_types is specified
    if allowed_edge_types is not None:
        edges_to_remove = []
        for u, v, k, data in subG.edges(keys=True, data=True):
            if data.get("edge_type") not in allowed_edge_types:
                edges_to_remove.append((u, v, k))
        for u, v, k in edges_to_remove:
            subG.remove_edge(u, v, key=k)

    return subG


def generate_subgraph_html(
    G_sub: nx.MultiDiGraph,
    center_node: Optional[str] = None,
    alerts_df: Optional[pd.DataFrame] = None,
    height: str = "550px",
) -> str:
    """Renders a Pyvis interactive graph visualization as an HTML string.

    Args:
        G_sub: Localized NetworkX subgraph.
        center_node: The focal wallet or entity node.
        alerts_df: Optional DataFrame of alerts to color code wallets by anomaly score.
        height: CSS height of the generated canvas.

    Returns:
        str: Self-contained HTML string with interactive physics, drag-drop, and tooltips.
    """
    net = Network(
        height=height,
        width="100%",
        bgcolor="#0e1117",
        font_color="#f1f5f9",
        directed=True,
    )

    # Disable heavy physics stabilization loops for instant responsiveness
    net.set_options("""
    {
      "nodes": {
        "font": {
          "size": 13,
          "face": "Inter, system-ui, sans-serif",
          "color": "#f8fafc"
        },
        "borderWidth": 2,
        "shadow": true
      },
      "edges": {
        "color": {
          "inherit": false
        },
        "smooth": {
          "type": "continuous",
          "roundness": 0.3
        },
        "shadow": false
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -60,
          "centralGravity": 0.01,
          "springLength": 95,
          "springConstant": 0.08,
          "damping": 0.85,
          "avoidOverlap": 0.7
        },
        "maxVelocity": 40,
        "minVelocity": 0.1,
        "solver": "forceAtlas2Based",
        "timestep": 0.45,
        "stabilization": {
          "enabled": true,
          "iterations": 150
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "zoomView": true,
        "navigationButtons": true
      }
    }
    """)

    # Quick lookup for anomaly scores
    anomaly_scores: Dict[str, float] = {}
    anomaly_conf: Dict[str, str] = {}
    top_reasons: Dict[str, str] = {}
    if alerts_df is not None and not alerts_df.empty:
        for addr, row in alerts_df.iterrows():
            anomaly_scores[str(addr)] = float(row.get("anomaly_score", 0.0))
            anomaly_conf[str(addr)] = str(row.get("anomaly_confidence", "Low"))
            top_reasons[str(addr)] = str(row.get("top_reasons", ""))

    # Add Nodes
    for node, data in G_sub.nodes(data=True):
        node_str = str(node)
        node_type = data.get("node_type", "unknown")
        is_center = node_str == center_node

        # Truncate label for UI cleanliness
        if len(node_str) > 16:
            label = f"{node_str[:6]}...{node_str[-4:]}"
        else:
            label = node_str

        # Node styling defaults
        shape = "dot"
        color = "#94a3b8"
        size = 18
        border_color = "#cbd5e1"

        if node_type == "wallet":
            shape = "hexagon"
            score = anomaly_scores.get(node_str, 0.0)
            conf = anomaly_conf.get(node_str, "Low")
            reasons = top_reasons.get(node_str, "N/A")

            if is_center:
                color = "#ef4444" if score > 0.4 else "#3b82f6"
                border_color = "#fef08a"
                size = 28
                label = f"★ {label}"
            elif conf == "High":
                color = "#ef4444"  # Red
                border_color = "#b91c1c"
                size = 22
            elif conf == "Medium":
                color = "#f59e0b"  # Amber
                border_color = "#d97706"
                size = 18
            else:
                color = "#10b981"  # Green
                border_color = "#059669"
                size = 15

            title_html = (
                f"<b>Wallet:</b> {html.escape(node_str)}<br>"
                f"<b>Anomaly Score:</b> {score:.4f}<br>"
                f"<b>Risk Level:</b> {html.escape(conf)}<br>"
                f"<b>Reasons:</b> {html.escape(reasons[:120])}"
            )

        elif node_type == "ip":
            shape = "dot"
            is_tor = bool(data.get("is_tor_exit", False))
            country = data.get("geo_country", "Unknown")
            asn = data.get("asn", "Unknown")

            if is_tor:
                color = "#dc2626"  # Tor exit alert
                border_color = "#fee2e2"
                size = 20
                label = f"🧅 {label}"
            else:
                color = "#3b82f6"  # Blue
                border_color = "#1d4ed8"
                size = 16

            title_html = (
                f"<b>IP Node:</b> {html.escape(node_str)}<br>"
                f"<b>Tor Exit Node:</b> {'YES (High Risk)' if is_tor else 'No'}<br>"
                f"<b>Country:</b> {html.escape(str(country))}<br>"
                f"<b>ASN:</b> {html.escape(str(asn))}"
            )

        elif node_type == "tx":
            shape = "diamond"
            color = "#64748b"  # Slate
            border_color = "#475569"
            size = 14
            amount = data.get("amount_btc", 0.0)
            timestamp = data.get("timestamp", "Unknown")

            title_html = (
                f"<b>TXID:</b> {html.escape(node_str)}<br>"
                f"<b>Amount:</b> {amount:.4f} BTC<br>"
                f"<b>Timestamp:</b> {html.escape(str(timestamp))}"
            )

        else:
            title_html = f"<b>Entity:</b> {html.escape(node_str)}"

        net.add_node(
            node_str,
            label=label,
            title=title_html,
            shape=shape,
            color={"background": color, "border": border_color, "highlight": {"background": "#38bdf8", "border": "#ffffff"}},
            size=size,
        )

    # Add Edges
    for u, v, data in G_sub.edges(data=True):
        u_str, v_str = str(u), str(v)
        edge_type = data.get("edge_type", "flow")

        if edge_type == "flow":
            color = "#06b6d4"  # Cyan
            dashes = False
            arrows = "to"
            amount = data.get("amount_btc", None)
            edge_label = f"{amount:.2f} BTC" if amount is not None else ""
            title = f"Flow Transfer: {amount} BTC" if amount is not None else "Flow Transfer"

        elif edge_type == "co_spend":
            color = "#a855f7"  # Purple
            dashes = True
            arrows = ""
            edge_label = "co-spend"
            title = "Shared Input Cluster (Co-spend Heuristic)"

        elif edge_type == "first_broadcast":
            color = "#f59e0b"  # Amber
            dashes = False
            arrows = "to"
            conf = data.get("attribution_confidence", 0.0)
            edge_label = f"prop ({conf:.2f})"
            title = f"P2P First Broadcast (Probable Initiator Confidence: {conf:.2%})"

        else:
            color = "#64748b"
            dashes = False
            arrows = "to"
            edge_label = ""
            title = str(edge_type)

        net.add_edge(
            u_str,
            v_str,
            title=title,
            label=edge_label,
            color=color,
            dashes=dashes,
            arrows=arrows,
            width=2 if edge_type == "flow" else 1.5,
        )

    return net.generate_html()
