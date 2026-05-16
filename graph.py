"""
Task 3 – Graph-Based Attack Path Simulation
Builds a directed attack chain graph and simulates multi-stage intrusion paths
using NetworkX and shortest-path / probabilistic traversal algorithms.
"""

import json
import random
import itertools
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from collections import defaultdict


# ─────────────────────────────────────────────
# A. ATT&CK Tactic-Level Kill-Chain Graph
# ─────────────────────────────────────────────

KILL_CHAIN_EDGES = [
    # (source_tactic, target_tactic, weight)
    ("Reconnaissance",       "Resource Development", 0.9),
    ("Resource Development", "Initial Access",       0.85),
    ("Initial Access",       "Execution",            0.9),
    ("Execution",            "Persistence",          0.8),
    ("Execution",            "Privilege Escalation", 0.75),
    ("Persistence",          "Defense Evasion",      0.7),
    ("Privilege Escalation", "Defense Evasion",      0.8),
    ("Defense Evasion",      "Credential Access",    0.75),
    ("Credential Access",    "Discovery",            0.85),
    ("Discovery",            "Lateral Movement",     0.8),
    ("Lateral Movement",     "Collection",           0.75),
    ("Lateral Movement",     "Execution",            0.5),  # loop-back
    ("Collection",           "Command and Control",  0.85),
    ("Command and Control",  "Exfiltration",         0.8),
    ("Command and Control",  "Impact",               0.6),
    ("Exfiltration",         "Impact",               0.5),
]

TACTIC_DESCRIPTIONS = {
    "Reconnaissance":       "Gather information about the target environment",
    "Resource Development": "Acquire infrastructure, accounts, and tooling",
    "Initial Access":       "Gain a foothold via phishing, exploits, or valid accounts",
    "Execution":            "Run malicious code on the target system",
    "Persistence":          "Maintain access across reboots and credential changes",
    "Privilege Escalation": "Gain higher-level permissions on the system",
    "Defense Evasion":      "Avoid detection by security tools",
    "Credential Access":    "Steal account credentials and hashes",
    "Discovery":            "Enumerate the environment (hosts, users, shares)",
    "Lateral Movement":     "Move through the network to additional systems",
    "Collection":           "Gather data of interest for exfiltration",
    "Command and Control":  "Communicate with compromised systems over C2 channels",
    "Exfiltration":         "Transfer stolen data to adversary-controlled infrastructure",
    "Impact":               "Disrupt, destroy, or manipulate business operations",
}


def build_kill_chain_graph() -> nx.DiGraph:
    """Build the tactic-level ATT&CK kill-chain directed graph."""
    G = nx.DiGraph()
    for tactic, desc in TACTIC_DESCRIPTIONS.items():
        G.add_node(tactic, description=desc, type="tactic")

    for src, dst, w in KILL_CHAIN_EDGES:
        G.add_edge(src, dst, weight=w, probability=w)

    return G


# ─────────────────────────────────────────────
# B. Technique-Level Subgraph Builder
# ─────────────────────────────────────────────

TECHNIQUE_EXAMPLES = {
    "Initial Access":       [("T1566", "Phishing"), ("T1190", "Exploit Public-Facing App"), ("T1078", "Valid Accounts")],
    "Execution":            [("T1059", "Command & Scripting Interpreter"), ("T1204", "User Execution"), ("T1053", "Scheduled Task")],
    "Persistence":          [("T1547", "Boot/Logon Autostart"), ("T1543", "Create/Modify System Process"), ("T1136", "Create Account")],
    "Privilege Escalation": [("T1068", "Exploitation for Privilege Escalation"), ("T1055", "Process Injection"), ("T1134", "Access Token Manipulation")],
    "Defense Evasion":      [("T1036", "Masquerading"), ("T1562", "Impair Defenses"), ("T1027", "Obfuscated Files")],
    "Credential Access":    [("T1003", "OS Credential Dumping"), ("T1110", "Brute Force"), ("T1555", "Credentials from Password Stores")],
    "Discovery":            [("T1082", "System Information Discovery"), ("T1046", "Network Service Discovery"), ("T1033", "System Owner/User Discovery")],
    "Lateral Movement":     [("T1021", "Remote Services"), ("T1550", "Use Alternate Auth Material"), ("T1570", "Lateral Tool Transfer")],
    "Collection":           [("T1005", "Data from Local System"), ("T1056", "Input Capture"), ("T1560", "Archive Collected Data")],
    "Command and Control":  [("T1071", "Application Layer Protocol"), ("T1095", "Non-Application Layer Protocol"), ("T1572", "Protocol Tunneling")],
    "Exfiltration":         [("T1048", "Exfiltration Over Alt Protocol"), ("T1041", "Exfiltration Over C2 Channel"), ("T1567", "Exfiltration Over Web Service")],
    "Impact":               [("T1486", "Data Encrypted for Impact"), ("T1485", "Data Destruction"), ("T1489", "Service Stop")],
}


def build_technique_graph(G_tactic: nx.DiGraph) -> nx.DiGraph:
    """Expand tactic nodes to individual technique nodes."""
    G_tech = nx.DiGraph()

    for tactic, techniques in TECHNIQUE_EXAMPLES.items():
        for tech_id, tech_name in techniques:
            node_id = f"{tech_id}::{tech_name}"
            G_tech.add_node(node_id, technique_id=tech_id, technique_name=tech_name, tactic=tactic)

    # Add intra-tactic edges (techniques within same tactic can chain)
    for tactic, techniques in TECHNIQUE_EXAMPLES.items():
        for i, (src_id, src_name) in enumerate(techniques):
            for j, (dst_id, dst_name) in enumerate(techniques):
                if i != j:
                    G_tech.add_edge(
                        f"{src_id}::{src_name}",
                        f"{dst_id}::{dst_name}",
                        weight=0.4,
                        edge_type="intra_tactic",
                    )

    # Add inter-tactic edges following kill chain
    for src_tactic, dst_tactic, prob in KILL_CHAIN_EDGES:
        src_techniques = TECHNIQUE_EXAMPLES.get(src_tactic, [])
        dst_techniques = TECHNIQUE_EXAMPLES.get(dst_tactic, [])
        for (s_id, s_name) in src_techniques:
            for (d_id, d_name) in dst_techniques:
                G_tech.add_edge(
                    f"{s_id}::{s_name}",
                    f"{d_id}::{d_name}",
                    weight=prob,
                    edge_type="cross_tactic",
                )

    return G_tech


# ─────────────────────────────────────────────
# C. Attack Path Simulation
# ─────────────────────────────────────────────

def simulate_intrusion_path(
    G: nx.DiGraph,
    start_node: str,
    end_node: str,
    n_paths: int = 5,
) -> list[dict]:
    """Find multiple simple attack paths sorted by cumulative probability."""
    all_paths = []
    try:
        simple_paths = list(
            itertools.islice(nx.all_simple_paths(G, source=start_node, target=end_node, cutoff=10), 100)
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []

    for path in simple_paths:
        prob = 1.0
        for u, v in zip(path[:-1], path[1:]):
            edge_data = G.get_edge_data(u, v, default={})
            prob *= edge_data.get("weight", 0.5)
        all_paths.append({"path": path, "probability": round(prob, 6), "length": len(path)})

    all_paths.sort(key=lambda x: x["probability"], reverse=True)
    return all_paths[:n_paths]


def probabilistic_walk(
    G: nx.DiGraph,
    start: str,
    max_steps: int = 12,
    stop_probability: float = 0.15,
    seed: int = 42,
) -> list[str]:
    """
    Simulate a random attacker walk through the graph
    following edge weights as transition probabilities.
    """
    random.seed(seed)
    path = [start]
    current = start

    for _ in range(max_steps):
        if random.random() < stop_probability:
            break
        neighbors = list(G.successors(current))
        if not neighbors:
            break
        weights = [G[current][n].get("weight", 0.5) for n in neighbors]
        total = sum(weights)
        probs = [w / total for w in weights]
        current = random.choices(neighbors, weights=probs, k=1)[0]
        if current not in path:
            path.append(current)

    return path


# ─────────────────────────────────────────────
# D. Graph Analytics
# ─────────────────────────────────────────────

def graph_analytics(G: nx.DiGraph) -> dict:
    """Compute key graph metrics for the ATT&CK kill-chain."""
    pagerank = nx.pagerank(G, weight="weight")
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    betweenness = nx.betweenness_centrality(G, weight="weight")

    analytics = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": round(nx.density(G), 4),
        "pagerank": {k: round(v, 4) for k, v in sorted(pagerank.items(), key=lambda x: -x[1])},
        "betweenness": {k: round(v, 4) for k, v in sorted(betweenness.items(), key=lambda x: -x[1])},
        "top_critical_nodes": sorted(pagerank, key=pagerank.get, reverse=True)[:5],
    }
    return analytics


# ─────────────────────────────────────────────
# E. Scenario Runner
# ─────────────────────────────────────────────

SCENARIOS = [
    {
        "name": "APT Espionage Campaign",
        "start": "Reconnaissance",
        "end": "Exfiltration",
        "description": "Nation-state actor targeting sensitive government data",
    },
    {
        "name": "Ransomware Attack",
        "start": "Initial Access",
        "end": "Impact",
        "description": "Financially motivated group deploying ransomware",
    },
    {
        "name": "Insider Threat",
        "start": "Execution",
        "end": "Exfiltration",
        "description": "Malicious insider with valid credentials",
    },
]


def run_scenarios(G: nx.DiGraph) -> list[dict]:
    """Run all predefined attack scenarios through the graph."""
    results = []
    for scenario in SCENARIOS:
        paths = simulate_intrusion_path(G, scenario["start"], scenario["end"])
        walk = probabilistic_walk(G, scenario["start"])
        results.append({
            "scenario": scenario,
            "top_paths": paths,
            "random_walk": walk,
        })
        print(f"\n[Scenario] {scenario['name']}")
        if paths:
            print(f"  Best path ({len(paths[0]['path'])} hops, p={paths[0]['probability']}):")
            print("    " + " → ".join(paths[0]["path"]))
        print(f"  Random walk: {' → '.join(walk)}")
    return results


# ─────────────────────────────────────────────
# F. Export graph as JSON (for D3 / frontend)
# ─────────────────────────────────────────────

def export_graph_json(G: nx.DiGraph, filepath: str = "data/attack_graph.json") -> None:
    """Export NetworkX graph to JSON for frontend D3 rendering."""
    nodes = [
        {
            "id": n,
            "description": G.nodes[n].get("description", ""),
            "type": G.nodes[n].get("type", "tactic"),
        }
        for n in G.nodes()
    ]
    links = [
        {
            "source": u,
            "target": v,
            "weight": round(G[u][v].get("weight", 0.5), 3),
        }
        for u, v in G.edges()
    ]
    with open(filepath, "w") as f:
        json.dump({"nodes": nodes, "links": links}, f, indent=2)
    print(f"[+] Graph exported to {filepath}")


# ─────────────────────────────────────────────
# G. Main runner
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("[*] Building kill-chain graph...")
    G = build_kill_chain_graph()
    analytics = graph_analytics(G)

    print(f"\n[+] Graph: {analytics['num_nodes']} nodes, {analytics['num_edges']} edges")
    print(f"    Density: {analytics['density']}")
    print(f"    Top critical nodes (PageRank): {analytics['top_critical_nodes']}")

    scenario_results = run_scenarios(G)

    Path("data").mkdir(exist_ok=True)
    export_graph_json(G)

    output = {
        "analytics": analytics,
        "scenarios": scenario_results,
    }
    with open("data/task3_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\n[+] Task 3 results saved to data/task3_results.json")