"""
MITRE ATT&CK TTP System – REST API (Flask)
Serves data to the React/HTML frontend.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import re
import os
import sys

# ── make backend importable ──────────────────
sys.path.insert(0, os.path.dirname(__file__))

from task1_classification import (
    ZeroShotTTPClassifier,
    SupervisedTTPClassifier,
    evaluate_classifiers,
    TACTIC_LABELS,
)
from task2_clustering import (
    build_embeddings,
    kmeans_clustering,
    hierarchical_clustering,
    project_2d,
    cluster_keywords,
    build_actor_profiles,
    CLUSTER_PROFILE_NAMES,
)
from task3_graph import (
    build_kill_chain_graph,
    simulate_intrusion_path,
    probabilistic_walk,
    graph_analytics,
    SCENARIOS,
    TACTIC_DESCRIPTIONS,
    KILL_CHAIN_EDGES,
)
from data_retrieval import get_sample_reports

app = Flask(__name__)
CORS(app)

# ─── Boot: initialise models once ────────────
print("[API] Initialising classifiers...")
zs_clf = ZeroShotTTPClassifier()
sup_clf = SupervisedTTPClassifier()

# Load techniques for supervised training
import pandas as pd
from pathlib import Path

techniques_path = Path("data/techniques_processed.csv")
if techniques_path.exists():
    df = pd.read_csv(techniques_path)
    df = df.dropna(subset=["text_for_nlp", "tactics"])
    df["primary_tactic"] = df["tactics"].apply(lambda x: x.split(",")[0].strip().title())
    df = df[df["primary_tactic"].isin(TACTIC_LABELS)]
    if len(df) > 10:
        sup_clf.fit(df["text_for_nlp"].tolist(), df["primary_tactic"].tolist())
    else:
        print("[API] Not enough data to train supervised model. Zero-shot only.")

reports_df = get_sample_reports()
G_kill_chain = build_kill_chain_graph()
analytics = graph_analytics(G_kill_chain)
print("[API] Ready!")


# ─────────────────────────────────────────────
# Routes – Task 1: Classification
# ─────────────────────────────────────────────

@app.route("/api/classify", methods=["POST"])
def classify():
    """Classify a text snippet using both classifiers."""
    body = request.get_json(force=True)
    text = body.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    zs_result = zs_clf.classify(text, top_k=5)
    sup_result = sup_clf.predict_proba(text, top_k=5) if sup_clf.is_fitted else []

    return jsonify({
        "zero_shot": zs_result,
        "supervised": sup_result,
        "text": text[:300],
    })


@app.route("/api/reports/classify", methods=["GET"])
def classify_reports():
    """Classify all sample threat reports."""
    result = []
    for _, row in reports_df.iterrows():
        text = row["text"]
        zs = zs_clf.classify(text, top_k=3)
        sup = sup_clf.predict_proba(text, top_k=3) if sup_clf.is_fitted else []
        result.append({
            "id": row["id"],
            "title": row["title"],
            "text_preview": text[:200] + "...",
            "zero_shot": zs,
            "supervised": sup,
        })
    return jsonify(result)


@app.route("/api/tactics", methods=["GET"])
def get_tactics():
    return jsonify(TACTIC_LABELS)


# ─────────────────────────────────────────────
# Routes – Task 2: Clustering
# ─────────────────────────────────────────────

@app.route("/api/cluster", methods=["GET"])
def cluster_reports():
    """Cluster sample reports and return profiles + 2D coordinates."""
    texts = reports_df["text"].tolist()
    n = len(texts)
    n_components = min(10, n - 1)
    n_clusters = min(4, n - 1)

    X, vect, _ = build_embeddings(texts, n_components=n_components)
    km = kmeans_clustering(X, n_clusters=n_clusters)
    hac = hierarchical_clustering(X, n_clusters=n_clusters)
    coords = project_2d(X)
    keywords = cluster_keywords(texts, km["labels"], vect)
    profiles = build_actor_profiles(reports_df, km["labels"], hac["labels"], keywords)

    scatter = [
        {
            "id": reports_df.iloc[i]["id"],
            "title": reports_df.iloc[i]["title"],
            "x": float(coords[i][0]),
            "y": float(coords[i][1]),
            "cluster": km["labels"][i],
            "profile": CLUSTER_PROFILE_NAMES.get(km["labels"][i], f"Cluster {km['labels'][i]}"),
        }
        for i in range(n)
    ]

    return jsonify({
        "profiles": profiles,
        "scatter": scatter,
        "km_silhouette": km["silhouette"],
        "hac_silhouette": hac["silhouette"],
        "cluster_names": CLUSTER_PROFILE_NAMES,
        "keywords": {str(k): v for k, v in keywords.items()},
    })


# ─────────────────────────────────────────────
# Routes – Task 3: Attack Graph
# ─────────────────────────────────────────────

@app.route("/api/graph", methods=["GET"])
def get_graph():
    """Return the ATT&CK kill-chain graph as nodes + links JSON."""
    nodes = [
        {
            "id": n,
            "description": TACTIC_DESCRIPTIONS.get(n, ""),
            "pagerank": analytics["pagerank"].get(n, 0),
            "betweenness": analytics["betweenness"].get(n, 0),
        }
        for n in G_kill_chain.nodes()
    ]
    links = [
        {"source": u, "target": v, "weight": round(G_kill_chain[u][v].get("weight", 0.5), 3)}
        for u, v in G_kill_chain.edges()
    ]
    return jsonify({
        "nodes": nodes,
        "links": links,
        "analytics": analytics,
    })


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """Simulate an attack path for a given start/end tactic."""
    body = request.get_json(force=True)
    start = body.get("start", "Reconnaissance")
    end = body.get("end", "Exfiltration")

    paths = simulate_intrusion_path(G_kill_chain, start, end, n_paths=3)
    walk = probabilistic_walk(G_kill_chain, start)

    return jsonify({
        "start": start,
        "end": end,
        "paths": paths,
        "random_walk": walk,
    })


@app.route("/api/scenarios", methods=["GET"])
def get_scenarios():
    """Run all predefined attack scenarios."""
    results = []
    for s in SCENARIOS:
        paths = simulate_intrusion_path(G_kill_chain, s["start"], s["end"], n_paths=3)
        walk = probabilistic_walk(G_kill_chain, s["start"], seed=42)
        results.append({
            "name": s["name"],
            "description": s["description"],
            "start": s["start"],
            "end": s["end"],
            "best_path": paths[0]["path"] if paths else [],
            "best_probability": paths[0]["probability"] if paths else 0,
            "hops": paths[0]["length"] if paths else 0,
            "random_walk": walk,
            "all_paths": paths,
        })
    return jsonify(results)


# ─────────────────────────────────────────────
# Routes – Dashboard Summary
# ─────────────────────────────────────────────

@app.route("/api/summary", methods=["GET"])
def summary():
    return jsonify({
        "total_reports": len(reports_df),
        "total_tactics": len(TACTIC_LABELS),
        "graph_nodes": analytics["num_nodes"],
        "graph_edges": analytics["num_edges"],
        "top_critical": analytics["top_critical_nodes"],
        "supervised_ready": sup_clf.is_fitted,
        "tasks": [
            {"id": 1, "name": "Zero-Shot TTP Classification", "status": "active"},
            {"id": 2, "name": "Threat Actor Profiling", "status": "active"},
            {"id": 3, "name": "Attack Path Simulation", "status": "active"},
        ],
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)