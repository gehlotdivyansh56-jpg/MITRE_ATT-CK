"""
Task 2 – Threat Actor Profiling
Clusters threat reports by attack behavior and tools using:
  • TF-IDF sentence embeddings  → KMeans
  • Hierarchical Agglomerative Clustering (HAC)
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from collections import Counter

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# A. Feature Extraction – TF-IDF Embeddings
# ─────────────────────────────────────────────

def build_embeddings(texts: list[str], n_components: int = 50) -> np.ndarray:
    """Convert texts to dense embeddings via TF-IDF + SVD (LSA)."""
    vect = TfidfVectorizer(ngram_range=(1, 2), max_features=10_000, sublinear_tf=True)
    X_sparse = vect.fit_transform(texts)

    svd = TruncatedSVD(n_components=min(n_components, X_sparse.shape[1] - 1), random_state=42)
    X_dense = svd.fit_transform(X_sparse)
    X_norm = normalize(X_dense)

    return X_norm, vect, svd


# ─────────────────────────────────────────────
# B. KMeans Clustering
# ─────────────────────────────────────────────

def kmeans_clustering(X: np.ndarray, n_clusters: int = 4, random_state: int = 42) -> dict:
    """Fit KMeans and return labels + silhouette score."""
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels) if len(set(labels)) > 1 else 0.0
    return {"model": km, "labels": labels.tolist(), "silhouette": round(float(score), 4)}


# ─────────────────────────────────────────────
# C. Hierarchical Clustering
# ─────────────────────────────────────────────

def hierarchical_clustering(X: np.ndarray, n_clusters: int = 4) -> dict:
    """Fit Agglomerative Clustering and return labels + silhouette score."""
    hac = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = hac.fit_predict(X)
    score = silhouette_score(X, labels) if len(set(labels)) > 1 else 0.0
    return {"model": hac, "labels": labels.tolist(), "silhouette": round(float(score), 4)}


# ─────────────────────────────────────────────
# D. 2-D Projection for Visualization
# ─────────────────────────────────────────────

def project_2d(X: np.ndarray) -> np.ndarray:
    """Reduce to 2-D with PCA for scatter plot."""
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X)


# ─────────────────────────────────────────────
# E. Top-N Keywords per Cluster
# ─────────────────────────────────────────────

def cluster_keywords(texts: list[str], labels: list[int], vect: TfidfVectorizer, top_n: int = 8) -> dict:
    """Extract top TF-IDF keywords for each cluster."""
    feature_names = np.array(vect.get_feature_names_out())
    X = vect.transform(texts)
    result = {}
    for cluster_id in sorted(set(labels)):
        mask = np.array(labels) == cluster_id
        cluster_tfidf = np.asarray(X[mask].mean(axis=0)).flatten()
        top_idx = cluster_tfidf.argsort()[::-1][:top_n]
        result[cluster_id] = feature_names[top_idx].tolist()
    return result


# ─────────────────────────────────────────────
# F. Threat Actor Profile Builder
# ─────────────────────────────────────────────

CLUSTER_PROFILE_NAMES = {
    0: "Advanced Persistent Threat (APT)",
    1: "Ransomware / Cybercrime Group",
    2: "Insider Threat Actor",
    3: "State-Sponsored ICS / OT Attacker",
}

def build_actor_profiles(
    reports_df: pd.DataFrame,
    km_labels: list[int],
    hac_labels: list[int],
    keywords: dict,
) -> list[dict]:
    """Combine clustering results into structured threat actor profiles."""
    profiles = []
    for i, (_, row) in enumerate(reports_df.iterrows()):
        cluster = km_labels[i]
        profile_name = CLUSTER_PROFILE_NAMES.get(cluster, f"Cluster-{cluster}")
        profiles.append({
            "report_id": row["id"],
            "title": row["title"],
            "kmeans_cluster": cluster,
            "hac_cluster": hac_labels[i],
            "actor_profile": profile_name,
            "top_keywords": keywords.get(cluster, []),
        })
    return profiles


# ─────────────────────────────────────────────
# G. Optimal K selection (Elbow / Silhouette)
# ─────────────────────────────────────────────

def find_optimal_k(X: np.ndarray, k_range: range = range(2, 8)) -> dict:
    """Compute inertia and silhouette for multiple k values."""
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=15, random_state=42)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else 0.0
        results.append({"k": k, "inertia": round(km.inertia_, 2), "silhouette": round(sil, 4)})
    return results


# ─────────────────────────────────────────────
# H. Main runner
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from data_retrieval import get_sample_reports

    reports = get_sample_reports()
    texts = reports["text"].tolist()

    print("[*] Building TF-IDF + SVD embeddings...")
    X, vect, svd = build_embeddings(texts, n_components=min(10, len(texts) - 1))

    # Find optimal k
    k_analysis = find_optimal_k(X, range(2, min(6, len(texts))))
    print("\n[*] Optimal K analysis:")
    for row in k_analysis:
        print(f"    k={row['k']}  inertia={row['inertia']}  silhouette={row['silhouette']}")

    n_clusters = min(4, len(texts) - 1)

    # KMeans
    km_result = kmeans_clustering(X, n_clusters=n_clusters)
    print(f"\n[+] KMeans Silhouette Score : {km_result['silhouette']}")

    # HAC
    hac_result = hierarchical_clustering(X, n_clusters=n_clusters)
    print(f"[+] HAC    Silhouette Score : {hac_result['silhouette']}")

    # Keywords per cluster
    keywords = cluster_keywords(texts, km_result["labels"], vect)

    # Profiles
    profiles = build_actor_profiles(reports, km_result["labels"], hac_result["labels"], keywords)
    for p in profiles:
        print(f"\n  {p['report_id']} → KMeans={p['kmeans_cluster']} | Profile: {p['actor_profile']}")
        print(f"    Keywords: {p['top_keywords']}")

    # 2D projection for visualization
    coords_2d = project_2d(X)

    # Save results
    output = {
        "profiles": profiles,
        "k_analysis": k_analysis,
        "coords_2d": coords_2d.tolist(),
        "km_silhouette": km_result["silhouette"],
        "hac_silhouette": hac_result["silhouette"],
        "km_labels": km_result["labels"],
        "hac_labels": hac_result["labels"],
    }
    Path("data").mkdir(exist_ok=True)
    with open("data/task2_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n[+] Results saved to data/task2_results.json")