# MITRE ATT&CK NLP Threat Intelligence System

A complete, implementation-focused NLP pipeline for automated TTP extraction,
threat actor profiling, and attack path simulation aligned with the MITRE ATT&CK framework.

## Project Structure

```
MITRE_ATT-CK/
│
├── index.html                   ← Frontend dashboard (open directly in browser)
├── api.py                       ← Flask REST API (run this to serve endpoints)
│
├── data_retrieval.py            ← Task 0: STIX download, parsing, preprocessing
├── classification.py            ← Task 1: Zero-shot & supervised TTP classifiers
├── clustering.py                ← Task 2: Embeddings, KMeans, HAC, profiles
└── graph.py                     ← Task 3: Kill-chain graph, path simulation, analytics
```

## Running

```bash
# Step 1 — Install dependencies
pip install -r requirements.txt

# Step 2 — Download and preprocess ATT&CK data
python data_retrieval.py

# Step 3 — Start the Flask API
python api.py

# Step 4 — Open the dashboard (no server needed)
open index.html
```

## Phases Implemented

| Phase | Module | Description |
|-------|--------|-------------|
| 0 | `data_retrieval.py` | Downloads MITRE ATT&CK STIX bundle, parses tactics/techniques/groups, cleans text, extracts IOCs |
| 1 | `classification.py` | Zero-shot keyword classifier + TF-IDF Logistic Regression trained on ATT&CK descriptions |
| 2 | `clustering.py` | TF-IDF + SVD embeddings, KMeans, Hierarchical Agglomerative Clustering, 2D PCA projection |
| 3 | `graph.py` | Directed weighted kill-chain graph, BFS path enumeration, PageRank, Betweenness Centrality |
| 4 | `api.py` | Flask REST API with 9 endpoints serving all NLP and graph modules |
| 5 | `index.html` | Standalone D3.js dashboard — classification, scatter plot, force graph, path simulator |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/summary` | Dashboard statistics |
| GET | `/api/tactics` | List of 14 ATT&CK tactics |
| POST | `/api/classify` | Classify user-supplied text |
| GET | `/api/reports/classify` | Classify all sample reports |
| GET | `/api/cluster` | Clustering results + 2D scatter data |
| GET | `/api/graph` | Kill-chain nodes + edges + PageRank analytics |
| POST | `/api/simulate` | Simulate attack path for given start/end tactic |
| GET | `/api/scenarios` | Run all predefined attack scenarios |

## Classifier Design

**Zero-Shot (Keyword Scoring):**  score(tactic) = keyword_hits / total_keywords,  normalised across 14 tactics

**Supervised (TF-IDF + LR):**  TfidfVectorizer(ngram_range=(1,2), max_features=15000, sublinear_tf=True) → LogisticRegression(C=5, max_iter=1000)

**Technique Matching:**  cosine_similarity(report_tfidf, technique_tfidf) → top-k ATT&CK technique IDs

## Graph Formulas

**PageRank:**  PR(u) = (1−d)/N + d × Σ PR(v)/out(v),  identifies most critical tactic nodes

**Path Probability:**  P(path) = Π w(u,v) for all edges,  BFS enumerates all simple paths up to 10 hops

**Probabilistic Walk:**  successor sampled ∝ edge weight,  Monte Carlo simulation of attacker behaviour

## Key Design Choices

- **Keyword zero-shot** over transformer inference — works fully offline, CPU-only, no GPU required
- **TF-IDF + SVD** over sentence-transformers — no model download, consistent across environments
- **Left-to-right kill chain** with one loop-back edge (Lateral Movement → Execution) to model re-exploitation
- **Heuristic edge weights** (0.5–0.9) derived from ATT&CK technique frequency data, not empirical incidents
- **Baked-in frontend data** — dashboard works standalone without Flask for demos and offline use
- **Exhaustive BFS** for path search (graph is small, 14 nodes), no pruning needed
