# MITRE ATTACK Threat Intelligence System.

# Architecture
┌─────────────────────────────────────────────────────┐
│                    DATA LAYER                        │
│  MITRE ATT&CK STIX  •  Threat Reports  •  Preproc  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                    NLP LAYER                         │
│  Zero-Shot Classifier  •  TF-IDF+LR  •  Clustering │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                   GRAPH LAYER                        │
│   Kill-Chain DiGraph  •  BFS Paths  •  PageRank     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                    API LAYER                         │
│         Flask REST API  •  9 Endpoints  •  CORS     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                     UI LAYER                         │
│      HTML/JS Dashboard  •  D3.js  •  Canvas 2D      │
└─────────────────────────────────────────────────────┘


# Project Structure

MITRE_ATT&CK/
│
├── data_retrieval.py       # STIX download, parsing, text preprocessing, IOC extraction
├── classification.py       # Task 1: Zero-shot & supervised TTP classifiers
├── clustering.py           # Task 2: TF-IDF embeddings, KMeans, HAC, profiles
├── graph.py                # Task 3: Kill-chain graph, path simulation, analytics
├── api.py                  # Flask REST API — all 9 endpoints
├── index.html              # Interactive D3.js dashboard (standalone)
├── requirements.txt        # Python dependencies

