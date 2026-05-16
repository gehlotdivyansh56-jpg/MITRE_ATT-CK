"""
Task 1 – Zero-Shot TTP Classification
Uses a transformer zero-shot classifier and compares with a supervised TF-IDF + LR baseline.
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# A. Zero-Shot Classifier (Transformer-based)
# ─────────────────────────────────────────────

TACTIC_LABELS = [
    "Initial Access",
    "Execution",
    "Persistence",
    "Privilege Escalation",
    "Defense Evasion",
    "Credential Access",
    "Discovery",
    "Lateral Movement",
    "Collection",
    "Command and Control",
    "Exfiltration",
    "Impact",
    "Reconnaissance",
    "Resource Development",
]

TACTIC_KEYWORDS = {
    "Initial Access": ["phishing", "exploit", "drive-by", "supply chain", "external", "valid accounts",
                        "replication", "hardware", "trusted relationship", "watering hole"],
    "Execution": ["execute", "powershell", "command", "script", "run", "invoke", "shell", "macro",
                   "wmi", "scheduled task", "api"],
    "Persistence": ["persistence", "registry", "startup", "scheduled task", "service", "boot",
                     "account", "implant", "backdoor"],
    "Privilege Escalation": ["privilege", "escalation", "token", "impersonation", "sudo", "bypass",
                              "administrator", "root", "uac"],
    "Defense Evasion": ["evasion", "obfuscation", "disable", "antivirus", "log", "masquerade",
                         "inject", "rootkit", "signed", "dll sideload"],
    "Credential Access": ["credential", "password", "mimikatz", "dump", "keylog", "hash",
                           "brute force", "kerberos", "lsass"],
    "Discovery": ["discovery", "scan", "enumerate", "network", "account", "system", "nmap",
                   "query", "file", "process"],
    "Lateral Movement": ["lateral", "movement", "pass the hash", "remote", "smb", "rdp",
                          "wmi", "ssh", "psexec"],
    "Collection": ["collect", "clipboard", "capture", "screen", "keylog", "file", "email",
                    "audio", "video"],
    "Command and Control": ["c2", "c&c", "command", "control", "beacon", "dns tunnel",
                             "encrypted", "covert", "backdoor"],
    "Exfiltration": ["exfiltration", "upload", "transfer", "compress", "cloud", "ftp",
                      "dns", "out", "steal", "exfil"],
    "Impact": ["ransomware", "encrypt", "delete", "destroy", "disrupt", "deface",
                "wipe", "disable", "defacement", "shutdown"],
    "Reconnaissance": ["reconnaissance", "gather", "osint", "scan", "phishing for info",
                        "victim", "research"],
    "Resource Development": ["infrastructure", "develop", "acquire", "compromise", "stage",
                              "malware", "tool", "account"],
}


class ZeroShotTTPClassifier:
    """
    Keyword-similarity zero-shot classifier.
    In production, replace score_text with a real HuggingFace pipeline:
        from transformers import pipeline
        self.pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    """

    def __init__(self):
        self.labels = TACTIC_LABELS
        self.keywords = TACTIC_KEYWORDS

    def _score(self, text: str, tactic: str) -> float:
        text_lower = text.lower()
        hits = sum(1 for kw in self.keywords[tactic] if kw in text_lower)
        return hits / max(len(self.keywords[tactic]), 1)

    def classify(self, text: str, top_k: int = 3) -> list[dict]:
        scores = {t: self._score(text, t) for t in self.labels}
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        total = sum(s for _, s in sorted_scores) or 1
        results = [
            {"label": t, "score": round(s / total, 4)}
            for t, s in sorted_scores[:top_k]
        ]
        return results

    def predict_batch(self, texts: list[str], top_k: int = 1) -> list[str]:
        return [self.classify(t, top_k=1)[0]["label"] for t in texts]


# ─────────────────────────────────────────────
# B. Supervised Classifier (TF-IDF + Logistic Regression)
# ─────────────────────────────────────────────

class SupervisedTTPClassifier:
    """TF-IDF + Logistic Regression trained on ATT&CK technique descriptions."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_features=15_000, sublinear_tf=True
        )
        self.clf = LogisticRegression(max_iter=1000, C=5.0)
        self.le = LabelEncoder()
        self.is_fitted = False

    def fit(self, texts: list[str], labels: list[str]) -> None:
        y = self.le.fit_transform(labels)
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, y)
        self.is_fitted = True
        print(f"[+] Supervised model trained on {len(texts)} samples, {len(self.le.classes_)} classes.")

    def predict(self, texts: list[str]) -> list[str]:
        X = self.vectorizer.transform(texts)
        y_pred = self.clf.predict(X)
        return self.le.inverse_transform(y_pred).tolist()

    def predict_proba(self, text: str, top_k: int = 3) -> list[dict]:
        X = self.vectorizer.transform([text])
        proba = self.clf.predict_proba(X)[0]
        top_idx = np.argsort(proba)[::-1][:top_k]
        return [
            {"label": self.le.classes_[i], "score": round(float(proba[i]), 4)}
            for i in top_idx
        ]


# ─────────────────────────────────────────────
# C. Evaluation & Comparison
# ─────────────────────────────────────────────

def evaluate_classifiers(techniques_csv: str = "data/techniques_processed.csv") -> dict:
    """Train supervised model and evaluate both classifiers."""
    df = pd.read_csv(techniques_csv)
    df = df.dropna(subset=["text_for_nlp", "tactics"])
    df = df[df["tactics"].str.strip() != ""]

    # Use first tactic as target label
    df["primary_tactic"] = df["tactics"].apply(lambda x: x.split(",")[0].strip().title())
    df = df[df["primary_tactic"].isin(TACTIC_LABELS)]

    texts = df["text_for_nlp"].tolist()
    labels = df["primary_tactic"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # ---- Supervised ----
    sup_clf = SupervisedTTPClassifier()
    sup_clf.fit(X_train, y_train)
    y_pred_sup = sup_clf.predict(X_test)
    sup_acc = accuracy_score(y_test, y_pred_sup)
    sup_report = classification_report(y_test, y_pred_sup, output_dict=True, zero_division=0)

    # ---- Zero-Shot ----
    zs_clf = ZeroShotTTPClassifier()
    y_pred_zs = zs_clf.predict_batch(X_test)
    zs_acc = accuracy_score(y_test, y_pred_zs)
    zs_report = classification_report(y_test, y_pred_zs, output_dict=True, zero_division=0)

    results = {
        "supervised": {
            "accuracy": round(sup_acc, 4),
            "report": sup_report,
            "model": "TF-IDF + Logistic Regression",
        },
        "zero_shot": {
            "accuracy": round(zs_acc, 4),
            "report": zs_report,
            "model": "Keyword Similarity Zero-Shot",
        },
    }

    print(f"\n{'='*50}")
    print(f"  Supervised  Accuracy : {sup_acc:.2%}")
    print(f"  Zero-Shot   Accuracy : {zs_acc:.2%}")
    print(f"{'='*50}")

    with open("data/task1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[+] Results saved to data/task1_results.json")

    return results, sup_clf, zs_clf


# ─────────────────────────────────────────────
# D. Classify real threat reports
# ─────────────────────────────────────────────

def classify_threat_reports(
    reports_df: pd.DataFrame,
    sup_clf: SupervisedTTPClassifier,
    zs_clf: ZeroShotTTPClassifier,
) -> pd.DataFrame:
    """Apply both classifiers to sample threat reports."""
    records = []
    for _, row in reports_df.iterrows():
        text = row["text"]
        sup_preds = sup_clf.predict_proba(text, top_k=3) if sup_clf.is_fitted else []
        zs_preds = zs_clf.classify(text, top_k=3)
        records.append({
            "report_id": row["id"],
            "title": row["title"],
            "supervised_top3": sup_preds,
            "zeroshot_top3": zs_preds,
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    from data_retrieval import get_sample_reports

    results, sup_clf, zs_clf = evaluate_classifiers("data/techniques_processed.csv")

    reports = get_sample_reports()
    classified = classify_threat_reports(reports, sup_clf, zs_clf)
    print("\n[+] Threat Report Classifications:")
    for _, row in classified.iterrows():
        print(f"\n  {row['report_id']}: {row['title']}")
        print(f"    ZeroShot  → {row['zeroshot_top3']}")
        print(f"    Supervised→ {row['supervised_top3']}")