"""
MITRE ATT&CK TTP Extraction System
Task: Data Retrieval & Preprocessing
"""

import json
import re
import requests
import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# 1. Download ATT&CK STIX data from MITRE
# ─────────────────────────────────────────────

ATTACK_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
)

def download_attack_data(save_path: str = "data/enterprise-attack.json") -> dict:
    """Download the ATT&CK STIX bundle from MITRE CTI GitHub."""
    print("[*] Downloading MITRE ATT&CK data...")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    if Path(save_path).exists():
        print(f"[*] Found cached file at {save_path}. Loading...")
        with open(save_path) as f:
            return json.load(f)

    resp = requests.get(ATTACK_URL, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    with open(save_path, "w") as f:
        json.dump(data, f)
    print(f"[+] Saved to {save_path}")
    return data


# ─────────────────────────────────────────────
# 2. Parse STIX objects into structured DataFrames
# ─────────────────────────────────────────────

def parse_attack_stix(stix_bundle: dict) -> dict[str, pd.DataFrame]:
    """Parse STIX 2.0 bundle into Tactics, Techniques, and Groups DataFrames."""
    objects = stix_bundle.get("objects", [])

    tactics, techniques, groups, relationships = [], [], [], []

    for obj in objects:
        obj_type = obj.get("type", "")

        if obj_type == "x-mitre-tactic":
            tactics.append({
                "id": obj.get("external_references", [{}])[0].get("external_id", ""),
                "name": obj.get("name", ""),
                "description": obj.get("description", ""),
                "short_name": obj.get("x_mitre_shortname", ""),
            })

        elif obj_type == "attack-pattern":
            ext_refs = obj.get("external_references", [{}])
            tech_id = ext_refs[0].get("external_id", "") if ext_refs else ""
            platforms = obj.get("x_mitre_platforms", [])
            kill_chain = obj.get("kill_chain_phases", [])
            tactic_names = [kc.get("phase_name", "") for kc in kill_chain]

            techniques.append({
                "id": tech_id,
                "stix_id": obj.get("id", ""),
                "name": obj.get("name", ""),
                "description": obj.get("description", "")[:500],
                "tactics": ", ".join(tactic_names),
                "platforms": ", ".join(platforms),
                "is_subtechnique": obj.get("x_mitre_is_subtechnique", False),
                "detection": obj.get("x_mitre_detection", "")[:300],
                "data_sources": ", ".join(obj.get("x_mitre_data_sources", [])),
            })

        elif obj_type == "intrusion-set":
            groups.append({
                "stix_id": obj.get("id", ""),
                "name": obj.get("name", ""),
                "aliases": ", ".join(obj.get("aliases", [])),
                "description": obj.get("description", "")[:500],
            })

        elif obj_type == "relationship":
            relationships.append({
                "source_ref": obj.get("source_ref", ""),
                "target_ref": obj.get("target_ref", ""),
                "relationship_type": obj.get("relationship_type", ""),
            })

    return {
        "tactics": pd.DataFrame(tactics),
        "techniques": pd.DataFrame(techniques),
        "groups": pd.DataFrame(groups),
        "relationships": pd.DataFrame(relationships),
    }


# ─────────────────────────────────────────────
# 3. Text Preprocessing
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Clean and normalize text for NLP processing."""
    if not isinstance(text, str):
        return ""
    # Remove STIX references and URLs
    text = re.sub(r"\(Citation:[^)]+\)", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)   # wikify links
    text = re.sub(r"[^\w\s.,;:()\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_techniques(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning and feature engineering to the techniques DataFrame."""
    df = df.copy()
    df["clean_description"] = df["description"].apply(clean_text)
    df["clean_detection"] = df["detection"].apply(clean_text)
    df["text_for_nlp"] = df["name"] + ". " + df["clean_description"]
    df = df[df["id"].str.match(r"T\d+", na=False)].reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# 4. Synthetic Threat Reports (for demo / no-CVE env)
# ─────────────────────────────────────────────

SAMPLE_REPORTS = [
    {
        "id": "RPT-001",
        "title": "APT29 Cozy Bear Campaign Analysis",
        "text": (
            "The threat actor used spear-phishing emails with malicious attachments to gain initial "
            "access. After execution, the malware established persistence via scheduled tasks and "
            "registry run keys. Lateral movement was achieved using Pass-the-Hash and remote services. "
            "Data was exfiltrated over encrypted C2 channels using custom backdoors. The group leveraged "
            "PowerShell scripts for discovery and credential dumping using Mimikatz."
        ),
    },
    {
        "id": "RPT-002",
        "title": "Ransomware Intrusion – Financial Sector",
        "text": (
            "Attackers exploited a public-facing RDP vulnerability for initial access. Privilege "
            "escalation was performed via token impersonation. Ransomware was deployed after "
            "disabling antivirus via command-line tools. Shadow copies were deleted using vssadmin. "
            "Network scanning with nmap revealed additional targets inside the environment."
        ),
    },
    {
        "id": "RPT-003",
        "title": "Supply Chain Attack via Software Build System",
        "text": (
            "Adversaries compromised a software build pipeline to inject malicious code into signed "
            "binaries. The trojanized update was distributed to downstream customers. Once executed, "
            "the implant performed DNS tunneling for C2 communication, used DLL side-loading for "
            "defense evasion, and harvested credentials from browser stores."
        ),
    },
    {
        "id": "RPT-004",
        "title": "Insider Threat – Data Exfiltration",
        "text": (
            "An insider abused valid credentials to access sensitive file shares. Large volumes of "
            "data were compressed and uploaded to cloud storage. Logs were cleared to cover tracks. "
            "Keyloggers were installed on shared workstations. Clipboard data was captured to collect "
            "credentials from other users."
        ),
    },
    {
        "id": "RPT-005",
        "title": "OT/ICS Targeted Attack – Energy Sector",
        "text": (
            "Initial compromise via watering-hole attack on an industry forum. The attacker moved "
            "laterally into the OT network using stolen VPN credentials. SCADA systems were targeted "
            "with custom ICS malware. Modbus protocol manipulation was observed. The payload attempted "
            "to disable safety systems and trigger physical damage."
        ),
    },
]


def get_sample_reports() -> pd.DataFrame:
    return pd.DataFrame(SAMPLE_REPORTS)


# ─────────────────────────────────────────────
# 5. Main runner
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Download + parse (uses cache if available)
    bundle = download_attack_data("data/enterprise-attack.json")
    dfs = parse_attack_stix(bundle)

    techniques_df = preprocess_techniques(dfs["techniques"])
    print(f"[+] Techniques loaded: {len(techniques_df)}")
    print(techniques_df[["id", "name", "tactics", "platforms"]].head())

    reports_df = get_sample_reports()
    print(f"\n[+] Sample threat reports: {len(reports_df)}")

    # Save processed data
    Path("data").mkdir(exist_ok=True)
    techniques_df.to_csv("data/techniques_processed.csv", index=False)
    dfs["tactics"].to_csv("data/tactics.csv", index=False)
    dfs["groups"].to_csv("data/groups.csv", index=False)
    reports_df.to_csv("data/sample_reports.csv", index=False)
    print("[+] All processed data saved to data/")