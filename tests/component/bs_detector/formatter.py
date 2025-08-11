"""
formatter.py

This module loads SciFact and returns claims in FORAGER-ready format.
"""

import json

def load_and_format_scifact(path="scifact/test.jsonl", limit=None):
    data = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if limit and i>= limit:
                break
            example = json.loads(line)
            if example["label"] == "NOT ENOUGH INFO":
                continue # skip NEI (FORAGER does not handle NEI)
            data.append({
                "claim": example["claim"],
                "evidence": example["evidence"],
                "label": example["label"].upper().replace("SUPPORT", "ENTAILMENT").REPLACE("CONTRADICT", "CONTRADICTION")
            })
    
    return data