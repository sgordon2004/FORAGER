"""
run_test.py

This module serves as the orchestrator of this test. It loads the dataset, runs predictions, and triggers evaluation.
"""

import sys
import os
import sys
import os

# Add the PROJECT root (outer FORAGER folder) to sys.path
project_root = os.path.abspath(os.path.join(__file__, "../../../../"))
sys.path.insert(0, project_root)
from formatter import load_and_format_scifact
from evaluate import evaluate_predictions
from FORAGER.bs import detect_bs
import csv
import json
from FORAGER.embedder import FAISSEmbedder
embedder = FAISSEmbedder.create_default()
embedder.initialize_faiss()

# Load claims
with open("tests/component/bs_detector/scifact/data/claims_dev.jsonl") as f:
    claims = [json.loads(line) for line in f]

# Load corpus
corpus_lookup = {}
with open("tests/component/bs_detector/scifact/data/corpus.jsonl") as f:
    for line in f:
        doc = json.loads(line)
        corpus_lookup[doc["doc_id"]] = doc["abstract"]

results = []
from collections import defaultdict
skipped_claims = 0
gold_label_counts = defaultdict(int)
# Run evaluation
for claim_obj in claims:
    claim = claim_obj["claim"]
    if not claim_obj["evidence"]:
        skipped_claims += 1
        continue
    evidence_dict = claim_obj["evidence"]

    for doc_id, evidences in evidence_dict.items():
        doc_id = int(doc_id)
        if doc_id not in corpus_lookup:
            continue
        abstract = corpus_lookup[doc_id]

        for evidence in evidences:
            sent_indices = evidence["sentences"]
            label = evidence["label"]
            gold_label_counts[label] += 1
            context = [abstract[i] for i in sent_indices if i < len(abstract)]

            prediction = detect_bs(embedder, claim, context)
            results.append({
                "claim": claim,
                "context": " ||| ".join(context),
                "prediction": prediction,
                "gold": label
            })

csv_path = "tests/component/bs_detector/results/predictions.csv"
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

with open(csv_path, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["claim", "context", "prediction", "gold"])
    writer.writeheader()
    writer.writerows(results)
print(f"Saved predictions to {csv_path}")

label_map = {
    "supported": "SUPPORT",
    "unsupported": "UNSUPPORTED",
    "contradicted": "CONTRADICT"
}

preds = [row["prediction"] for row in results]
golds = [row["gold"] for row in results]

normalized_preds = [label_map.get(p.strip().lower(), p.strip().upper()) for p in preds]


from sklearn.metrics import classification_report
print(classification_report(golds, normalized_preds, digits=3))

report = classification_report(golds, normalized_preds, digits=3)
with open("tests/component/bs_detector/results/classification_report.txt", "w") as f:
    f.write(report)
print("Saved classification report to results/classification_report.txt")

from collections import Counter
print("Unique Predictions:", Counter(preds))

unmatched = [p for p in preds if p.strip().lower() not in label_map]
print("Unmatched predictions:", Counter(unmatched))

print("Gold label distribution (evaluated):", dict(gold_label_counts))
print("Number of claims skipped due to no evidence:", skipped_claims)