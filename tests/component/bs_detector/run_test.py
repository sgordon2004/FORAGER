"""
run_test.py

This module serves as the orchestrator of this test. It loads the dataset, runs predictions, and triggers evaluation.
"""

import sys
import os
parent_dir = os.path.abspath(os.path.join(__file__, "../../../"))
sys.path.append(parent_dir)
from formatter import load_and_format_scifact
from evaluate import evaluate_predictions
from FORAGER.bs import detect_bs
import csv

if __name__ == "__main__":
    test_claims = load_and_format_scifact(limit=60) # returns List[Dict]
    predictions = []

    #TODO: Initialize FAISS embedder so it can be passed to detect_bs()
    embedder = None

    for item in test_claims:
        pred_label = detect_bs(embedder, item["claim"], item["evidence"])
        predictions.append({
            "claim": item["claim"],
            "evidence": item["evidence"],
            "gold_label": item["label"],
            "predicted_label": pred_label
        })

    with open("bs_detector/predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=predictions[0].keys)
        writer.writeheader()
        writer.writerows(predictions)

    evaluate_predictions(predictions)
