"""
evaluate.py

This module computes precision, recall, and F1 from predicted vs. gold.
"""

from sklearn.metrics import classification_report

def evaluate_predictions(results):
    y_true = [r["gold_label"] for r in results]
    y_pred = [r["predicted_label"] for r in results]

    print("BS Detector Evaulation Results")
    print(classification_report(y_true, y_pred, digits=3))