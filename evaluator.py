"""
	•	Task: evaluator.py (Self-Evaluation & Metrics)
	•	Compares Groq’s answers to the ground truth in the JSON
	•	Computes accuracy, confidence levels if provided
	•	Flags incorrectly answered prompts and writes them to a separate JSON for review
"""

import formatter
import json

with open("raw_results.json") as f: # These are Groq's answers - change this to whatever Aurora names it
    raw_results = json.load(f)

for i in range(0,5):
    print(formatter.format_json())

# data = {#incorrectly answered questions here}

# Creates a new json file and writes the incorrectly answered questions and answer choices to it
# with open("data/ground_truth.json") as f:
#     json.dump(data, f, indent = 4)