"""
	•	Task: evaluator.py (Self-Evaluation & Metrics)
	•	Compares Groq’s answers to the ground truth in the JSON
	•	Computes accuracy, confidence levels if provided
	•	Flags incorrectly answered prompts and writes them to a separate JSON for review
"""

import json

# Ask user which JSON test file was being used
file = input("Enter the path/name of file that was used to test Groq: ")
file = f"data/{file}"

# Open this file
with open(file) as f:
    data = json.load(f)


# Open the JSON with Groq's answers
with open("data/raw_results.json") as f: # Change this name to whatever Aurora names it
    llm_answers = json.load(f)



# Creates a new json file and writes the incorrectly answered questions and answer choices to it
with open("data/ground_truth.json") as f:
    json.dump(data, f, indent = 4)