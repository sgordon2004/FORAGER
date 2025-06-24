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
    test_questions = json.load(f)

# Open the JSON with Groq's answers
with open("data/llm_responses.json") as f: # Change this name to whatever Aurora names it
    llm_answers = json.load(f)

# Create a dictionary from LLM's answers for quick lookup by ID
response_dict = {item[id]: item["llm_response"] for item in llm_answers}

incorrect_questions = []

# Go through each question, compare, and add is_correct
for q in test_questions:
    qid = q[id]
    correct_answer = next(k for k, v in q["target_scores"].items() if v == 1)
    llm_response = response_dict.get(qid, "").strip().lower()
    q["llm_response"] = llm_response
    q["is_correct"] = int(llm_response == correct_answer.strip().lower())
    if (q["is_correct"] == 0):
        incorrect_questions.append(q)

# Creates a new json file and writes the incorrectly answered questions and answer choices to it
with open("incorrect_questions.json", "w") as out_file:
    json.dump(incorrect_questions, out_file, indent=4)