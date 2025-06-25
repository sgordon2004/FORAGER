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

# Create dictionary to store correct answers from original JSON (answer_key)
# Format: Question # : Correct Answer

correct_answers = {}
answer_key = {}
for idx, question in enumerate(test_questions, 1): # idx is the index of the input-target score pairs (there are 3)
    correct_answers = [k for k, v in question["target_scores"].items() if v == 1]
    # print(f"Q{idx}: {question['input']}")
    # print(f"Correct answer: {correct_answers}\n")
    answer_key[idx] = correct_answers

# Compare LLM response to correct answers
responses = {}

# for idx, question in enumerate(llm_answers, 1): # idx is the index of the input-target score pairs (there are 3)
#     print(question)


i = 1
for k, v in llm_answers.items(): # k = batch, v = dictionary representing batch
    # Right now, each v is a dictionary of all the questions in the batch
    # We have to access the "answer" field in v now
    for key, question_set in v.items():
        for question in question_set:
            responses[i] = question["answer"]
            i += 1

incorrect_questions = {}
# print(answer_key)
# print(responses)
# Go through each question, compare, and add is_correct
i = 1
for entry, value in answer_key.items():
    if value[0] != responses[entry]:
        incorrect_questions[entry] = responses[entry]

# Save updated results
with open("data/incorrect_questions.json", "w") as out_file:
    json.dump(incorrect_questions, out_file, indent=4)