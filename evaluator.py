"""
	•	Task: evaluator.py (Self-Evaluation & Metrics)
	•	Compares Groq's answers to the ground truth in the JSON
	•	Computes accuracy, confidence levels if provided (this part has not been implemented yet)
	•	Flags incorrectly answered prompts and writes them to a separate JSON for review
"""

import json
from formatter import file
import os

def load_files():
    # Ask user which JSON test file was being used if file is NULL
	global file
	if file == None:
		file = input("Enter the path/name of file that was used to test Groq: ")
		file = f"data/{file}"
	else:
		file = f"{file}"
	# Open the JSON file used to test Groq
	with open(file) as f:
		global test_questions
		test_questions = json.load(f)
          
	# Open the JSON file with Groq's responses
	with open("data/llm_responses.json") as f:
		global llm_answers
		llm_answers = json.load(f)

def evaluate():
	# Create dictionary to store correct answers from original JSON (answer_key)
	# Format: Question # : Correct Answer
	correct_answers = {}
	answer_key = {}
	for idx, question in enumerate(test_questions, 1): # idx is the index of the input-target score pairs (there are 3)
		# For each question, the answer choice (key) with a value of 1, i.e. the correct answer, is selected
		correct_answers = [k for k, v in question["target_scores"].items() if v == 1]
		# print(f"Q{idx}: {question['input']}")
		# print(f"Correct answer: {correct_answers}\n")
		answer_key[idx] = correct_answers

	# Compare LLM response to correct answers
	responses = {}

	i = 1
	for k, v in llm_answers.items(): # k = batch, v = dictionary representing batch
		# Right now, each v is a dictionary of all the questions in the batch
		# We have to access the "answer" field in v now
		for key, question_set in v.items():
			for question in question_set:
				responses[i] = question["answer"]
				i += 1

	# Dictionary to store questions that Groq got incorrect and its answers
	global incorrect_questions
	incorrect_questions = {}

	# Go through each question, compare, and add incorrect questions to the new dictionary
	for entry, value in answer_key.items():
		if value[0] != responses[entry]:
			incorrect_questions[entry] = responses[entry]

def store_results():
	# Save updated results
	with open("data/incorrect_questions.json", "w") as out_file:
		json.dump(incorrect_questions, out_file, indent=4)
	
	output_path = os.path.join("data", "incorrect_questions.json")
	print(f"Successfully saved evaluation results to {output_path}!")

def run_eval_process():
	load_files()
	evaluate()
	store_results()

if __name__ == "__main__":
    run_eval_process()