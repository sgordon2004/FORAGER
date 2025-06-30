"""
evaluator.py

This module evaluates the performance of Groq's LLM by comparing its responses
against ground-truth answers. It computes accuracy metrics, flags incorrect answers,
and saves them for further review.

Key Functions:
- load_files(): Loads both the original question file and LLM responses.
- evaluate(): Compares LLM answers to the answer key and identifies mismatches.
- store_results(): Writes incorrect responses to a separate JSON file.
- run_eval_process(): Executes the full evaluation workflow.

TODO: Make evaluation for each round show up separately like in the llm_responses folder
"""

import json
from .formatter import get_input_file
import os

def load_files(test_file, responses_file):
	"""
	Opens the JSON files containing the original questions and LLM responses.
	"""
	global test_questions, llm_answers

	# Load the original question file
	with open(f"FORAGER/data/{test_file}") as f:
	# with open(test_file) as f:
		test_questions = json.load(f)

	# Load the LLM responses
	with open(f"FORAGER/data/llm_responses/{responses_file}") as f:
		llm_answers = json.load(f)

def evaluate():
	"""
	Creates three new dictionaries:
	1. answer_key - holds the orignal questions matched with their correct answers
	2. responses - holds the LLM responses in order
	3. incorrect_questions - stores only the questions the LLM answered incorrectly
	Compares the data in answer_key and incorrect_questions to evaluate the LLMs accruacy.
	"""
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
	# Check structure: batch-formatted (round 0) or flat-formatted (later rounds)?
	if all(isinstance(v, dict) and "questions" in v for v in llm_answers.values()):
		# Handle round_0_responses.json
		for batch in llm_answers.values():
			for question in batch["questions"]:
				responses[i] = question["answer"]
				i += 1
	else:
		# Handle round_1_responses.json and later
		for question_num, question_data in llm_answers.items():
			responses[int(question_num[1:])] = question_data["answer"]
			i += 1

	# Dictionary to store questions that Groq got incorrect and its answers
	global incorrect_questions
	incorrect_questions = {}

	# Go through each question, compare, and add incorrect questions to the new dictionary
	for entry, value in answer_key.items():
		if entry not in responses:
			continue # LLM did not answer this question in this round
		if not value:
			continue # No correct answer available for this question
		if value[0] != responses[entry]:
			incorrect_questions[entry] = responses[entry]

	# Adds a short summary of Groq's accuracy at the end of incorrect_questions.json
	incorrect_questions["Total Questions"] = len(test_questions)
	incorrect_questions["Correct Answers"] = len(test_questions) - len(incorrect_questions)
	incorrect_questions["Incorrect Answers"] = len(incorrect_questions)
	accuracy = (len(test_questions) - len(incorrect_questions)) / len(test_questions)
	accuracy_str = f"{accuracy * 100:.2f}%"
	incorrect_questions["Accuracy"] = accuracy_str

def store_results(round_number): # maybe rename this summarize_round(round_num) and add a summary printed to console
	"""
	Stores the incorrect answers in a new JSON file.
	"""
	# Save updated results
	os.makedirs("FORAGER/data/incorrect_questions", exist_ok=True)
	with open(f"FORAGER/data/incorrect_questions/round_{round_number}_incorrect.json", "w") as out_file:
		json.dump(incorrect_questions, out_file, indent=4)
	
	output_path = os.path.join("FORAGER", "data", "incorrect_questions", f"round_{round_number}_incorrect.json")
	print(f"\033[1;92m✅ Successfully saved evaluation results to {output_path}!\033[0m\n")

# def summarize_round(round_num):


def run_eval_process(test_file, responses_file, round_number):
	"""
	Runs the entire evaluation pipeline, from loading the files to storing the final results.
	"""
	load_files(test_file, responses_file)
	evaluate()
	store_results(round_number)

if __name__ == "__main__":
    run_eval_process()