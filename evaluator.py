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
"""

import json
from .formatter import get_input_file
import os

def load_files():
	"""
	Opens the JSON files containing the original questions and LLM responses.
	"""
    # Ask user which JSON test file was being used if file is NULL
	# global file
	file = get_input_file()
	if not file:
		file = input("Enter the path/name of file that was used to test Groq: ")
		file = f"data/{file}"
	# Open test file
	with open(file) as f:
		global test_questions
		test_questions = json.load(f)
          
	# Open the JSON with Groq's responses
	with open("data/llm_responses.json") as f: # Change this name to whatever Aurora names it
		global llm_answers
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

	global incorrect_questions
	incorrect_questions = {}

	# Go through each question, compare, and add is_correct
	for entry, value in answer_key.items():
		if value[0] != responses[entry]:
			incorrect_questions[entry] = responses[entry]

def store_results():
	"""
	Stores the incorrect answers in a new JSON file.
	"""
	# Save updated results
	with open("data/incorrect_questions.json", "w") as out_file:
		json.dump(incorrect_questions, out_file, indent=4)
	
	output_path = os.path.join("data", "incorrect_questions.json")
	print(f"Successfully saved evaluation results to {output_path}!")

def run_eval_process():
	"""
	Runs the entire evaluation pipeline, from loading the files to storing the final results.
	"""
	load_files()
	evaluate()
	store_results()

if __name__ == "__main__":
    run_eval_process()