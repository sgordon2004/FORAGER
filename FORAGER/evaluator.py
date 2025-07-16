"""
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
__docformat__ = "google"
import json
from formatter import get_input_file
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
			# {
			# 	"Question and Answer Choices": test_questions[entry - 1],
			# 	"Groq's Answer": responses[entry],
			# 	"Correct Answer": answer_key[entry]
			# }

def store_results(round_number):
	"""
	Stores the incorrect answers in a new JSON file.
	"""
	# Save updated results
	os.makedirs("FORAGER/data/incorrect_questions", exist_ok=True)
	with open(f"FORAGER/data/incorrect_questions/round_{round_number}_incorrect.json", "w") as out_file:
		json.dump(incorrect_questions, out_file, indent=4)
	
	output_path = os.path.join("FORAGER", "data", "incorrect_questions", f"round_{round_number}_incorrect.json")
	print(f"\033[1;92m✅ Successfully saved evaluation results to {output_path}!\033[0m\n")

def summarize_round(round_number):
	"""
	Adds a short summary of Groq's accuracy at the end of incorrect_questions.json
	"""
	# Print the round summary
	print(f"\033[1;92m🔎 Round {round_number} Summary\033[0m")
	print(f"\033[1;92mNumber of Questions Asked: {len(test_questions)}\033[0m")
	num_correct = len(test_questions) - len(incorrect_questions)
	num_incorrect = len(incorrect_questions)
	print(f"\033[1;92mNumber Correct: {num_correct}\033[0m")
	print(f"\033[1;92mNumber Incorrect: {num_incorrect}\033[0m")
	accuracy = num_correct / len(test_questions)
	accuracy_str = f"{accuracy * 100:.2f}%"
	print(f"\033[1;92mAccuracy: {accuracy_str}\033[0m")

	# Load or initialize the accuracy tracking file 
	summary_path = "FORAGER/data/incorrect_questions/accuracy_summary.json"
	if os.path.exists(summary_path):
		with open(summary_path, 'r') as f:
			past_scores = json.load(f)
	else:
		past_scores = {}

	prev_round_key = str(round_number - 1)
	if prev_round_key in past_scores:
		prev_data = past_scores[prev_round_key]
		prev_accuracy = prev_data["accuracy"]
		prev_correct = prev_data["num_correct"]

		# Accuracy comparison
		accuracy_change = (accuracy - prev_accuracy) * 100
		sign = "+" if accuracy_change >= 0 else "-"
		print(f"\033[1;94mAccuracy Change from Previous Round: {sign}{abs(accuracy_change):.2f}%\033[0m")

		# Number correct comparison
		correct_change = num_correct - prev_correct
		sign = "+" if correct_change >= 0 else "-"
		print(f"\033[1;94mCorrect Answer Change from Previous Round: {sign}{abs(correct_change)}\033[0m")

	else:
		print("\033[1;94mNo previous round score to compare.\033[0m")

	# Save current round's results
	past_scores[str(round_number)] = {
		"accuracy": accuracy,
		"num_correct": num_correct
	}
	with open(summary_path, 'w') as f:
		json.dump(past_scores, f, indent=2)


def run_eval_process(test_file, responses_file, round_number):
	"""
	Runs the entire evaluation pipeline, from loading the files to storing the final results.
	"""
	load_files(test_file, responses_file)
	evaluate()
	store_results(round_number)
	summarize_round(round_number)

if __name__ == "__main__":
    run_eval_process()