"""
loop_controller.py

Controls the closed feedback loop:
- Loads incorrectly answered prompts from runner.py
- Optionally modifies/rephrases prompts
- Feeds them back through the runner
- Tracks performance over multiple loops
"""

import json
import os
from .formatter import  load_json, clean_restructured_prompts
from .runner import get_llm_response, restructure_prompts, run_round, save_restructured_prompts
from .evaluator import run_eval_process
__docformat__ = "google"
def prompt_lock_loop(start_file, max_rounds=3):
    """
    Executes full prompt-feedback loop (only after the initial_run()),
    refining prompts and re-evaluating responses.

    Args:
        start_file (str): Path to the initial test file (e.g., "4_distractors.json").
        max_rounds (int): Maximum number of feedback loop iterations.
    """
    for i in range(1, max_rounds + 1): # i starts at 1 and ends at 3
        print(f"\033[1;96m🔁 === Starting Loop Iteration {i} ===\033[0m\n")

        # Strategy instructions for each round
        # if round_number == 1:
        #     strategy_instruction = "Mentally compare all options before selecting the best one."
        # elif round_number == 2:
        #     strategy_instruction = "Internally identify any important keywords before answering."
        # elif round_number == 3:
        #     strategy_instruction = "Think carefully through the reasoning before selecting your answer."
        # else:
        #     strategy_instruction = ""

        # Step 1: Evaluate responses from previous round to create incorrect_questions.json
        response_file = f"round_{i-1}_responses.json"
        print(f"\033[94mEvaluating responses from {response_file} against test set {start_file}.\033[0m")
        run_eval_process(start_file, response_file, i)

        # Step 2: Load previous round's incorrect questions and original prompts
        incorrect = load_json(f"FORAGER/data/incorrect_questions/round_{i}_incorrect.json")
        print(f"\033[1;94m📊 Loaded {len(incorrect)} incorrect questions from previous round.\033[0m\n")

        prompt_history = load_json(f"FORAGER/data/prompt_history/prompt_history_round_{i - 1}.json")

        # Step 3: Restructure prompts based on incorrect questions and rerun LLM on them
        restructured = restructure_prompts(incorrect, prompt_history)
        clean_prompts = clean_restructured_prompts(restructured)

        # Save cleaned prompts to JSON for round i
        save_restructured_prompts(clean_prompts, i)
        
        # Feed into next round
        run_round(f"new_prompts/restructured_prompts_round_{i}.json", i)