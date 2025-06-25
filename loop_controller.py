"""
loop_controller.py

Controls the closed feedback loop:
- Loads incorrectly answered prompts from runner.py
- Optionally modifies/rephrases prompts
- Feeds them back through the runner
- Tracks performance over multiple loops
"""

import json
import time
from runner import run_groq # feeds incorrect questions to loop_controller one at a time for checking

INCORRECT = "incorrect.json"  # placeholder name for when runner makes the JSON file
GROUND_TRUTH = "4_distractors.json" # will have to change later

# load incorrect prompts
with open(INCORRECT, "r") as f:
    incorrect_data = json.load(f)
print(f"Loaded {len(incorrect_data)} incorrect prompts.")

# load ground truth
with open(GROUND_TRUTH, "r") as f:
    ground_truth_data = json.load(f)
print(f"Loaded {len(ground_truth_data)} ground truth entries.")

# helper function for grabbing correct answer for a given id
def get_correct_answer(qid):
    for i in ground_truth_data:
        if i["id"] == qid:
            return next(k for k, v in i["target_scores"].items() if v == 1)
    return None

# check if model's answer matches correct answer
def is_correct(response, correct):
    return response.strip().lower() == correct.strip().lower()

# optionally rephrase prompt
def tweak_prompt(prompt):
    return prompt + " That is incorrect. Please check your answer again and be more accurate." # this needs to change in a way

# retry loop parameters
attempts = 3
retry_results = []

# prompt-lock loop
for ex in incorrect_data:
    qid = ex["id"]
    correct_answer = get_correct_answer(qid)
    if not correct_answer:
        print(f"Warning: No ground truth found for ID {qid}")
        continue

    # start with original prompt
    prompt_dict = ex.copy()

    for attempt in range(1, attempts + 1):
        response = run_groq(prompt_dict)
        print(f"[{qid}] Attempt {attempt}: {response}")

        if is_correct(response, correct_answer):
            retry_results.append({
                "id": qid,
                "input": ex["input"],
                "final_prompt": prompt_dict["input"],
                "response": response,
                "attempts": attempt,
                "status": "correct"
            })
            break

        # tweak for second/third attempts
        if attempt == 1:
            prompt_dict["input"] = tweak_prompt(prompt_dict["input"])

        time.sleep(1)  # prevents spam in API with 1 sec delay

    else:
        retry_results.append({
            "id": qid,
            "input": ex["input"],
            "final_prompt": prompt_dict["input"],
            "response": response,
            "attempts": attempts,
            "status": "still incorrect"
        })

# save results
with open("data/retry_results.json", "w") as f:
    json.dump(retry_results, f, indent=2)

print("Loop complete. Results written to data/retry_results.json")