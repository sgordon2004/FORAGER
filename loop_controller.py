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
from runner import get_llm_response

INCORRECT = "incorrect_questions.json"
GROUND_TRUTH = "4_distractors.json"

if __name__ == "__main__":
    # Load incorrectly answered questions
    with open("data/incorrect_questions.json") as f:
        incorrect_data = json.load(f)
    print(f"Loaded {len(incorrect_data)} incorrectly answered questions.")

    # Load ground truth
    with open(GROUND_TRUTH, "r") as f:
        ground_truth_data = json.load(f)
    print(f"Loaded {len(ground_truth_data)} given questions.")

    # Mock LLM function that simulates what runner.py will do (send the prompt to Groq and get a response)
    # Right now it always returns "improved response here" just for testing
    def run_llm(prompt):
        print(f"[RUNNER] Prompt sent: {prompt}")
        return "improved response here"  # Mocked improved response

    # Simulates the evaluator: checks whether the model’s output matches the true answer (ignoring case and extra spaces)
    def is_correct(response, ground_truth):
        return response.strip().lower() == ground_truth.strip().lower()

    # Optionally rephrases a prompt to help the LLM give a better answer on the next attempt
    def tweak_prompt(prompt):
        return prompt + " Please be specific and accurate."

    # Load failed prompts into a list of dictionaries
    with open("data/failed_prompts.json") as f:
        failed_prompts = json.load(f)

    # Number of attempts to try to get correct answer is 3 and results are stored in the array
    attempts = 3
    retry_results = []

    # Feedback Loop

    # Start looping through each failed prompt. Pulls out the prompt (input) and correct answer (answer).
    # The variable prompt can be tweaked later if needed.
    for ex in failed_prompts:
        original = ex["input"]
        answer = ex["answer"]
        prompt = original

        # For each prompt, try it up to 3 times. Call the mock LLM and print the result of each attempt.
        for attempt in range(1, attempts + 1):
            response = run_llm(prompt)
            print(f"Attempt {attempt}: {response}")

            # If the LLM gets the right answer, record that result (including how many tries it took),
            # label it as "correct", and exit the retry loop for that prompt.
            if is_correct(response, answer):
                retry_results.append({
                    "input": original,
                    "final_prompt": prompt,
                    "response": response,
                    "attempts": attempt,
                    "status": "correct"
                })
                break
            
            # If the first attempt failed, modify the prompt to (hopefully) improve the LLM’s next response.
            if attempt == 1:
                prompt = tweak_prompt(prompt)

            # Pause for 1 second before trying again to avoid spamming the LLM API
            # (practical when real API calls are made)
            time.sleep(1)

        # If the LLM still didn’t get it right after all attempts, log the final try and label the status as "still incorrect"
        else:
            retry_results.append({
                "id": ex.get("id", None),
                "input": ex["input"],
                "final_prompt": prompt,
                "response": response,
                "attempts": attempts,
                "status": "still incorrect"
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
