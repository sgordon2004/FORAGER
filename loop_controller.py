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
from formatter import  load_json, clean_restructured_prompts
from runner import get_llm_response, build_prompt

if __name__ == "__main__":

    # Load incorrect questions and original questions
    incorrect = load_json("data/incorrect_questions.json")
    ground_truth = load_json("data/4_distractors.json")
    prompt_history = load_json("data/prompt_history.json")

    # Build new prompt based off of incorrect questions
    new_prompts = {}
    def build_new_prompts():
        """
        Uses the original prompt for each incorrect question and asks the LLM to restructure it.

        Returns:
            dict: Rephrased prompts to feed back to the LLM.
        """
        for k, v in incorrect.items():
            q_id = f"Q{k}"
            original_prompt = prompt_history.get(q_id, "[Prompt not found]")
            restructure_prompt = f"""
        You answered the following question incorrectly.
        
        Here was the original prompt:

        {original_prompt}

        Your previous answer:
        "{v}"

        Please revise the original prompt so that it is clearer and more direct. Then extract the rephrased question and its multiple-choice options in the following format:
        
        Respond with a JSON object like this:
        {{
        "question": "<rephrased question>",
        "options": ["Option A", "Option B", "Option C", "Option D", "Option E"]
        }}

        Do not include any explanation or formatting outside of the JSON object.
        """
            new_prompt = get_llm_response(restructure_prompt)
            new_prompts[q_id] = new_prompt
        
        with open("data/restructured_prompts.json", "w") as f:
            json.dump(new_prompts, f, indent = 2)
        
        return new_prompts

    restructured = build_new_prompts()

    clean_prompts = clean_restructured_prompts(restructured)

    with open("data/clean_restructured_prompts.json", "w") as f:
            json.dump(clean_prompts, f, indent = 2)

    # # Save new answers to dictionary
    # retry_answers = {}

    # for qid, prompt in restructured.items():
    #     new_response = get_llm_response(prompt)
    #     retry_answers[qid] = new_response
    
    # with open("data/retry_answers.json", "w") as f:
    #     json.dump(retry_answers, f, indent = 2)

#     # Mock LLM function that simulates what runner.py will do (send the prompt to Groq and get a response)
#     # Right now it always returns "improved response here" just for testing
#     def run_llm(prompt):
#         print(f"[RUNNER] Prompt sent: {prompt}")
#         return "improved response here"  # Mocked improved response

#     # Simulates the evaluator: checks whether the model’s output matches the true answer (ignoring case and extra spaces)
#     def is_correct(response, ground_truth):
#         return response.strip().lower() == ground_truth.strip().lower()

#     # Optionally rephrases a prompt to help the LLM give a better answer on the next attempt
#     def tweak_prompt(prompt):
#         return prompt + " Please be specific and accurate."

#     # Load failed prompts into a list of dictionaries
#     with open("data/failed_prompts.json") as f:
#         failed_prompts = json.load(f)

#     # Number of attempts to try to get correct answer is 3 and results are stored in the array
#     attempts = 3
#     retry_results = []

#     # Feedback Loop

#     # Start looping through each failed prompt. Pulls out the prompt (input) and correct answer (answer).
#     # The variable prompt can be tweaked later if needed.
#     for ex in failed_prompts:
#         original = ex["input"]
#         answer = ex["answer"]
#         prompt = original

#         # For each prompt, try it up to 3 times. Call the mock LLM and print the result of each attempt.
#         for attempt in range(1, attempts + 1):
#             response = run_llm(prompt)
#             print(f"Attempt {attempt}: {response}")

#             # If the LLM gets the right answer, record that result (including how many tries it took),
#             # label it as "correct", and exit the retry loop for that prompt.
#             if is_correct(response, answer):
#                 retry_results.append({
#                     "input": original,
#                     "final_prompt": prompt,
#                     "response": response,
#                     "attempts": attempt,
#                     "status": "correct"
#                 })
#                 break
            
#             # If the first attempt failed, modify the prompt to (hopefully) improve the LLM’s next response.
#             if attempt == 1:
#                 prompt = tweak_prompt(prompt)

#             # Pause for 1 second before trying again to avoid spamming the LLM API
#             # (practical when real API calls are made)
#             time.sleep(1)

#         # If the LLM still didn’t get it right after all attempts, log the final try and label the status as "still incorrect"
#         else:
#             retry_results.append({
#                 "id": ex.get("id", None),
#                 "input": ex["input"],
#                 "final_prompt": prompt,
#                 "response": response,
#                 "attempts": attempts,
#                 "status": "still incorrect"
#             })
#             break

#         # tweak for second/third attempts
#         if attempt == 1:
#             prompt_dict["input"] = tweak_prompt(prompt_dict["input"])

#         time.sleep(1)  # prevents spam in API with 1 sec delay

#     else:
#         retry_results.append({
#             "id": qid,
#             "input": ex["input"],
#             "final_prompt": prompt_dict["input"],
#             "response": response,
#             "attempts": attempts,
#             "status": "still incorrect"
#         })

# # save results
# with open("data/retry_results.json", "w") as f:
#     json.dump(retry_results, f, indent=2)

# print("Loop complete. Results written to data/retry_results.json")
