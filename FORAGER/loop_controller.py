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
from .runner import get_llm_response, initial_run
from .evaluator import run_eval_process

# Build new prompt based off of incorrect questions
new_prompts = {}
def build_new_prompts(incorrect, prompt_history):
    """
    Uses the original prompt for each incorrect question and asks the LLM to restructure it.

    Returns:
        dict: Rephrased prompts to feed back to the LLM.
    """
    for k, v in incorrect.items():
        if not k.isdigit():
            continue # Skip metadata keys
        q_id = f"Q{k}"
        original_prompt = prompt_history.get(q_id, "[Prompt not found]")
        restructure_prompt = f"""
    You answered the following question incorrectly.
    
    Here was the original prompt:

    {original_prompt}

    Your previous answer:
    "{v}"

    Please revise the original prompt so that it is clearer and more direct. Be sure to reuse the exact original multiple-choice options without changing, reordering, or adding new ones.
    Then extract the rephrased question and its multiple-choice options in the following format:
    
    Respond with a JSON object like this:
    {{
    "question": "<rephrased question>",
    "options": ["Option A", "Option B", "Option C", "Option D", "Option E"]
    }}

    Do not include any explanation or formatting outside of the JSON object.
    """
        new_prompt = get_llm_response(restructure_prompt)
        new_prompts[q_id] = new_prompt
    
    os.makedirs("FORAGER/data/new_prompts", exist_ok=True)
    with open("FORAGER/data/new_prompts/restructured_prompts.json", "w") as f:
        json.dump(new_prompts, f, indent = 2)
    
    return new_prompts


# Feed new prompts to LLM
responses = {}

def rerun(round_number, clean_prompts):
    for qid, qdata in clean_prompts.items():
        q_prompt = (
            "You are a helpful assistant. Answer the following question clearly and concisely. "
            "Provide only the final answer.\n\n"
            f"Question:\n{qdata['question']}\nOptions:\n"
        )
        for opt in qdata["options"]:
            q_prompt += f"{opt}\n"
        q_prompt += """\n
    Only choose from the options listed above. Do not create new answers or reword existing ones.
    Format your response as JSON with this structure:
    {
    "1": "..."
    }
        """

        # Get response and format answer
        response_json = get_llm_response(q_prompt)
        try:
            parsed = json.loads(response_json)
            final_answer = parsed.get("1", "").strip()
        except Exception:
            final_answer = "[PARSE ERROR]"
        
        responses[qid] = {
            "question": qdata["question"],
            "options": qdata["options"],
            "answer": final_answer
        }

    # Save responses to file
    with open(f"FORAGER/data/llm_responses/round_{round_number}_responses.json", "w") as f:
        json.dump(responses, f, indent=2)

    print(f"\033[1;92m✅ Saved LLM responses to to data/round_{round_number}_responses.json!\n")
            

def prompt_lock_loop(test_file, rounds=3):
    """
    Executes full prompt-feedback loop (only after the initial_run()),
    refining prompts and re-evaluating responses.

    Args:

    """
    for i in range(1, rounds + 1): # i starts at 1 and ends at 3
        prev_round = i

        

        # Step 2: Evaluate responses to create incorrect_questions.json

        response_file = f"round_{i-1}_responses.json"
        print(f"\033[94mEvaluating responses from {response_file} against test set {test_file}.\033[0m")
        run_eval_process(test_file, response_file, i)

        print(f"\n\033[1;96m🔁 === Starting Loop Iteration {i} ===\033[0m\n")

        # Step 3: Load incorrect questions and original prompts
        incorrect = load_json("FORAGER/data/incorrect_questions/incorrect_questions.json")

        print(f"\033[1;94m📊 Loaded {len(incorrect)-3} incorrect questions from previous round.\033[0m\n")

        prompt_history = load_json(f"FORAGER/data/prompt_history/prompt_history_round_{prev_round - 1}.json")

        # Step 4: Build improved prompts and rerun
        restructured = build_new_prompts(incorrect, prompt_history)
        clean_prompts = clean_restructured_prompts(restructured)

        with open("FORAGER/data/new_prompts/clean_restructured_prompts.json", "w") as f:
            json.dump(clean_prompts, f, indent = 2)

        rerun(i, clean_prompts)

        # Step 5: Create prompt_hisotry for the next round based on current round response
        with open(f"FORAGER/data/llm_responses/round_{i}_responses.json") as f:
            round_responses = json.load(f)
        
        next_prompt_history = {}
        for qid, data in round_responses.items():
            question = data.get("question", "")
            options = data.get("options", [])
            formatted = f"{question}\nOptions:\n" + "\n".join(options)
            next_prompt_history[qid] = formatted

        with open(f"FORAGER/data/prompt_history/prompt_history_round_{i}.json", "w") as f:
            json.dump(next_prompt_history, f, indent=2)