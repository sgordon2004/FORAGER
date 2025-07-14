# PLL CONTROLLER

from runner import get_llm_response, regenerate_with_strict_grounding, regenerate_or_retrieve_more
from evaluator import evaluate
from bs import detect_bs
from embedder import confidence_checker

def run_ppl_on_prompt(task_prompt, max_attempts=5, threshold=0.9):
    history = []
    for atttempt in range(max_attempts):
        candidate = get_llm_response(task_prompt)
        score, label = evaluate(candidate)

        eval_label = eval_result['eval_label']
        confidence_label = eval_result['confidence_label']

        history.append((candidate, eval_label, confidence_label))

        if eval_label == "Unsupported":
            if confidence_label == "Medium" or confidence_label == "High Confidence":
                log("Answer is confident but unsupported - likely true but unverifiable.")
                regenerate_or_retrieve_more()
            else:
                log("Answer is low-confidence and unsupported - discard.")
        elif eval_label == "Contradicted":
            log("Contradiction with evidence - block or alert.")
            regenerate_with_strict_grounding()
        if score >= threshold and label == "correct":
            return {"status": "success", "candidate": candidate, "history": history}

    return {"status": "fail", "best_guess": max(history, key= "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"), "history": history}
    


# Step 1: Generate an intitial response 

# Step 2: Evaluate the response using FORAGER

# Step 3: Check if evaluation passed 

# Step 4: Retry using grounding and feedback

