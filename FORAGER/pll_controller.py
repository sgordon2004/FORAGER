# PLL CONTROLLER

from runner import get_llm_response, regenerate_with_strict_grounding, regenerate_or_retrieve_more, lock_answer
from evaluator import evaluate
from bs import detect_bs
from embedder import confidence_checker

def log(message):
    print(f"[PLL_LOG] {message}")


def run_pll_on_prompt(task_prompt,question_id="Q1", max_attempts=5, threshold=0.9):
    log(f"Starting PLL Controller with prompt: {task_prompt}")
    history = []
    for atttempt in range(max_attempts):
        candidate = get_llm_response(task_prompt)
        eval_result = evaluate(candidate)

        score = eval_result.get("score", 0)
        label = eval_result.get("label", "unkown")
        eval_label = eval_result.get("eval_label", "unknown")
        confidence_label = eval_result.get("confidence", "unknown")
        rationale = eval_result.get("rationale", None)

        history.append({
            "candidate": candidate,
            "score": score,
            "label": label,
            "eval_label": eval_label,
            "confidence_label": confidence_label
        })

        if eval_label == "Unsupported":
            if confidence_label == "Medium":
                log("Answer is confident but unsupported - likely true but unverifiable.")
                candidate = regenerate_or_retrieve_more()
            else:
                log("Answer is low-confidence and unsupported - discard.")
        elif eval_label == "Contradicted":
            log("Contradiction with evidence - block or alert.")
            candidate = regenerate_with_strict_grounding()
        elif eval_label == "Supported":
            if confidence_label == "High Confidence":
                lock_answer(
                    question_id=question_id,
                    final_answer = candidate,
                    confidence_score = score,
                    rationale = rationale
                )
                return {"status": "locked", "candidate": candidate, "history": history}

            else: 
                log("Weak semantic match despite support - rephrase or rerank.")
        
        if score >= threshold and label == "correct":
            return {"status": "success", "candidate": candidate, "history": history}

    best_guess = max(history, key=lambda x: x["score"])
    return {"status": "fail", "best_guess": best_guess, "history": history}
    

