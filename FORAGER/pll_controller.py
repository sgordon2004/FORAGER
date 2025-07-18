# PLL CONTROLLER
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from FORAGER.runner import get_llm_response, regenerate_with_strict_grounding, regenerate_or_retrieve_more, lock_answer
from FORAGER.evaluator import evaluate

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
    
def regenerate_with_strict_grounding(question, claim, eval_label, confidence_label, retrieved_docs_text):
    """
    Regenerates the answer or sub-answer based on a single failed claim and source grounding.

    Args:
        question (str): Original user question.
        claim (str): The specific atomic claim being regenerated.
        answer (str): The original LLM answer.
        eval_label (str): Evaluation label for the claim (Unsupported/Contradicted).
        confidence_label (str): Confidence level (Low, Medium, High).
        retrieved_docs_text (List[str]): Retrieved grounding context chunks.

    Returns:
        str: A regenerated answer focused on the claim with strict grounding.
    """

    prompt = f"""
You are evaluating a specific claim made about the following question:

QUESTION:
{question}

CLAIM TO EVALUATE:
"{claim}"

EVALUATION LABEL: {eval_label}
CONFIDENCE LEVEL: {confidence_label}

AVAILABLE CONTEXT:
{chr(10).join(retrieved_docs_text)}

INSTRUCTIONS:
- Rewrite the claim ONLY IF it can be fully supported by the provided context.
- If the context contradicts the claim, explain the contradiction briefly.
- If the context does not support the claim, respond with: "❌ The claim is not supported by the provided context and should be discarded."
- Do NOT use any outside knowledge or make up information.

REVISED CLAIM (or appropriate rejection message):
"""

    print(f"🔄 Regenerating claim with strict grounding:\nClaim: {claim}\nEval: {eval_label}, Confidence: {confidence_label}")
    regenerated_answer = get_llm_response(prompt)
    print(f"✅ Regenerated Answer for Claim:\n{regenerated_answer}")

    return regenerated_answer

from FORAGER.embedder import embed_text, search_database, get_chunk_embeddings, cosine_similarity

def rerank_or_rephrase(question, claim, retrieved_docs_text, top_n=3):
    """
    Rerank retrieved chunks based on claim similarity and rephrase claim accordingly.
    """
    log(f"🔎 Reranking chunks based on claim-to-chunk similarity for claim: {claim}")
    
    # Step 1: Embed claim
    claim_embedding = embed_text(claim)

    # Step 2: Embed each retrieved chunk
    chunk_embeddings = get_chunk_embeddings(retrieved_docs_text)  # returns List[np.array]

    # Step 3: Compute similarities
    similarities = [cosine_similarity(claim_embedding, chunk_emb) for chunk_emb in chunk_embeddings]

    # Step 4: Sort chunks by similarity
    ranked_chunks = [doc for _, doc in sorted(zip(similarities, retrieved_docs_text), key=lambda x: x[0], reverse=True)]

    # Step 5: Select top-N chunks
    top_chunks = ranked_chunks[:top_n]

    # Step 6: Rephrase claim using top-ranked context
    prompt = f"""
QUESTION: {question}

CLAIM TO IMPROVE: "{claim}"

TOP CONTEXT:
{chr(10).join(top_chunks)}

INSTRUCTIONS:
- Rewrite the claim to more directly reflect the above context.
- Only use the provided context.
- If the claim cannot be improved, respond: "❌ Claim should be discarded."

REPHRASED CLAIM:
"""
    rephrased_claim = get_llm_response(prompt)
    log(f"✅ Rephrased Claim:\n{rephrased_claim}")

    return rephrased_claim

def retrieve_more_or_rephrase(question, claim, k=5):
    """
    Expands retrieval based on claim and optionally rephrases the claim after refreshing context.
    """
    log(f"🔎 Expanding retrieval focused on claim: {claim}")

    # Step 1: Reformulate retrieval query based on claim
    claim_query = f"{question}. Focus on: {claim}"

    # Step 2: Re-run retrieval using claim-augmented query
    expanded_retrieval = search_database(claim_query, top_k=k)
    expanded_context = [doc["text"] for doc in expanded_retrieval]

    # Step 3: Rephrase claim using expanded context
    prompt = f"""
QUESTION: {question}

CLAIM TO IMPROVE: "{claim}"

EXPANDED CONTEXT:
{chr(10).join(expanded_context)}

INSTRUCTIONS:
- Rewrite the claim to be directly grounded in this new context.
- Only use the provided context.
- If the context does not support the claim, respond: "❌ Claim is unsupported and should be discarded."

REPHRASED CLAIM:
"""
    improved_claim = get_llm_response(prompt)
    log(f"✅ Rephrased Claim after expanded retrieval:\n{improved_claim}")

    return improved_claim

def pll(eval_label, confidence_label):
    """
    Makes a decision for a single claim based on its evaluation and confidence.

    Args:
        eval_label (str): The evaluation label assigned to the claim by the BS detector.
        confidence_label (str): The confidence level assigned to the claim by the confidence checker.
    
    Return:
        str: The PLL path the claim should follow.
    """
    if eval_label == "Unsupported":
        if confidence_label in ["Medium", "High"]:
            log("✅ Confident but unsupported: Possibly correct but unverifiable.")
            return "RETRIEVE_MORE_OR_REPHRASE"
        else:
            log("❌ Low-confidence unsupported: Discarding claim.")
            return "DISCARD"
        
    elif eval_label == "Contradicted":
        log("🚨 Contradicted by evidence: Blocking or forcing strict grounding.")
        return "STRICT_REGENERATION"

    elif eval_label == "Supported":
        if confidence_label == "High Confidence":
            log("✅ Supported with high confidence: Locking in answer.")
            return "LOCK"
        else:
            log("⚠️ Supported but weak semantic match: Reranking or rephrasing.")
            return "RERANK_OR_REPHRASE"

    else:
        log(f"⚠️ Unhandled label combo: eval={eval_label}, confidence={confidence_label}")
        return "REVIEW_MANUALLY"

def handle_decision(decision, claim, claim_eval, question):
    eval_label = claim_eval["label"]
    confidence = claim_eval["confidence"]

    retrieved_docs_text = [doc["text"] for doc in claim_eval.get("supporting_chunks", [])]

    if decision == "LOCK":
        log(f"🔒 Locking claim: {claim}")
        return None
    elif decision == "DISCARD":
        log(f"🗑️ Discarding claim: {claim}")
        return None
    elif decision == "STRICT_REGENERATION":
        log(f"🔄 Strictly regenerating claim: {claim}")
        new_claim = regenerate_with_strict_grounding(question, claim, eval_label, confidence, retrieved_docs_text)
        return new_claim
    elif decision == "RETRIEVE_MORE_OR_REPHRASE":
        log(f"♻️ Rephrasing or retrieving more for claim: {claim}")
        new_claim = retrieve_more_or_rephrase(question, claim)
        return new_claim
    elif decision == "RERANK_OR_REPHRASE":
        log(f"♻️ Reranking context and rephrasing claim: {claim}")
        new_claim = rerank_or_rephrase(question, claim, retrieved_docs_text)
        return new_claim
    else:
        log(f"❓ Manual review needed for claim: {claim} with eval {claim_eval}")
        return None



def prompt_locked_loop(question, eval, max_retry=3):
    from bs import detect_bs
    from confidence import check_confidence
    # Initialize log
    pll_logs = []
    pll_round = 1

    while pll_round <= max_retry:
        log(f"🔁 Starting PLL round {pll_round}")
        round_log = {"pll_round": pll_round, "claims": []}
        updated_claims = []

        # Iterate over each atomic claim and its evaluation (label + confidence)
        # and decide the PLL path for each claim
        for claim, claim_eval in eval.items():
            eval_label = claim_eval["label"]
            confidence_label = claim_eval["confidence"]

            decision = pll(eval_label, confidence_label)

            log(f"📌 Claim: {claim}\nDecision: {decision}")

            round_log["claims"].append({
                "claim": claim,
                "eval_label": eval_label,
                "confidence_label": confidence_label,
                "pll_decision": decision,
                "reason": f"Decision {decision} due to eval {eval_label} and confidence {confidence_label}"
            })

            new_claim = handle_decision(decision, claim, claim_eval, question)
            
            if new_claim:
                updated_claims.append(new_claim)
    
        pll_logs.append(round_log)
        log(f"✅ Completed PLL round {pll_round}")

        # Re-evaluate updated claims if any
        if updated_claims:
            log("🔄 Re-evaluating updated claims...")
            new_eval = {}

            for claim in updated_claims:
                # Use supporting docs from retrieved context 
                supporting_docs = claim_eval["supporting_chunks"]

                # Detect BS and Confidence
                label = detect_bs(claim, [doc["text"] for doc in supporting_docs])
                confidence = check_confidence(claim, label, supporting_docs)

                # Store re-evaluation result
                new_eval[claim] = {
                    "label": label,
                    "confidence": confidence,
                    "supporting_chunks": supporting_docs
                }

            eval = new_eval
        else:
            log(f"✅ No new claims to evaluate, carrying forward previous eval.")

        pll_round += 1

    log("🏁 Reached max PLL rounds, stopping.")
    return pll_logs