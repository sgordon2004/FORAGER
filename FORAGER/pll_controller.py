# PLL CONTROLLER
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from FORAGER.runner import get_llm_response
from FORAGER.embedder import FAISSEmbedder
from typing import List, Tuple

def log(message):
    print(f"[PLL_LOG] {message}")
    
def regenerate_with_strict_grounding(question, claim, eval_label, confidence_label, retrieved_docs: List[str]):
    """
    Regenerates a revised version of a calim using strict grounding from provided context.

    This function prompts an LLM to rewrite the claim only if it is fully supported by the
    retrieved documents. It enforces tight constraints to avoid hallucination or speculative
    phrasing, ensuring the revised claim is short, factual, and faithful to the source material.

    If the claim cannot be supported or is contradicted by the evidence, a standardized rejection
    message is returned.

    Args:
        question (str): Original user question.
        claim (str): The specific atomic claim being regenerated.
        answer (str): The original LLM answer.
        eval_label (str): Evaluation label for the claim (Unsupported/Contradicted).
        confidence_label (str): Confidence level (Low, Medium, High).
        retrieved_docs_text (List[str]): Retrieved grounding context chunks.

    Returns:
        str: A regenerated version of the claim grounded strictly in the provided context,
            or a rejection message if it cannot be supported.
    """

    # System Prompt - Defines LLM behavior and expected return format
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
                - Rewrite the claim ONLY if it can be fully supported by the provided context, ensuring the claim is as short and concise as possible while preserving meaning:
                - Focus strictly on the central fact.
                - Do NOT restate examples or explanations from the context.
                - Avoid phrases like “this enables” or “which allows”; directly state the key fact.
                - If the context contradicts the claim, explain the contradiction briefly.
                - If the context does not support the claim, respond with: "❌ The claim is not supported by the provided context and should be discarded."
                - Avoid adding unnecessary qualifiers, explanations, or details.
                - Do NOT use any outside knowledge or make up information.
                - Do NOT include any text in your response other than the revised claim or rejection message. NO MESSAGES, NOTES, OR EXPLANATIONS.

                REVISED CLAIM (or appropriate rejection message):
            """

    print(f"🔄 Regenerating claim with strict grounding:\nClaim: {claim}\nEval: {eval_label}, Confidence: {confidence_label}")
    regenerated_answer = get_llm_response(prompt)
    print(f"✅ Regenerated Claim :\n{regenerated_answer}")

    return regenerated_answer

# def rerank_or_rephrase(embedder: FAISSEmbedder, question, claim, retrieved_docs: List[dict], top_n=3):
#     """
#     Reranks retrieved chunks based on their semantic similarity to the claim and rephrases
#     the claim using the most relevant context.

#     This function enhances the grounding of a claim by first computing the similarity
#     between the claim and each retrieved context chunk, selecting the top-N most relevant chunks,
#     and prompting an LLM to rephrase the claim strictly based on that context.

#     Args:
#         embedder (FAISSEmbedder): An embedder instance used to compute embeddings and similarity between claim and context.
#         question (str): The original question asked by the user from which the claim was derived.
#         claim (str): The atomic claim to be rephrased.
#         retrieved_docs (List[dict]): A list of dictionaries. Each dictionary key is a chunk and values are scores, id, etc.
#         top_n (int): The number of most relevant chunks to retrieve. Defaults to 3.

#     Returns:
#         str: A rephrased version of the claim grounded in the top-ranked context,
#             or a rejection message if the claim cannot be improved.
#         list: A list of strings storing the newly ranked chunks.
#     """
#     log(f"🔎 Reranking chunks based on claim-to-chunk similarity for claim: {claim}")
    
#     # Step 1: Embed original claim
#     claim_embedding = embedder.embed_text(claim)

#     # Step 2: Embed each retrieved chunk
#     chunk_embeddings = embedder.embed_chunks(retrieved_docs)

#     # Step 3: Compute claim-chunk similarities
#     similarities = [embedder.cosine_similarity(claim_embedding, chunk_emb) for chunk_emb in chunk_embeddings]

#     # Step 4: Sort chunks by similarity
#     ranked_chunks = [doc for _, doc in sorted(zip(similarities, retrieved_docs), key=lambda x: x[0], reverse=True)]

#     # Step 5: Select top-N chunks
#     top_chunks = ranked_chunks[:top_n]

#     # Step 6: Rephrase claim using top-ranked context
#     # System Prompt - Defines LLM behavior and expected return format
#     prompt = f"""
#                 QUESTION: {question}

#                 CLAIM TO IMPROVE: "{claim}"

#                 TOP CONTEXT:
#                 {chr(10).join(top_chunks)}

#                 INSTRUCTIONS:
#                 - Rewrite the claim to more directly reflect the context, while keeping it as short and concise as possible.
#                 - Avoid repeating phrases from the context verbatim.
#                 - Do NOT use any outside knowledge or make up information.
#                 - If the claim cannot be improved, respond: "❌ Claim should be discarded."
#                 - Do NOT include any text in your response other than the revised claim or rejection message. NO MESSAGES, NOTES, OR EXPLANATIONS.

#                 REPHRASED CLAIM:
#             """
#     rephrased_claim = get_llm_response(prompt)
#     log(f"✅ Rephrased Claim:\n{rephrased_claim}")

#     return rephrased_claim, top_chunks


def rerank_or_rephrase(embedder: FAISSEmbedder, question, claim, retrieved_docs: List[dict], top_n=3):
    """
    Reranks retrieved chunks based on max sentence-level similarity to the claim,
    and rephrases the claim using the most relevant full chunks.

    Args:
        embedder (FAISSEmbedder): An embedder instance used to compute embeddings and similarity between claim and context.
        question (str): The original question asked by the user from which the claim was derived.
        claim (str): The atomic claim to be rephrased.
        retrieved_docs (List[dict]): A list of dictionaries, each with a 'text' key (chunk) and optional metadata.
        top_n (int): The number of most relevant chunks to retrieve. Defaults to 3.

    Returns:
        str: A rephrased version of the claim grounded in the top-ranked context,
             or a rejection message if the claim cannot be improved.
        list: A list of strings storing the newly ranked full chunk texts.
    """
    from sentence_transformers.util import cos_sim
    import re

    def split_into_sentences(text):
        return [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if len(s.strip()) > 10]

    log(f"🔎 Sentence-aware reranking for claim: {claim}")

    claim_embedding = embedder.embed_text(claim)

    chunk_scores = []
    print(f"retrieved docs: {retrieved_docs}")
    for chunk in retrieved_docs["supporting_chunks"]:
        print(f"Chunk format: {chunk}")
        full_text = chunk["text"]
        sentences = split_into_sentences(full_text)

        if not sentences:
            continue

        sentence_embeddings = [embedder.embed_text(sentence) for sentence in sentences]
        max_score = max(
            float(cos_sim(claim_embedding, s_emb))
            for s_emb in sentence_embeddings
        )
        chunk_scores.append((max_score, full_text))

    # Rank by max sentence similarity
    chunk_scores.sort(reverse=True, key=lambda x: x[0])
    top_chunks = [chunk for _, chunk in chunk_scores[:top_n]]

    top_texts = set(top_chunks)

    # Rephrase the claim using the top full chunks
    prompt = f"""
                QUESTION: {question}

                CLAIM TO IMPROVE: "{claim}"

                TOP CONTEXT:
                {chr(10).join(top_chunks)}

                INSTRUCTIONS:
                - Rewrite the claim to more directly reflect the context, while keeping it as short and concise as possible.
                - Avoid repeating phrases from the context verbatim.
                - Do NOT use any outside knowledge or make up information.
                - If the claim cannot be improved, respond: "❌ Claim should be discarded."
                - Do NOT include any text in your response other than the revised claim or rejection message. NO MESSAGES, NOTES, OR EXPLANATIONS.

                REPHRASED CLAIM:
            """
    rephrased_claim = get_llm_response(prompt)
    log(f"✅ Rephrased Claim:\n{rephrased_claim}")

    # Use original dicts from retrieved_docs["supporting_chunks"] where the text matches
    used_chunks = [
        chunk for chunk in retrieved_docs["supporting_chunks"]
        if chunk["text"] in top_texts
    ]
    return rephrased_claim, used_chunks
    
def retrieve_more_or_rephrase(embedder: FAISSEmbedder, question, claim, k=5) -> Tuple[str, List[str]]:
    """
    Retrieves additional context based on the atomic claim and rephrases the claim using
    the newly retrieved grounding.
    
    This function reformulates the search query to emphasize the claim itself, retrieves
    fresh supporting context from the vector database, and prompts and LLM to revise the claim
    strictly based on that context. It is used when the original evidence is insufficient
    to validate or rephrase the claim confidently.

    Args:
        embedder (FAISSEmbedder): An embedder instance used as the semantic search engine for context retrieval.
        question (str): The original user question that led to the atomic claim.
        claim (str): The atomic claim to be re-evaluated and rephrased.
        k (int): The number of top documents to retrieve using the updated query. Defaults to 5.

    Returns:
        str: A revised claim grounded in the newly retrieved context, or a rejection message 
            if the claim remains unsupported.
        list: A list of the newly retrieved chunks.
    """
    log(f"🔎 Expanding retrieval focused on claim: {claim}")

    # Step 1: Reformulate retrieval query based on claim
    claim_query = f"{question}. Focus on: {claim}"

    # Step 2: Re-run retrieval using claim-augmented query
    expanded_retrieval = embedder.search_database(claim_query, top_k=k)
    expanded_context = [doc["text"] for doc in expanded_retrieval]

    # Step 3: Rephrase claim using expanded context
    prompt = f"""
                QUESTION: {question}

                CLAIM TO IMPROVE: "{claim}"

                EXPANDED CONTEXT:
                {chr(10).join(chunk for chunk in top_chunks)}

                INSTRUCTIONS:
                - Rewrite the claim to be directly grounded in this new context.
                - Only use the provided context.
                - If the context does not support the claim, respond: "❌ Claim is unsupported and should be discarded."
                - Do NOT include any text in your response other than the revised claim or rejection message. NO MESSAGES, NOTES, OR EXPLANATIONS.

                REPHRASED CLAIM:
            """
    
    improved_claim = get_llm_response(prompt)
    log(f"✅ Rephrased Claim after expanded retrieval:\n{improved_claim}")

    return improved_claim, expanded_context

def pll(eval_label, confidence_label):
    """
    Determines the appropriate Prompt Locked Loop (PLL) action for a given claim based on its
    evaluation label and confidence level.

    This function acts as the decision policy layer of the PLL pipeline, using predefined rules
    to route each claim through an appropriate refinement path (e.g., reranking, regeneration, discard).

    Args:
        eval_label (str): The evaluation label assigned to the claim by the BS detector (e.g., "Supported", "Unsupported", "Contradicted").
        confidence_label (str): The confidence level assigned to the claim by the confidence checker (e.g., "High", "Medium", "Low", "Zero").
    
    Return:
        str: The decision path to apply to the claim. One of:
            - "LOCK": Claim is accepted as valid and does not require changes.
            - "DISCARD": Claim is rejected and removed from further consideration.
            - "RERANK_OR_REPHRASE": Claim should be rephrased using existing context.
            - "RETRIEVE_MORE_OR_REPHRASE": Claim should trigger retrieval of additional context, then be rephrased.
            - "STRICT_REGENERATION": Claim should be regenerated with stricter grounding constraints.
            - "REVIEW_MANUALLY": The system cannot determine a clear decision path; requires manual review.
    """
    if eval_label == "Unsupported":
        if confidence_label in ["Medium", "High"]:
            log("✅ Confident but unsupported: Possibly correct but unverifiable.")
            return "RETRIEVE_MORE_OR_REPHRASE"
        else:
            # --- OLD LOGIC ---
            # log("❌ Low-confidence unsupported: Discarding claim.")
            # return "DISCARD"

            # --- NEW LOGIC ---
            log("❌ Low-confidence and unsupported: Rephrasing prompt")
            return "RERANK_OR_REPHRASE"
        
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

def handle_decision(embedder: FAISSEmbedder, decision, claim, claim_eval, question):
    """
    Handles a PLL decision for a given claim by executing the appropriate corrective action,
    such as reranking, rephrasing, or strict regeneration.

    This function serves as a dispatch mechanism, applying the specified `decision` to the claim
    using the appropriate transformation strategy. It uses retrieved context to guide rephrasing
    or regeneration and returns the updated version of the claim, if applicable.

    Args:
        embedder (FAISSEmbedder): An embedder instance used for chunk retrieval and similarity computation.
        decision (str): The decision strategy returned by the PLL logic (e.g., "LOCK", "DISCARD", "STRICT_REGENERATION").
        claim (str): The atomic claim to be acted on.
        claim_eval (dict): The evaluation result for the claim, including 'label', 'confidence', and optionally 'supporting_chunks'.
        question (str): The original user question associated with the claim.

    Returns:
        str or None: A revised claim string if the claim is rephrased or regenerated.
                     Returns None if the claim is locked, discarded, or marked for manual review.
    """
    eval_label = claim_eval["label"]
    confidence = claim_eval["confidence"]

    print(f"⚠️⚠️⚠️ Value of `claim_eval` in handle_decision ⚠️⚠️⚠️\n{claim_eval}")
    retrieved_docs = claim_eval # with metadata
    retrieved_docs_text = [claim_eval["supporting_chunks"][i]["text"] for i, claim in enumerate(claim_eval.items())]

    if decision == "LOCK":
        log(f"🔒 Locking claim: {claim}")
        return None
    elif decision == "DISCARD":
        log(f"🗑️ Discarding claim: {claim}")
        return None
    elif decision == "STRICT_REGENERATION":
        log(f"🔄 Strictly regenerating claim: {claim}")
        new_claim = regenerate_with_strict_grounding(question, claim, eval_label, confidence, retrieved_docs_text)
        return new_claim, retrieved_docs_text
    elif decision == "RETRIEVE_MORE_OR_REPHRASE":
        log(f"♻️ Rephrasing or retrieving more for claim: {claim}")
        new_claim, used_chunks = retrieve_more_or_rephrase(embedder, question, claim)
        return new_claim, used_chunks
    elif decision == "RERANK_OR_REPHRASE":
        log(f"♻️ Reranking context and rephrasing claim: {claim}")
        new_claim, used_chunks = rerank_or_rephrase(embedder, question, claim, retrieved_docs)
        return new_claim, used_chunks
    else:
        log(f"❓ Manual review needed for claim: {claim} with eval {claim_eval}")
        return None

def prompt_locked_loop(embedder: FAISSEmbedder, question, eval, max_retry=3):
    """
    Runs the Prompt Locked Loop (PLL) on a set of evaluated atomic claims, iteratively
    refining and re-evaluating them to improve grounding and reliability.

    This function filters out high-confidence supported claims entering the loop,
    then processes the rest through multiple PLL rounds. Each round evaluates unsupported
    or contradicted claims using actions such as rephrasing, reranking context, or strict
    regeneration based on their evaluation and confidence. Claims are re-evaluated after each
    round. Claims that become high-confidence and supported after re-evaluation are locked
    and preserved.

    Args:
        embedder (FAISSEmbedder): An embedder instance used for semantic search and similarity operations.
        question (str): The original user question the claims are generated for.
        eval (dict): A dictionary where keys are atomic claims and values are their evaluation results,
            including 'label', 'confidence', and optionally 'supporting_chunks'.
        max_retry (int): The maximum number of PLL iterations to perform. Defaults to 3.

    Returns:
        tuple[list[dict], list[dict]]:
            - A list of dictionaries loging each PLL round, including claim decisions and reasoning.
            - A list of final locked claims that are supported with high confidence, each with its metadata.
    """
    from bs import detect_bs
    from confidence import check_confidence

    # Create a list for final claims (Supported + High Confidence)
    final_locked_claims = []

    # Initialize a Prompt Locked Loop Log to store the details of all the round's results
    pll_logs = []

    # Log the initial claims and evaluation before PLL rounds start
    initial_log = {
        "pll_round": 0,
        "claims": [
            {
                "claim": claim,
                "eval_label": claim_eval["label"],
                "confidence_label": claim_eval["confidence"],
                "pll_decision": "Initial",
                "reason": f"Initial evaluation detected {claim_eval['label']} with {claim_eval['confidence']} confidence."
            }
            for claim, claim_eval in eval.items()
        ]
    }
    pll_logs.append(initial_log)
    log("✅ Logged initial claims before starting PLL rounds.")

    # --- VERBOSITY FILTER ---
        # Compares the length of rephrased claims to the length of the original,
        # unmodified claim. If the rephrased claim ever becomes more than 2.5 times longer than
        # its original, it is discarded.

    # Store the original claim lengths for divergence detection
    original_claim_lengths = {claim: len(claim.split()) for claim in eval.keys()}
    # --------------------------------
    
    # Initiate a Round Log to store the PLL details for the pre-PLL evaluation round
    round_log = {"pll_round": "Pre-PLL Lock", "claims": []}

    # Step 1: Pre-PLL Check
        # - Check if any LLM claims are Supported + High confidence immediately after initial generation
        # - If so, lock them and remove them from further PLL evaluation
    for claim, claim_eval in eval.items():
        label = claim_eval["label"]
        confidence = claim_eval["confidence"]
        supporting_docs = claim_eval.get("supporting_chunks", [])

        if label == "Supported" and confidence == "High":
            log(f"✅ Locking claim before PLL: {claim}")
            # Update the round log to include these locked claims
            round_log["claims"].append({
                "claim": claim,
                "eval_label": label,
                "confidence_label": confidence,
                "pll_decision": "Locked before PLL",
                "reason": "Claim auto-locked with high confidence and support."
            })
            # Update the list of locked claims
            final_locked_claims.append({
                        "claim": claim,
                        "label": label,
                        "confidence": confidence,
                        "supporting_chunks": supporting_docs
            })
            # Skip further processing of this claim
            continue

    # Append this pre-round log to the PLL Log
    pll_logs.append(round_log)

    # Remove any immediately-locked claims to avoid further PLL processing
    eval = {
        claim: claim_eval
        for claim, claim_eval in eval.items()
        if not (claim_eval["label"] == "Supported" and claim_eval["confidence"] == "High")
    }
    print(f"\n‼️ STATE OF `eval` IN prompted_lock_loop() BEFORE PLL ROUND 1 ‼️\n")
    print(eval)
    # Step 2: Initiate the Prompt Locked Loop
    # Variable to track PLL rounds
    pll_round = 1

    while pll_round <= max_retry:
        log(f"🔁 Starting PLL round {pll_round}")

        # A list to keep track of rephrased claims
        updated_claims = []

        # Step 1: Decide PLL action for each claim
        for claim, claim_eval in eval.items():
            eval_label = claim_eval["label"]
            confidence_label = claim_eval["confidence"]

            decision = pll(eval_label, confidence_label)
            log(f"📌 Claim: {claim}\nDecision: {decision}")

            print(f"claim_eval: {claim_eval}")
            # Apply the decision and store the rephrased claim (or None or standardized message if the claim could not be supported)
            new_claim, used_chunks = handle_decision(embedder, decision, claim, claim_eval, question)

            # new_claim will be None if the claim could not be supported
            if new_claim is None:
                log(f"🗑️ Discarding claim: {claim}")
                continue

            # VERBOSITY CHECK
            # Discard claims that have diverged too much in length (e.g., >2.5x original length)
            MAX_REPHRASED_WORDS = 30
            if len(new_claim.split()) > MAX_REPHRASED_WORDS:
                    log(f"❌ Rephrased claim too LONG ({len(new_claim.split())} words). Discarding.")
                    log(f"📏 Rephrased claim length: {len(new_claim.split())} words (limit: {MAX_REPHRASED_WORDS})")
                    continue
            
            # Save the new rephrased claim to the list of ALL rephrased claims
            updated_claims.append((new_claim, used_chunks))
        
        # Initalize a new Round Log to store PLL details of rounds 1+
        round_log = {"pll_round": pll_round, "claims": []}

        # Step 3: Re-evaluate all updated claims and log fresh evaluation results
        if updated_claims:
            log("🔄 Re-evaluating updated claims...")
            new_eval = {}

            for claim, supporting_docs in updated_claims:
                label = detect_bs(embedder, claim, supporting_docs)
                confidence = check_confidence(embedder, claim, label, supporting_docs)

                # Locking check
                if label == "Supported" and confidence == "High":
                    log(f"✅ Locking claim after re-evaluation: {claim}")
                    round_log["claims"].append({
                    "claim": claim,
                    "eval_label": label,
                    "confidence_label": confidence,
                    "pll_decision": "Locked after re-evaluation",
                    "reason": "Claim auto-locked after reevaluation with high confidence and support."
                })
                    final_locked_claims.append({
                        "claim": claim,
                        "label": label,
                        "confidence": confidence,
                        "supporting_chunks": supporting_docs
                    })
                    # Skip further processing of this claim
                    continue

                # Store re-evaluation result
                new_eval[claim] = {
                    "label": label,
                    "confidence": confidence,
                    "supporting_chunks": supporting_docs
                }

                round_log["claims"].append({
                    "claim": claim,
                    "eval_label": label,
                    "confidence_label": confidence,
                    "pll_decision": "Reevaluated after processing",
                    "reason": f"Post-PLL reevaluation detected {label} with {confidence} confidence."
                })

            eval = new_eval
        else:
            log("✅ No new claims generated this round, stopping PLL.")
            eval = {}

        # Log the round results
        pll_logs.append(round_log)
        log(f"✅ Completed PLL round {pll_round}")
        pll_round += 1

        if not updated_claims:
            log("🏁 No claims left to process, stopping PLL.")
            break # Exit early if no claims left

    log("🏁 Reached max PLL rounds or no more claims to process, stopping.")
    return pll_logs, final_locked_claims

def synthesize_final_answer(question: str, locked_claims: list[str]) -> str:
    """
    Uses LLM to generate a final, fluent answer based on locked claims.

    Args:
        question (str): The original question asked by the user.
        locked_claims (list): All of the locked claims (Supported + High Confidence) returned by the LLM.
    
    Returns:
        str: The final well-supported answer to the user's original question, synthesized by the LLM.
             Turns the scattered, context-less atomic claims into a complete, more human-sounding answer.
    """
    # Extracts the text content of the locked claims
    claims_text = "\n".join(f"-{claim}" for claim in locked_claims)

    # System Prompt - Defines LLM behavior and expected return format
    prompt = f"""
                You are a helpful scientific assistant summarizing findings based on reliable information.

                QUESTION:
                {question}

                LOCKED CLAIMS:
                {claims_text}

                TASK:
                - Synthesize a clear, human-readable answer based on the locked claims.
                - Write in a concise and informative tone.
                - You may reorder or combine claims, but do not introduce any information not in the list.
                - The result should be a single coherent paragraph.
                - Do NOT include any bullet points or numbered lists in your output.

                FINAL ANSWER:
            """
    
    # Fallback - Runs if there are no locked claims to synthesize into a complete answer
    if not locked_claims:
        return "No claims were validated to answer the question."

    return get_llm_response(prompt).strip()