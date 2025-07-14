from FORAGER.runner import get_llm_response
import json
from dotenv import load_dotenv

llm_claim_prompt = """
You are an information extraction engine. Your task is to extract all **atomic factual claims** from the input text below.

An atomic factual claim:
- Expresses one complete fact only
- Should be stated as a full, standalone sentence
- Should be stated in active voice if possible
- Should avoid combining multiple facts into one sentence

You must:
- Extract appositional phrases as separate claims  
  (e.g., "TSMC, the world's largest contract chipmaker, manufactures..." → "TSMC is the world's largest contract chipmaker" and "TSMC manufactures ...")
- Split compound sentences into separate atomic claims
- Do not omit any fact, even if it is embedded in commas or parentheses
- Ignore rhetorical questions, opinions, or vague implications
- Remove punctuation and special characters (e.g., . , $ / &)

Output only a valid **Python list of strings**, one per claim. Do not include explanations or code blocks. Do not include anything at all in your response besides the Python list of strings, and ONLY that.

### Input:
\"\"\"{input_text}\"\"\"

### Output:
"""

def extract_atomic_claims_llm(text: str) -> str:
    """
    Extracts atomic factual claims from text using Groq LLM.

    Args:
        text (str): Input text (paragraphs).

    Returns:
        list[str]: List of atomic factual claims.
    """
    import ast
    prompt = llm_claim_prompt.format(input_text=text)
    response = get_llm_response(prompt)

    try:
        claims = ast.literal_eval(response)
        if isinstance(claims, list):
            return claims
        else:
            print("[WARNING] LLM returned non-list:", response)
            return []
    except Exception as e:
        print(f"[ERROR] Failed to parse LLM response: {e}")
        print(response)
        return []
    
test = "NVIDIA, the leader in graphics processing units, announced its latest AI chip in 2024. The chip is manufactured using TSMC’s 3nm process technology. Despite global supply chain disruptions, NVIDIA reported record quarterly revenue. Meanwhile, Intel continues to expand its foundry services but struggles to match TSMC’s efficiency. Apple has also shifted more chip production to TSMC to support its custom silicon roadmap."

print(f"[DEBUG] llm_extractor.py Sending input to LLM:\n{test}")
print(extract_atomic_claims_llm(test))