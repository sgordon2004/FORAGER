from FORAGER.runner import get_llm_response
import json
from dotenv import load_dotenv

llm_claim_prompt = """
You are an information extraction engine. Your task is to extract all **atomic factual claims** from the input text below.

An atomic factual claim:
- Expresses one complete fact only
- Should be stated as a full, standalone sentence
- Should be written in active voice when possible
- Avoids combining multiple facts into one sentence
- Extracts appositional phrases as separate claims
  (e.g., "TSMC, the world's largest chipmaker, manufactures..." → "TSMC is the world's largest chipmaker" and "TSMC manufactures...")
- Skips rhetorical questions, opinions, or vague implications
- Removes all punctuation (. , ! $ &)

Your output should be a valid Python list of strings. Do not include explanations or extra formatting.
Do not include anything in your answer at all, besides the final Python list of strings.

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
    prompt = llm_claim_prompt.format(input_text=text)
    response = get_llm_response(prompt)

    return(response)
    
test = "NVIDIA, the leader in graphics processing units, announced its latest AI chip in 2024. The chip is manufactured using TSMC’s 3nm process technology. Despite global supply chain disruptions, NVIDIA reported record quarterly revenue. Meanwhile, Intel continues to expand its foundry services but struggles to match TSMC’s efficiency. Apple has also shifted more chip production to TSMC to support its custom silicon roadmap."

print(extract_atomic_claims_llm(test))