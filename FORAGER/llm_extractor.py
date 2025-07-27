"""
This module extracts atomic factual claims from text using a large language model (LLM) via Groq API. It is a key component 
of the FORAGER pipeline, enabling structured extraction of factual statements from unstructured LLM-generated responses.

Core Functionality:
- Defines a strict prompt for the LLM instructing it to extract atomic, standalone factual claims from a given input text.
- Sends the formatted prompt to the LLM using the `get_llm_response()` function and parses the raw response.
- Ensures the output is a valid Python list of strings, each representing a single factual claim.

Key Features:
- Automatically handles appositional phrases and compound sentences by splitting them into separate claims.
- Filters out rhetorical, opinionated, or vague statements to retain only verifiable factual claims.
- Includes error handling to catch parsing failures and unexpected LLM output formats.

Intended Usage:
This module is used after an LLM generates a response to a user query, providing a structured set of factual claims for 
further validation in the FORAGER pipeline (e.g., via BS detection and confidence scoring).

"""

import os 
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from FORAGER.runner import get_llm_response
import json
from dotenv import load_dotenv
__docformat__ = "google"

llm_claim_prompt = """
You are an information extraction system.

Your task is to extract **atomic, factual claims** from the provided text.

**Atomic factual claims** must follow these rules:
- Each claim expresses **only one complete fact**.
- Each claim must be a **standalone full sentence** that is understandable without additional context.
- Each claim must be in **active voice** whenever possible.
- **Do not combine multiple facts into a single claim**. If multiple benefits, features, or facts are listed, **split them into separate claims**.
- **If appositional phrases are used** (e.g., "TSMC, the largest contract chipmaker,"), **split them into distinct claims** (e.g., "TSMC is the largest contract chipmaker.").
- **Do not omit any factual content**, including information embedded in parentheses, commas, or after colons.

**Subject Handling Rules**:
- If the input text focuses on a main subject (e.g., "heterogeneous integration"), **each claim must clearly reference this subject as the actor or cause of the fact**. Claims should begin with this subject or use it as the grammatical focus of the sentence.
- **Do not shift the subject** to other entities (e.g., devices, products, benefits, or applications). Always **attribute actions and effects to the main subject**.

**Examples of correct claim splitting**:
Input:  
"Heterogeneous integration enhances performance and enables smaller devices."

Output:  
["Heterogeneous integration enhances performance.", "Heterogeneous integration enables smaller devices."]

**What to exclude**:
- ❌ Ignore rhetorical questions, opinions, speculative statements, or vague implications.
- ❌ Never start claims with vague phrases like "The benefits are…", "Advantages include…", "A key reason is…", "This approach…", or "It allows…". Always restate the main subject explicitly.
- ❌ DO NOT include any commentary, explanations, or statements like "Here are the claims:" or "Here is the output:".
- ❌ Do not use markdown, code blocks, or bullet points.

**Formatting Requirements**:
- ✅ Output **only** a valid **Python list of strings**, starting with `[` and ending with `]`, with no text before or after.
- ✅ Example of correct final output:  
["Heterogeneous integration enhances performance.", "Heterogeneous integration enables smaller devices.", "Heterogeneous integration improves system efficiency."]

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
        print(f"[DEBUG] LLM raw response:\n{response}")
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