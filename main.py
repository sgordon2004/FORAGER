"""
main.py

Entry point for the FORAGER system.

This script initiates the complete workflow:
- Loads and formats question batches
- Sends them to Groq's LLM
- Stores the LLM’s responses
- Evaluates the accuracy of responses
- Saves incorrectly answered questions for review

To run the full pipeline, execute this file directly:
    python main.py
"""

from dotenv import load_dotenv
import os
from .runner import initial_run
from .evaluator import run_eval_process



# Entry point
if __name__ == "__main__":

    # Import your API key from .env file
    load_dotenv()
    API_KEY = os.getenv("GROQ_API_KEY")
    print(f"Using API key: {API_KEY[:8]}")

    initial_run() # Feeds questions to LLM and saves output file
    # Now we want to evaluate the answers
    run_eval_process()