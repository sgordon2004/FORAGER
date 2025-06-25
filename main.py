from dotenv import load_dotenv
import os
from loop import prompt_lock_loop
from runner import initial_run
from evaluator import run_eval_process

# Import your API key from .env file
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
print(f"Using API key: {API_KEY[:8]}")

# Entry point
if __name__ == "__main__":
    initial_run() # Feeds questions to LLM and saves output file
    # Now we want to evaluate the answers
    run_eval_process()