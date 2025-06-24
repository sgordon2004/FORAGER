from dotenv import load_dotenv
import os
from loop import prompt_lock_loop

# Import your API key from .env file
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
print(f"Using API key: {API_KEY[:8]}")

# Entry point
if __name__ == "__main__":
    prompt_lock_loop()