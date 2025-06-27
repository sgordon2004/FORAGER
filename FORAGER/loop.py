"""
This module handles all the loop logic for our Prompt-Lock Loop (PLL) system.
"""

import requests
import os
from dotenv import load_dotenv

# Importing GROQ API Key
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# Loop function
def prompt_lock_loop():
    """
    Skeleton
    """
    while True:
        prompt = input("Enter prompt (or 'q' to quit): ")
        if prompt.lower() == 'q':
            break

        # Call to Groq 
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}"
            },
            json={
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        print("LLM Response: ", response.json())