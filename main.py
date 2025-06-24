from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
print(f"Using API key: {API_KEY[:8]}")