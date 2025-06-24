import requests
import os
import json
from dotenv import load_dotenv
load_dotenv()

#load API key from environment
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

#Groq API setup
Groq_Endpoint = "https://api.groq.com/openai/v1/chat/completions"
model_name = "llama3-8b-8192"
HEADERS = {
    "Authorization":f"Bearer {API_KEY}"
}


os.makedirs("data", exist_ok=True) #ensure data directory exists

#loading prompts from data folder 
def load_prompts(path = "data/4_distractors.json", limit=5):
    try:
        with open(path, "r") as f:
            all_prompts = json.load(f)
            return all_prompts[:limit]
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return []

#sending 5 to Groq
def get_llm_response(prompt):
    try:
        response = requests.post(
            Groq_Endpoint,
            headers = HEADERS,
            json = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout = 10
        )
        response.raise_for_status()
        return response.json()["choice"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching response: {e}")
        return str(e)

def main():
    prompts = load_prompts()
    results = {}

    for i, prompt in enumerate(prompts, 1):
        print(f"{i}: {prompt}")
        answer = get_llm_response(prompt)
        results[f"Q{i}"] = {
            "prompt": prompt,
            "answer": answer
        }

#saving responses to a JSON file
    output_path = os.path.join("data", "llm_responses.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(output_path)


if __name__ == "__main__":
    main()




    #Iterate through 5 questions only and get the output from the LLM then 
    #Store output in a new JSON
    #Then put that file in the Data folder