"""
LLM Interaction & Response Collector
	•	Handles calling the Groq API with the formatted JSON prompts
	•	Stores the raw LLM responses back into a results JSON
	•	Manages retries & error handling (e.g., API rate limits, timeouts)

Iterate through 5 questions only and get the output from the LLM then 
Store output in a new JSON
Format it
Then put that file in the Data folder

"""

import requests                             # Makes HTTP requests (like to Groq)
import os                                   # To interact with env variables and OS
import json                                 # To encode/decode JSON data 
from dotenv import load_dotenv              # To load env variable from .env file into OS env
load_dotenv()                               # Calls to load the function 
from formatter import format_json           # To reformat prompts 
import re                                   # To use regular expressions (replace, search, pattern matching)
import time                                 # To use time related functions like delays 

# Load API key from environment
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# Groq API setup
groq_endpoint = "https://api.groq.com/openai/v1/chat/completions"
model_name = "llama3-8b-8192"

# HTTP headers for the request (used for auth. + content type)
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Load questions from formatter.py
questions = format_json()

# Batches questions into 5 at a time to build combined prompt
def build_prompt(formatted_batch): # Parameter imported from formatter.py
    prompt = (
        "You are a helpful assistant. Answer the following questions clearly and concisely. Provide only the final answer for each question, labeled by its number.\n\n"
        "Questions:\n"
    )

    for idx, question in enumerate(formatted_batch, 1):             #Formatting response 
        prompt += f"\n{idx}) {question['input']}\nOptions:\n"
        for opt in question["options"]:
            prompt += f"{opt}\n"
    prompt += """
Format your response as JSON with this structure:
{
    "1": "...",
    "2": "...",
    ...
}
    """
    return prompt

# print(build_prompt(questions[0]))

# Sends requests to Groq
def get_llm_response(prompt, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                groq_endpoint,
                headers = HEADERS,
                json = {
                    "model": "llama3-8b-8192",
                    "messages": [
                        {"role": "user",
                        "content": prompt}
                    ],
                    "temperature": 0 # forces no creativity
                },
                timeout = 10
            )
            # Raises HTTP error, if one occured
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:
            print(f"Attempt #{attempt} failed: {e}")
            if attempt < max_retries:
                print("Retrying in 8 seconds...\n")
                time.sleep(8)
            else:
                print("Max retries reached. Returning error message.")
                return str(e)

def initial_run():
    results = {}

    # Loop through batched questions
    for i, questions_batch in enumerate(questions, 1):
        print(f"Working on Batch #{i}/{len(questions)} total batches...\n")
        prompt = build_prompt(questions_batch)
        raw_answer = get_llm_response(prompt)

        try:
            # Extract JSON block safely using regex
            match = re.search(r"\{[\s\S]*\}", raw_answer)
            if match:
                parsed_answer = json.loads(match.group(0))
            else:
                print(f"Warning: No valid JSON found for Batch #{i}")
                parsed_answer = {}
        except json.JSONDecodeError:    # Meaning: If a JSON decoding error occurs, handle it here 
            print(f"Warning: could not parse answer for Batch_{i}")
            parsed_answer = {}

        # Attach answers directly to each question
        for idx, q in enumerate(questions_batch, 1):
            q["answer"] = parsed_answer.get(str(idx), None)

        results[f"Batch_{i}"] = {
            "questions": questions_batch,
        }

    # Save responses to a JSON file
    output_path = os.path.join("data", "llm_responses.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved LLM responses to {output_path}!")


if __name__ == "__main__":
    initial_run()


