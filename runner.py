import requests
import os
import json
from dotenv import load_dotenv
load_dotenv()
from formatter import format_json 
import re

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
def build_prompt(formatted_batch):
    prompt = (
        "You are a helpful assistant. Answer the following questions clearly and concisely. Provide only the final answer for each question, labeled by its number.\n\n"
        "Questions:\n"
    )

    for idx, question in enumerate(formatted_batch, 1):
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
def get_llm_response(prompt):
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
    except requests.exceptions.RequestException as e:
        print(f"Error fetching response: {e}")
        return str(e)

def main():
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
        except json.JSONDecodeError:
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

    print(output_path)


if __name__ == "__main__":
    main()




    #Iterate through 5 questions only and get the output from the LLM then 
    #Store output in a new JSON
    #Then put that file in the Data folder