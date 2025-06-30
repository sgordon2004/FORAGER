"""
This module coordinates the full evaluation workflow for FORAGER.

It loads and formats batched question sets, sends them to the Groq LLM for processing,
saves the LLM's responses to a JSON file, and initiates the evaluation process to
identify incorrect answers.

Key Functions:
- initial_run(): Orchestrates the full prompt → response → evaluation pipeline.
- rerun_incorrect(): Allows targeted re-evaluation of previously incorrect questions.
"""

import requests
import os
import json
from dotenv import load_dotenv
from .formatter import format_json, load_json
import re
import time

# Load API key from environment
load_dotenv()

# Groq API setup
groq_endpoint = "https://api.groq.com/openai/v1/chat/completions"
model_name = "llama3-8b-8192"

# HTTP headers for the request (used for auth. + content type)
def get_headers():
    """
    Returns the headers required for Groq API requests.
    """
    return {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }

__pdoc__ = {
    "get_headers": False,  # Hide this function from documentation
}


# Batches questions into 5 at a time to build combined prompt
def build_prompt(formatted_batch):
    """
    Builds a prompt string from a batch of formatted questions for LLM input.

    Args:
        formatted_batch (list): A list of dictionaries, each containing a question and its options.

    Returns:
        str: A complete prompt formatted for Groq's LLM with JSON instructions.
    """

    prompt = ("You are a helpful assistant. Answer the following questions clearly and concisely. Provide only the final answer for each question, labeled by its number.\n\n"
        "Questions:\n")

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

def get_llm_response(prompt, max_retries=3):
    """
    Sends a prompt to the Groq LLM and retrieves the JSON response.

    Args:
        prompt (str): The input prompt to send to the LLM.
        max_retries (int, optional): Number of retry attempts if the request fails. Defaults to 3.

    Returns:
        str: The LLM's raw response content or error message if all retries fail.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                groq_endpoint,
                headers = get_headers(),
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
            print(f"*** Attempt #{attempt} failed: {e} ***\n")
            if attempt < max_retries:
                print("*** Retrying in 8 seconds... ***\n")
                time.sleep(8)
            else:
                print("*** Max retries reached. Returning error message. ***\n")
                return str(e)

def initial_run(test_file):
    """
    Executes the full LLM evaluation pipeline:
    - Builds prompts from formatted question batches
    - Sends prompts to the LLM and collects responses
    - Parses and attaches answers to each question
    - Saves all results to a JSON file in the data directory
    """
    qid_counter = 1

    # Dictionary to hold original prompts
    prompt_history = {}
    results = {}

    # Load questions from formatter.py
    test_path = os.path.join("FORAGER", "data", test_file)
    questions = format_json(load_json(test_path))

    # Loop through batched questions
    for i, questions_batch in enumerate(questions, 1):

        print(f"\033[1;94m🔄 Working on Batch #{i}/{len(questions)} total batches...\033[0m\n")

        # Build a batch prompt and get LLM response
        prompt = build_prompt(questions_batch)
        raw_answer = get_llm_response(prompt)

        # Save individual prompts for each question in this batch
        for idx, q in enumerate(questions_batch, 1):
            q_prompt = (
                "You are a helpful assistant. Answer the following question clearly and concisely. "
            "Provide only the final answer.\n\n"
            f"Question:\n{q['input']}\nOptions:\n"
            )
            for opt in q["options"]:
                q_prompt += f"{opt}\n"
            q_prompt += """\n
Format your response as JSON with this structure:
{
    "1": "..."
}
            """
            prompt_history[f"Q{qid_counter}"] = q_prompt
            qid_counter += 1

        try:
            # Extract JSON block safely using regex
            match = re.search(r"\{[\s\S]*\}", raw_answer)
            if match:
                parsed_answer = json.loads(match.group(0))
            else:
                print(f"\033[1;91m*** Warning: No valid JSON found for Batch #{i} ***\033[0m\n")
                parsed_answer = {}
        except json.JSONDecodeError:
            print(f"\033[1;91m*** Warning: could not parse answer for Batch_{i} ***\n")
            parsed_answer = {}

        # Attach answers directly to each question
        for idx, q in enumerate(questions_batch, 1):
            q["answer"] = parsed_answer.get(str(idx), None)

        results[f"Batch_{i}"] = {
            "questions": questions_batch,
        }
    
    os.makedirs("FORAGER/data/prompt_history", exist_ok=True)
    prompt_history_path = os.path.join("FORAGER", "data", "prompt_history", "prompt_history_round_0.json")
    with open(prompt_history_path, "w") as f:
        json.dump(prompt_history, f, indent=2)

    # Save responses to a JSON file
    llm_response_dir = os.path.join("FORAGER", "data", "llm_responses")
    os.makedirs(llm_response_dir, exist_ok=True)
    output_path = os.path.join("FORAGER", "data", "llm_responses", "round_0_responses.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\033[1;92m✅ Saved LLM responses to {output_path}!\033[0m\n")


if __name__ == "__main__":
    initial_run()


    #Iterate through 5 questions only and get the output from the LLM then 
    #Store output in a new JSON
    #Then put that file in the Data folder