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
load_dotenv()
from formatter import format_json, load_json
import re
import time

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
questions = format_json(load_json())

# Variable to store general "helpful assistant prompt"
general_prompt = ("You are a helpful assistant. Answer the following questions clearly and concisely. Provide only the final answer for each question, labeled by its number.\n\n"
        "Questions:\n")

# Batches questions into 5 at a time to build combined prompt
def build_prompt(formatted_batch, prompt=general_prompt):
    """
    Builds a prompt string from a batch of formatted questions for LLM input.

    Args:
        formatted_batch (list): A list of dictionaries, each containing a question and its options.

    Returns:
        str: A complete prompt formatted for Groq's LLM with JSON instructions.
    """

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
    """
    Executes the full LLM evaluation pipeline:
    - Builds prompts from formatted question batches
    - Sends prompts to the LLM and collects responses
    - Parses and attaches answers to each question
    - Saves all results to a JSON file in the data directory
    """
    global_qid = 1

    # Dictionary to hold original prompts
    prompt_history = {}
    results = {}

    # Loop through batched questions
    for i, questions_batch in enumerate(questions, 1):
        print(f"Working on Batch #{i}/{len(questions)} total batches...\n")

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
            prompt_history[f"Q{global_qid}"] = q_prompt
            global_qid += 1

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
        
    with open("data/prompt_history.json", "w") as f:
        json.dump(prompt_history, f, indent=2)

    # Save responses to a JSON file
    output_path = os.path.join("data", "llm_responses.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved LLM responses to {output_path}!")


if __name__ == "__main__":
    initial_run()


    #Iterate through 5 questions only and get the output from the LLM then 
    #Store output in a new JSON
    #Then put that file in the Data folder