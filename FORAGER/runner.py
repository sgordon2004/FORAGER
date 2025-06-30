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

# Hide the get_headers function from documentation
# This is useful if you want to keep the implementation details private
# and not expose it in the generated documentation.
__pdoc__ = {
    "get_headers": False
}

def parse_llm_response(raw_response, batch_num=None):
    """
    Extracts and parses the JSON portion of the LLM's raw response.

    Args:
        raw_response (str): The full string returned by the LLM.
        batch_num (int, optional): Used for printing contextual error messages.

    Returns:
        dict: Parsed JSON answers, or an empty dict if parsing fails.
    """
    try:
        match = re.search(r"<json>(.*?)</json>", raw_response, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        else:
            if batch_num is not None:
                print(f"\033[1;91m*** Warning: No valid JSON found for Batch #{batch_num} ***\033[0m\n")
            return {}
    except json.JSONDecodeError:
        if batch_num is not None:
            print(f"\033[1;91m*** Warning: could not parse answer for Batch_{batch_num} ***\n")
        return {}

def build_question_prompt(q):
    """
    Builds a prompt for a single question with options, formatted for LLM input.
    
    Args:
        q (dict): A dictionary containing the question and its options.
    
    Returns:
        str: A formatted prompt string for the question.
    """
    q_prompt = (
        "You are a helpful assistant. Answer the following question clearly and concisely."
        "Provide only the final answer.\n\n"
        f"Question:\n{q['input']}\nOptions:\n"
    )
    for opt in q["options"]:
        q_prompt += f"{opt}\n"
    q_prompt += """\n
        Format your response as JSON with this structure:
        <json>
        {
            "1": "..."
        }
        </json>
    """
    return q_prompt

def load_formatted_batch(test_file):
    """
    Loads and formats the test questions from the specified JSON file.
    """
    test_path = os.path.join("FORAGER", "data", test_file)
    return format_json(load_json(test_path))

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
<json>
{
    "1": "...",
    "2": "...",
    ...
}
</json>
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

def save_prompt_history(prompt_history, round_number):
    """
    Saves the individual prompt history for each question to a JSON file.

    This function creates a diirectory (if it doesn't exist) and writes the prompt
    history dictionary to a JSON file named according to the specified round number.

    Args:
        prompt_history (dict): A dictionary containing question IDs (e.g., "Q1", "Q2", ...) as keys and their prompts as values.
        round_number (int): The current evaluation round number (e.g., 0 for initial run), used to name the output file.

    Side Effects:
        Creates a JSON file at FORAGER/data/prompt_history/prompt_history_round_{round_number}.json
        containing all question prompts used during evaluation.
    """
    os.makedirs(os.path.join("FORAGER", "data", "prompt_history"), exist_ok=True)
    filename = f"prompt_history_round_{round_number}.json"
    path = os.path.join("FORAGER", "data", "prompt_history", filename)
    with open(path, "w") as f:
        json.dump(prompt_history, f, indent=2)

def save_llm_responses(results, round_number):
    """
    Saves the LLM-labeled responses to a JSON file.

    This function creates a directory (if it doesn't exist) and writes the results dictionary
    to a file named according to the specified round number.

    Args:
        results (dict): A dictionary mapping batch labels (e.g., "Batch_1", "Batch_2", ...) to their corresponding question and answer data.
        round_number (int): The current evaluation round number (e.g., 0 for initial run), used to name the output file.

    Side Effects:
        Creates a JSON file at FORAGER/data/llm_responses/round_{round_number}_responses.json
        containing all batches and their corresponding labeled questions.
    """
    os.makedirs(os.path.join("FORAGER", "data", "llm_responses"), exist_ok=True)
    filename = f"round_{round_number}_responses.json"
    path = os.path.join("FORAGER", "data", "llm_responses", filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\033[1;92m✅ Saved LLM responses to {path}!\033[0m\n")

def initial_run(test_file):
    """
    Coordinates the full LLM evaluation pipeline for a given test dataset.

    This function performs the following steps:
    1. Loads and formats questions from a specified JSON file.
    2. Splits questions into batches and constructs prompts.
    3. Sends each prompt batch to the Groq LLM and receives raw responses.
    4. Parses and attaches LLM answers to the original questions.
    5. Builds individual prompts for each question and stores them for traceability.
    6. Saves both prompt history and LLM-labeled responses to JSON files.

    Args:
        test_file (str): The name of the test file (inside `FORAGER/data/`) to load and evaluate.

    Side Effects:
        - Writes prompt history to `FORAGER/data/prompt_history/prompt_history_round_0.json`
        - Writes labeled LLM responses to `FORAGER/data/llm_responses/round_0_responses.json`

    Raises:
        Prints warnings if the LLM response is invalid or if JSON parsing fails.
    """
    qid_counter = 1

    # Dictionary to hold original prompts
    prompt_history = {}
    results = {}

    # Step 1: Loads raw JSON file and formats it
    question_batches = load_formatted_batch(test_file)

    for i, questions_batch in enumerate(question_batches, 1):
        print(f"\033[1;94m🔄 Working on Batch #{i}/{len(question_batches)} total batches...\033[0m\n")
        
        # Step 2: Build and send prompt for current batch of questions and store raw response
        prompt = build_prompt(questions_batch)
        raw_answer = get_llm_response(prompt)

        # Step 3: Build and store individual question prompts for traceability
        for q in questions_batch:
            q_prompt = build_question_prompt(q)
            prompt_history[f"Q{qid_counter}"] = q_prompt
            qid_counter += 1

        # Step 4: Parse the raw response to extract JSON answers
        parsed_answer = parse_llm_response(raw_answer, batch_num=i)

        # Step 5: Attach parsed answers to the original questions
        for idx, q in enumerate(questions_batch, 1):
            q["answer"] = parsed_answer.get(str(idx), None)

        # Step 6: Store the parsed answers in a results dictionary
        results[f"Batch_{i}"] = {"questions": questions_batch}
    
    # Step 7: Save the prompt history to a JSON file
    save_prompt_history(prompt_history, round_number=0)

    # Step 8: Save the LLM responses to a JSON file
    save_llm_responses(results, round_number=0)


if __name__ == "__main__":
    initial_run()


    #Iterate through 5 questions only and get the output from the LLM then 
    #Store output in a new JSON
    #Then put that file in the Data folder