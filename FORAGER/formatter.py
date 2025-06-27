"""
formatter.py

This module loads the original JSON dataset and formats it into question batches
for LLM processing. Each batch contains a fixed number of questions with clearly
structured input and multiple-choice options.

Key Functions:
- load_json(): Loads the raw JSON data from file.
- format_json(): Splits the questions into batches and standardizes their format.
- write_formatted_json(): Saves the formatted batches to disk for LLM input.
"""

import json
import re

filename = None

def get_input_file():
    """
    Prompts the user to enter the path or filename to be uploaded to Groq.

    Returns:
        str: A valid path to the input JSON file.
    """
    global filename
    if filename is None:
        filename = input("Enter the path/name of file to be uploaded to Groq: ")
    return f"FORAGER/data/{filename}"

def load_json(filepath):
    """
    Opens the input JSON to be used.

    Args:
        file (str): Optional path to a file. If None, prompts user for it.
    
    Returns:
        dict: loaded JSON data
    """
    with open(filepath) as f:
        print(f"Loading {filepath}...\n")
        return json.load(f)

# # Set batch size
# batch_size = 5

# Creates batches of 5 questions each

def format_json(data, batch_size=5):
    """
    Splits the loaded JSON data into batches and formats each question with its options.
    
    Args:
        batch_size (int): Number of questions per batch. Defaults to 5.

    Returns:
        list: A list of batches, where each batch is a list of formatted question dictionaries.
    """
    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size] # Holds 5 input-target pairs
        
        formatted_batch = []
        # Iterate through each key-value pair in batch
        for ex in batch:
            options = list(ex["target_scores"].keys()) # Isolate the 5 answer options
            plain = "\n".join([f"- {opt}" for opt in options]) # Puts each option next to a dash for formatting
            # Store each question as a dictionary
            formatted_question = {
                "input": ex["input"],
                "options": options
            }
            formatted_batch.append(formatted_question)
        batches.append(formatted_batch)
    
    return batches

def clean_restructured_prompts(data):
    cleaned = {}

    for qid, raw_str in data.items():
        try:
            obj = json.loads(raw_str)
            question = obj["question"].strip()
            options = [opt.strip() for opt in obj["options"]]
        except Exception:
            question = raw_str.strip()
            options = []

        cleaned[qid] = {
            "question": question,
            "options": options
        }

    return cleaned

if __name__ == "__main__":
    get_input_file()