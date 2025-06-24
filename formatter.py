"""
This module handles all the formatting of our JSON data.
"""

import json

# Ask user which file to upload
file = input("Enter the path/name of file to be uploaded to Groq: ")
file = f"data/{file}"

# Open JSON
with open(file) as f:
    data = json.load(f)

# Set batch size
batch_size = 5

# Creates batches of 5 questions each
def format_json():
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size] # Holds 5 input-target pairs
        
        batch_inputs = ""
        # Iterate through each key-value pair in batch
        for idx, ex in enumerate(batch):
            options = list(ex["target_scores"].keys()) # Isolate the 5 answer options
            plain = "\n".join([f"- {opt}" for opt in options]) # Puts each option next to a dash for formatting
            batch_inputs += f"{idx+1}. Input:\n{ex['input']}\nOptions:\n{plain}\n\n" # Formats answer and options into consistent schema

