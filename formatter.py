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

# Isolate 5 inputs
for i in range(0, batch_size):
    example = data[i]["input"]
    print(example)