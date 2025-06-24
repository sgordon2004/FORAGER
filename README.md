# FORAGER

Instructions on running formatter.py:
1. Import the format_json() method into the file you're working on that needs the data
2. format_json() returns a dictionary of all the questions
    - Each entry in the dictionary holds 5 questions
    - Each key is the input
    - Each value holds the answer choices
3. Feed format_json() into Groq one index at a time (since each index holds 5 questions)