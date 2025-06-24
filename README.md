# FORAGER

Instructions on running formatter.py:
import the format_json() method into the file you're working on that needs the data
format_json() returns all the questions in the JSON in a formatted version
Slice return by 5 so that you only feed 5 questions to Groq at a time (to avoid token timeout)