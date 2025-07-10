# FORAGER Prompt Library

--- 

# Prompt: 'regenerate_with_strict_grounding'
**Version:** v1.1
**Status:** Active
**Last Updated:** 2025-07-10

# Purpose: 
Used to generate a response after evaluator or grounding failure, ensuring strict adherence to RAG source text and feedback instructions.

--- 

# Prompt Structure: 

**Variables:**
 - task: The original task or question.
 - previous_output: The intitial (failed) model response.
 - source_text: Retrieved source content to ground the answer.
 - evaluator_feedback: Feedback on what failed in the previous response. 

**Prompt Template:**
'Your previous answer contradicted the source. Regenerate your response strictly using the information below. 

TASK: 
{task}

PREVIOUS RESPONSE: 
"""{previous_output}"""

SOURCE: 
"""{source_text}"""

EVALUATOR_FEEDBACK:
{evaluator_feedback}

INSTRUCTIONS:
 - Use only facts from the source. 
 - Do not add or speculate. 
 - If unsure, say: "The answer is not in the source."

Corrected Response: 
