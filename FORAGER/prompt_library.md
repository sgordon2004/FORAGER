# FORAGER Prompt Library

----------------------------------------------------------

# Prompt: 'regenerate_with_strict_grounding'
**Version:** v1.0
**Status:** Active
**Last Updated:** 2025-07-10

# Purpose: 
Used to generate a response after evaluator or grounding failure, ensuring strict adherence to RAG source text and feedback instructions.

# Prompt Structure: 

**Variables:**
 - task: The original task or question.
 - previous_output: The intitial (failed) model response.
 - source_text: Retrieved source content to ground the answer.
 - evaluator_feedback: Feedback on what failed in the previous response. 

**Prompt Template:**
'Your previous answer contradicted the source. Regenerate your response strictly using the information below.'

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

----------------------------------------------------------

# Prompt: 'restructure_prompt' 
**Version:** v2.0
**Status:** Active
**Last Updated:** 2025-07-10

# Purpose: 
Used to rewrite incorrect questions using their original prompts via the LLM 

# Prompt Structure: 

**Variables:**
 - task: The original task or question.
 - previous_output: The intitial (failed) model response.
 - source_text: Retrieved source content to ground the answer.
 - evaluator_feedback: Feedback on what failed in the previous response.

**Prompt Template:** 
'Reword the following question prompt for clarity and conciseness'

TASK: 
{task}

PREVIOUS PROMPT: 
"""{previous_input}"""

PREVIOUS RESPONSE: 
"""{previous_output}"""

EVALUATOR_FEEDBACK:
{evaluator_feedback}

INSTRUCTIONS:
 - Do NOT change, paraphrase, or relabel the answer choices in any way.
 - Do NOT add letters, numbers, or any other prefixes in front of the options.
 - Only reword the question itself.
 
 Correct Response: 

 ----------------------------------------------------------

# Prompt: 'generator_prompt' 
**Version:** v3.0
**Status:** Active
**Last Updated:** 2025-07-10

# Purpose: 
Instructs the LLM to complete the task using context

# Prompt Structure:

**Variables:**
 - task: The original task or question.
 - source_text: Retrieved source content to ground the answer.
 - evaluator_feedback: Feedback on what failed in the previous response.

**Prompt Template:** 
'You are an expert assistant. Use the following source to answer the question.'

SOURCE: 
{source_text}

TASK: 
{task}

INSTRUCTIONS:
 - Only include the information that is directly supported by the source. 
 - Do NOT guess or add external knowledge.
 - If not enough information is provided, say: "The answer is not provided in the source."
 
 Correct Response: 

 ----------------------------------------------------------

# Prompt: 'refine_on_failure_prompt' 
**Version:** v4.0
**Status:** Active
**Last Updated:** 2025-07-10

# Purpose: 
Used to adjust and resubmit failed outputs based on the evaluation feedback

# Prompt Structure: 

**Variables:**
 - task: The original task or question.
 - source_text: Retrieved source content to ground the answer.
 - evaluator_feedback: Feedback on what failed in the previous response.

**Prompt Template**
'Your previous response included unsupported statements. Revise your answer using only the provided source.'

SOURCE:
{source_text}

TASK: 
{task}

EVALUATOR_FEEDBACK:
{evaluator_feedback}

INSTRUCTIONS:
 -Do not include any information that is not directly supported by the source.
 - Quote or paraphrase exact lines from the source where possible. 

 Correct Response: 

 ----------------------------------------------------------

 # Prompt: 'clarifying_instructions_prompt' 
**Version:** v4.1
**Status:** Active
**Last Updated:** 2025-07-10

# Purpose: 
Used to clarify instructions for LLM to process, in order to correct response 

# Prompt Structure: 

**Variables:**
 - task: The original task or question.
 - source_text: Retrieved source content to ground the answer.
 - previous_output: The intitial (failed) model response.
 - evaluator_feedback: Feedback on what failed in the previous response.

**Prompt Template:**
'Your previous response included unsupported statements. Revise your answer using only the provided source.'

You may NOT:
 - Invent, guess, or add external knowledge
 - Generalize unless clearly supported by the source 

You MUST: 
 - Use only the retrieved source to answer
 - Quote or paraphrase directly 
 - Acknowledge if the source is incomplete 

TASK: 
{task}

SOURCE: 
{source_text}

PREVIOUS RESPONSE: 
{previous_output}

EVALUATOR FEEDBACK: 
{evaluator_feedback}

 Revise your answer below:

----------------------------------------------------------

# Prompt: 'rejection_prompt' 
**Version:** v5.0
**Status:** Active
**Last Updated:** 2025-07-10

# Purpose: 
Used to control retries, escalation, and final locking of results

# Prompt Structure: 

**Variables:**
 - task: The original task or question.
 - source_text: Retrieved source content to ground the answer.

**Prompt Template:**
'The task was attempted multiple times but could not be completed with grounded accuracy.'

TASK: 
{task}

SOURCE: 
{source_text}

STATUS: Unanswerable based on available context

Recommended Action: Flag for human review or return "insufficient data" 