## Creating a sanity checker for incorrect_questions.json 
## to validate against original input 

import json

# Load incorrect responses (question number, response)
with open("FORAGER/data/incorrect_questions/incorrect_questions.json", "r") as f:
    incorrect_raw = json.load(f)

# Load 4_distractors dataset
with open("FORAGER/data/4_distractors.json", "r") as f:
    distractors_4 = json.load(f)

# Build question map -> question number, prompt, and options 
question_map = {}
for idx, item in enumerate(distractors_4, start=1):
    question_map[str(idx)] = {
        "input": item["input"].strip(),
        "options": list(item["target_scores"].keys())
    }

# Evaluate responses 
results = []

for qid, response in incorrect_raw.items():
    if qid not in question_map:
        print(f"Question ID {qid} not found in 4_distractors.json, skipping.")
        continue 

    prompt_data = question_map[qid]
    prompt_text = prompt_data["input"]
    valid_options = set(opt.strip().lower() for opt in prompt_data["options"])
    response_clean = response.strip().lower()

    is_valid = response_clean in valid_options 

    results.append({
        "question_number": qid,
        "input": prompt_text,
        "response": response,
        "is_valid": is_valid,
        "reason": "Valid response" if is_valid else "Hallucinated response"
    })

#Print results: 
for r in results: 
    print(f"Question #{r['question_number']}")
    print(f"Prompt: {r['input']}")
    print(f"Response: {r['response']}")
    print(f"Valid? {r['is_valid']} - {r['reason']}")
    print("-" * 60)

with open("FORAGER/data/evaluated_response.json", "w") as outfile: 
    json.dump(results, outfile, indent=2)
