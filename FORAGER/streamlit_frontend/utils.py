import json

def load_json(path):
    with open(path) as f:
        return json.load(f)

def get_all_questions(llm_data):
    all_qs = {}
    qid = 1
    for batch_name, batch in llm_data.items():
        for q in batch["questions"]:
            all_qs[f"Q{qid}"] = {
                "batch": batch_name,
                "input": q["input"],
                "options": q["options"],
                "answer": q["answer"]
            }
            qid += 1
    return all_qs
