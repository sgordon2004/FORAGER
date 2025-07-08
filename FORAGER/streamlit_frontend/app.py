import streamlit as st
import os
import json

# load JSON
def load_json(path):
    with open(path) as f:
        return json.load(f)

# paths
RESPONSES_DIR = "FORAGER/data/llm_responses"
PROMPTS_DIR = "FORAGER/data/prompt_history"
INCORRECT_PATH = "FORAGER/data/incorrect_questions/incorrect_questions.json"
RESTRUCTURED_PATH = "FORAGER/data/new_prompts/clean_restructured_prompts.json"

# select round
round_files = sorted([f for f in os.listdir(RESPONSES_DIR) if f.endswith(".json")])
round_options = [int(f.split("_")[1]) for f in round_files if f.startswith("round_")]
selected_round = st.sidebar.selectbox("Select Round", sorted(round_options), index=len(round_options)-1)

# file paths
response_file = os.path.join(RESPONSES_DIR, f"round_{selected_round}_responses.json")
if selected_round == 0:
    prompt_file = os.path.join(PROMPTS_DIR, "prompt_history_round_0.json")
else:
    prompt_file = os.path.join(PROMPTS_DIR, f"prompt_history_round_{selected_round - 1}.json")
incorrect_file = INCORRECT_PATH
restructured_file = RESTRUCTURED_PATH

# load data
llm_data = load_json(response_file)
prompt_history = load_json(prompt_file)
incorrect_qs = load_json(incorrect_file)
restructured = load_json(restructured_file)

# extract all questions from batches
def get_all_questions(llm_data):
    all_qs = {}
    for batch_id, data in llm_data.items():
        if "questions" not in data:
            continue
        for i, q in enumerate(data["questions"]):
            qid = f"{batch_id}_Q{i+1}"
            all_qs[qid] = {
                "batch": batch_id,
                "index": i,
                "question": q["input"],
                "options": q["options"],
                "answer": q["answer"]
            }
    return all_qs

all_questions = get_all_questions(llm_data)

# display all questions in the round
st.title(f"Round {selected_round} — All Questions")
for qid, q in all_questions.items():
    with st.expander(f"{qid}: {q['question']}"):
        st.markdown("### Options")
        st.write(q["options"])

        st.markdown("### LLM Answer")
        st.success(q["answer"])

        st.markdown("### Original Prompt")
        st.code(prompt_history.get(qid, "Not found."), language="text")

        st.markdown("### Rephrased Prompt")
        if qid in restructured:
            formatted = f"{restructured[qid]['question']}\nOptions:\n" + "\n".join(restructured[qid]['options'])
            st.code(formatted, language="text")
        else:
            st.warning("No restructured prompt found.")

        if qid in incorrect_qs:
            st.error("❌ Incorrect in last evaluation")
        else:
            st.success("✅ Correct in last evaluation")