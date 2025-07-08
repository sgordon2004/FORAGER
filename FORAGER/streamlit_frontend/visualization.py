import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import os

# Load incorrect questions
with open("../data/incorrect_questions/incorrect_questions.json") as f:
    incorrect_raw = json.load(f)

# Convert all incorrect keys to consistent QID format: "Q2", "Q17", etc.
incorrect_qids = {f"Q{int(k)}" for k in incorrect_raw if k.isdigit()}

# Load all LLM response rounds
all_rows = []

for round_num in range(4):
    file_path = f"../data/llm_responses/round_{round_num}_responses.json"
    try:
        with open(file_path) as f:
            data = json.load(f)
            st.success(f"✅ Loaded: {file_path}")
            if data:
                st.write(f"Sample from round {round_num}:", list(data.values())[0])
    except FileNotFoundError:
        st.warning(f"⚠️ File not found: {file_path}")
        continue
    except Exception as e:
        st.error(f"❌ Failed to load {file_path}: {e}")
        continue

    q_counter = 1  # Index used for assigning qid if missing
    for key, batch in data.items():
        # Round 0 format
        if isinstance(batch, dict) and "questions" in batch:
            for q in batch["questions"]:
                qid = f"Q{q_counter}"
                all_rows.append({
                    "index": q_counter,
                    "round": round_num,
                    "qid": qid,
                    "input": q.get("input") or q.get("question"),
                    "is_incorrect": qid in incorrect_qids
                })
                q_counter += 1

        # Rounds 1–3
        elif isinstance(batch, dict) and "question" in batch:
            qid = key  # Use "Q2", "Q17", etc.
            all_rows.append({
                "index": q_counter,
                "round": round_num,
                "qid": qid,
                "input": batch.get("input") or batch.get("question"),
                "is_incorrect": qid in incorrect_qids
            })
            q_counter += 1

        else:
            st.warning(f"⚠️ Skipping malformed batch: {batch}")

# Convert to DataFrame
df = pd.DataFrame(all_rows)

# displays rounds (mainly for checking that all rounds are going thru)
st.write("Rounds present in dataset:", df["round"].unique())
st.write("Shape of DataFrame:", df.shape)
st.write("Unique QIDs by round:")
st.dataframe(df.groupby("round")["qid"].unique().reset_index())

# table
st.subheader("Questions Across Rounds")
st.dataframe(df.head(20))

# dot plot of retry attempts per question
st.subheader("Retry Attempts per Question (Incorrect in Red)")
fig, ax = plt.subplots(figsize=(10, 6))

unique_qids = df["qid"].unique()

for i, qid in enumerate(unique_qids):
    group = df[df["qid"] == qid]
    latest_round = group["round"].max()
    still_incorrect = group[group["round"] == latest_round]["is_incorrect"].any()
    color = "red" if still_incorrect else "blue"
    ax.plot(group["round"], [i] * len(group), 'o-', color=color)

ax.set_xlabel("FORAGER Round")
ax.set_ylabel("Question ID")
ax.set_title("Retry Attempts per Question (Incorrect in Red)")
ax.set_yticks(range(len(unique_qids)))
ax.set_yticklabels(unique_qids)

st.pyplot(fig)