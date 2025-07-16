import streamlit as st
import pdfplumber
from bs4 import BeautifulSoup
import os
import requests
from dotenv import load_dotenv
import json
import pandas as pd
import datetime
from fpdf import FPDF
from io import BytesIO
import time
import sys
import os

# Add the FORAGER directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Importing backend functions
from bs import detect_bs
from confidence import check_confidence
from runner import get_llm_response
from embedder import search_database

# Load environment variable
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama3-8b-8192"

# === Confidence Checker Placeholder ===
# def confidence_checker(eval_label, similarity_score):
#     if eval_label == "Supported" and similarity_score >= 0.7:
#         return "High Confidence"
#     elif eval_label == "Supported" and similarity_score < 0.7:
#         return "Medium Confidence"
#     elif eval_label == "Unsupported":
#         return "Low Confidence"
#     elif eval_label == "Contradicted":
#         return "Zero Confidence"
#     return "Unknown"

# === Placeholder mock function for evaluation ===
# def mock_evaluate_answer(answer, doc_chunks):
#     eval_label = "Supported"
#     similarity_score = 0.76
#     confidence_label = confidence_checker(eval_label, similarity_score)
#     return {
#         "bs_label": eval_label,
#         "similarity_score": similarity_score,
#         "confidence_label": confidence_label
#     }

# === Placeholder mock function for source chunks ===
# def mock_retrieve_chunks(doc_text, question):
#     return [
#         "Chunk 1: This section discusses the financial trends from 2020 to 2023.",
#         "Chunk 2: The document outlines how the budget increased due to policy shifts."
#     ]

# === RAG Simulation Function ===
# def mock_rag_pipeline(text, question):
#     chunks = mock_retrieve_chunks(text, question)
#     prompt = f"""Given the following document:\n\n{text}\n\nAnswer the question:\n{question}"""
#     payload = {
#         "model": MODEL_NAME,
#         "messages": [
#             {"role": "system", "content": "You are an expert assistant for document QA."},
#             {"role": "user", "content": prompt}
#         ],
#         "temperature": 0.2
#     }
#     headers = {
#         "Authorization": f"Bearer {GROQ_API_KEY}",
#         "Content-Type": "application/json"
#     }
# 
    # response = requests.post(GROQ_ENDPOINT, json=payload, headers=headers)
    # response.raise_for_status()
    # result = response.json()
    # answer = result["choices"][0]["message"]["content"]
    # return answer, chunks, result

# === PLL PATH MOCK GENERATOR ===
def generate_dynamic_pll_path(final_answer, final_bs, final_conf, final_similarity):
    pll_attempts = [
        {"answer": "Initial hallucinated response.", "bs": "Unsupported", "confidence": "Low", "similarity_score": 0.41, "status": "Retry"},
        {"answer": "Conflicting answer with sources.", "bs": "Contradicted", "confidence": "Zero", "similarity_score": 0.35, "status": "Rejected"},
        {"answer": final_answer, "bs": final_bs, "confidence": final_conf, "similarity_score": final_similarity, "status": "Locked"}
    ]
    return pll_attempts

# === PLL Renderer ===
def render_pll_history(attempts):
    st.markdown("## 🔁 PLL Evaluation History")
    for i, attempt in enumerate(attempts, 1):
        with st.container():
            st.markdown(f"### Attempt {i}: `{attempt['status']}`")
            st.markdown(f"**🧠 BS Label:** `{attempt['bs']}`")
            st.markdown(f"**📊 Confidence:** `{attempt['confidence']}`")
            st.markdown(f"**📐 Similarity Score:** `{attempt['similarity_score']:.2f}`")
            st.markdown(f"**📝 Answer:**\n> {attempt['answer']}")
            st.markdown("---")


# === Streamlit App ===


st.set_page_config(page_title="FORAGER RAG UI", layout="wide")
st.title("📄🔍 FORAGER")

# === Sidebar Developer Mode Toggle ===
st.sidebar.markdown("## ⚙️ Developer Settings")
dev_mode = st.sidebar.toggle("🛠 Enable Developer Mode", value=False)

if dev_mode:
    st.sidebar.markdown("### 🔧 Debug Tool Visibility")

    show_all = st.sidebar.toggle("📋 Show All Debug Sections")

    # Subtoggles
    show_llm_json = st.sidebar.toggle("🧾 Raw LLM Output", value=show_all)
    show_eval_json = st.sidebar.toggle("🧪 Evaluation JSON", value=show_all)
    show_traceback = st.sidebar.toggle("📂 Document Tracebacks", value=show_all)
    show_chunks = st.sidebar.toggle("🔗 Retrieved Chunks", value=show_all)
    show_logs = st.sidebar.toggle("📋 Show PLL Logs", value=show_all)

    # Manual override button
    st.sidebar.markdown("### 🔁 Manual Override")
    if st.sidebar.button("Force Rerun of LLM + Evaluation"):
        st.session_state["force_rerun"] = True

# Widen page with custom CSS
st.markdown("""
    <style>
        .block-container { padding-left: 2rem; padding-right: 2rem; }
        .main { max-width: 95vw; }
    </style>
""", unsafe_allow_html=True)

# === Upload and Read Multiple Files ===
uploaded_files = st.file_uploader(
    "Upload documents (PDF, HTML, or Markdown)",
    type=["pdf", "html", "md"],
    accept_multiple_files=True
)

all_text = ""
if uploaded_files:
    st.markdown("### 📂 Uploaded Files")
    for file in uploaded_files:
        content = file.read().decode("utf-8", errors="ignore")
        all_text += content + "\n\n"
        st.write(f"- {file.name}")

# === Question Input ===
user_question = st.text_input("Ask a question based on the uploaded document(s):")
if st.button("Submit"):
    st.session_state["submitted"] = True

# === Answer + Evaluation Display ===
if uploaded_files and user_question and (st.session_state.get("submitted", False) or st.session_state.get("force_rerun", False)):

    with st.spinner("⚙️ Processing with mock RAG pipeline..."):
        try:
            # 1. Get answer from LLM
            llm_json_response = get_llm_response(user_question)
            answer = llm_json_response

            # 2. Retrieve top-k chunks
            retrieved_chunks = search_database(answer, num_vectors=5)
            chunk_texts = [c["text"] for c in retrieved_chunks]

            # 3. Calculate similarity
            similarity_score = sum(c["score"] for c in retrieved_chunks) / len(retrieved_chunks)

            # 4. BS detection
            bs_label = detect_bs(answer, chunk_texts)

            # 5. Confidence level
            confidence_label = confidence_checker(bs_label, similarity_score)

            # 6. Create eval_results (to keep rest of your code the same)
            eval_results = {
                "bs_label": bs_label,
                "similarity_score": similarity_score,
                "confidence_label": confidence_label
            }
            chunks = chunk_texts  # So you can reuse the variable for display




            # Reset rerun trigger
            st.session_state["force_rerun"] = False

            bs_label = eval_results["bs_label"]
            similarity_score = eval_results["similarity_score"]
            confidence_label = eval_results["confidence_label"]

            # Display LLM Answer
            st.success("✅ LLM Answer")
            st.write(answer)

            # Display Source Chunks
            st.markdown("### 🔗 Source Chunks Used")
            for chunk in chunks:
                st.code(chunk.strip(), language="markdown")

            # Dashboard Tags
            st.markdown("### 🧠 Evaluation Dashboard")
            tag_style = "display:inline-block;padding:0.3em 0.8em;font-size:0.85em;font-weight:600;color:white;border-radius:1em;margin-right:1em;"

            bs_colors = {"Supported": "#3CB371", "Unsupported": "#FFD700", "Contradicted": "#FF6347"}
            conf_colors = {"High Confidence": "#3CB371", "Medium Confidence": "#FFD700", "Low Confidence": "#FFA07A", "Zero Confidence": "#FF6347"}

            bs_color = bs_colors.get(bs_label, "#808080")
            conf_color = conf_colors.get(confidence_label, "#808080")

            st.markdown(f"**BS Label:** <span style='{tag_style} background-color:{bs_color}'>{bs_label}</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** <span style='{tag_style} background-color:{conf_color}'>{confidence_label}</span>", unsafe_allow_html=True)

            # PLL Visualization
            pll_attempts = generate_dynamic_pll_path(answer, bs_label, confidence_label, similarity_score)
            render_pll_history(pll_attempts)

            # Transform PLL attempts into log entries (based on mock pll dynamic answer generator)
            mock_pll_logs = []
            for i, attempt in enumerate(pll_attempts, start=1):
                log_entry = {
                    "step": f"Attempt {i}",
                    "action": attempt["status"],
                    "bs_label": attempt["bs"],
                    "confidence": attempt["confidence"],
                    "similarity": attempt["similarity_score"]
                }
                mock_pll_logs.append(log_entry)
            
            # === Store export log in session state ===
            if 'log_export_ready' not in st.session_state:
                st.session_state['log_export_ready'] = False

            # Metadata setup
            log_export = {
                "question": user_question,
                "timestamp": datetime.datetime.now().isoformat(),
                "pll_attempts": mock_pll_logs
            }

            st.session_state['log_export'] = log_export
            st.session_state['log_export_ready'] = True

            # === Export Feature UI ===
            st.markdown("### 📥 Export PLL Logs")

            if st.session_state['log_export_ready']:
                # Track selection persistently
                if "export_format" not in st.session_state:
                    st.session_state["export_format"] = "JSON"

                # Always render selectbox (outside the init block)
                selected_format = st.selectbox("Choose download format", ["JSON", "TXT", "CSV", "PDF"])
                if selected_format != st.session_state["export_format"]:
                    st.session_state["export_format"] = selected_format
                    st.rerun()

                export_format = st.session_state["export_format"]

                # JSON
                if export_format == "JSON":
                    log_json_str = json.dumps(st.session_state['log_export'], indent=2)
                    st.download_button(
                        label="📄 Download as JSON",
                        data=log_json_str,
                        file_name="pll_log.json",
                        mime="application/json"
                    )

                # TXT
                elif export_format == "TXT":
                    log_txt_str = f"Question: {log_export['question']}\nTimestamp: {log_export['timestamp']}\n\n"
                    for log in log_export["pll_attempts"]:
                        log_txt_str += f"--- {log['step']} ({log['action']}) ---\n"
                        log_txt_str += f"BS Label: {log['bs_label']}\n"
                        log_txt_str += f"Confidence: {log['confidence']}\n"
                        log_txt_str += f"Similarity: {log['similarity']}\n\n"
                    st.download_button(
                        label="📄 Download as TXT",
                        data=log_txt_str,
                        file_name="pll_log.txt",
                        mime="text/plain"
                    )

                # CSV
                elif export_format == "CSV":
                    df = pd.DataFrame(log_export["pll_attempts"])
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="📄 Download as CSV",
                        data=csv_data,
                        file_name="pll_log.csv",
                        mime="text/csv"
                    )

                # PDF
                elif export_format == "PDF":
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt="FORAGER PLL Log", ln=True, align="C")
                    pdf.ln(10)
                    pdf.multi_cell(0, 10, txt=f"Question: {log_export['question']}")
                    pdf.multi_cell(0, 10, txt=f"Timestamp: {log_export['timestamp']}")
                    pdf.ln(5)

                    for log in log_export["pll_attempts"]:
                        pdf.multi_cell(0, 10, txt=(
                            f"{log['step']} ({log['action']})\n"
                            f"BS Label: {log['bs_label']}\n"
                            f"Confidence: {log['confidence']}\n"
                            f"Similarity: {log['similarity']}\n"
                        ))
                        pdf.ln(2)

                    pdf_output_str = pdf.output(dest='S').encode('latin-1')
                    pdf_bytes = BytesIO(pdf_output_str)

                    st.download_button(
                        label="📄 Download as PDF",
                        data=pdf_bytes,
                        file_name="pll_log.pdf",
                        mime="application/pdf"
                    )

            # === User Feedback Section ===
            st.markdown("### 📝 User Feedback")
            col1, col2 = st.columns(2)
            with col1:
                feedback_rating = st.radio("Was this answer helpful?", ["👍 Yes", "👎 No"], key="helpful_rating")
            with col2:
                feedback_conf = st.slider("Confidence in this answer:", 0, 100, 50, key="user_conf")

            user_comments = st.text_area("Any comments or corrections?", placeholder="Let us know what was good or what went wrong...")
            if st.button("Submit Feedback"):
                st.success("✅ Thanks for your feedback!")
                print({
                    "user_question": user_question,
                    "llm_answer": answer,
                    "bs_label": bs_label,
                    "confidence": confidence_label,
                    "similarity_score": similarity_score,
                    "user_feedback": feedback_rating,
                    "user_confidence": feedback_conf,
                    "user_comments": user_comments
                })
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
# === Dev Mode Debug Outputs ===
if dev_mode:
    st.markdown("### 🛠 Dev's Debugging Tools")

    # Raw LLM Response
    if show_llm_json:
        st.markdown("#### 🧾 Raw LLM Output")
        try:
            st.json(llm_json_response)
        except Exception as e:
            st.warning(f"⚠️ LLM response not available or improperly formatted: {e}")

    # Evaluation JSON
    if show_eval_json:
        st.markdown("#### 🧪 Evaluation JSON")
        try:
            st.json(eval_results)
        except Exception as e:
            st.warning(f"⚠️ No evaluation results to display: {e}")

    # Document Traceback
    if show_traceback:
        st.markdown("#### 📂 Uploaded Document Traceback")
        if uploaded_files:
            for file in uploaded_files:
                st.markdown(f"- **Filename:** `{file.name}`")
        else:
            st.info("No documents uploaded.")

    # Retrieved Chunks
    if show_chunks:
        st.markdown("#### 🔗 Retrieved Chunks")
        if 'chunks' in locals() and chunks:
            for i, chunk in enumerate(chunks):
                st.code(chunk.strip(), language="markdown")
        else:
            st.info("No chunks retrieved yet.")
    
    # PLL Logs
    if show_logs and 'mock_pll_logs' in locals():
        st.markdown("### 📜 PLL Event Logs")
        for log in mock_pll_logs:
            with st.expander(f"{log['step']}: {log['action']}", expanded=False):
                st.markdown(f"- **BS Label:** `{log['bs_label']}`")
                st.markdown(f"- **Confidence:** `{log['confidence']}`")
                st.markdown(f"- **Similarity Score:** `{log['similarity']}`")

if st.button("Reset"):
    st.session_state["submitted"] = False