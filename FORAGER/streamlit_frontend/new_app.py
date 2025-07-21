import streamlit as st
import os
from dotenv import load_dotenv
import time
import sys
import os
from pathlib import Path

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add FORAGER to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables, set up Groq API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
MODEL_NAME = "llama3-8b-8192"

# === STREAMLIT FRONTEND SETUP === #
# Connect .css file for styling
with open("FORAGER/streamlit_frontend/forager_styles.css", "r") as css_file:
    css = css_file.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Set page config (title, icon, layout)
st.set_page_config(page_title="FORAGER RAG UI", layout="wide")
st.title("📄🔍 FORAGER")

# Create 4 tabs for the app
tab_chat, tab_knowledge_base, tab_claims, tab_metrics, tab_logs = st.tabs(
    ["💬 Chat & Answer", "📚 Knowledge Base", "📑 Claims Breakdown", "📊 Metrics & Performance", "📜 PLL Logs"])

# Create sidebar for status updates
with st.sidebar:
    st.markdown("### 🚦 Pipeline Status")
    status_placeholder = st.empty()

# Tab 1: Chat & Final Answer
with tab_chat:
    st.header("💬 Chat & Final Answer")

    # File upload section
    uploaded_files = st.file_uploader(
        "Upload documents (PDF or HTML)",
        type=["pdf", "html"],
        accept_multiple_files=True
    )

    # Paths to uploaded files
    base_dir = Path("FORAGER_corpus/heterogenous_integration")
    html_dir = base_dir / "html"
    pdf_dir = base_dir / "pdf"

    # Begin document processing when button is clicked
    if st.button("Process Document(s)"):
        from ingestor import extract_pdf, dump_pdf_text

        if not uploaded_files:
            status_placeholder.warning("⚠️ No documents uploaded.")
        else:
            status_placeholder.info("📄 Starting text extraction...")
            # Extract text from uploaded files
            for file in uploaded_files:
                # Isolate file extension and read bytes
                file_ext = file.name.split(".")[-1].lower()
                file_bytes = file.read()

                if file_ext == "html":
                    save_path = html_dir / file.name
                elif file_ext == "pdf":
                    # Extract text from PDF
                    text = extract_pdf(file.name)
                    # Save text to pdf_text + create .json metadata file
                    dump_pdf_text(file.name, text)
                else:
                    st.warning(f"Unsupported file type: {file.name}")
                    continue
            time.sleep(1)
            status_placeholder.success("✅ Text extraction complete!")

            status_placeholder.info("🔗 Chunking documents...")
            from chunker import main as chunker_main
            # Chunk all the .JSON metadata files, chunk them, and save to chunks.jsonl
            chunker_main()
            time.sleep(1)
            status_placeholder.success("✅ Chunking complete!")

            status_placeholder.info("💾 Initializing FAISS...")
            # Initialize FAISS with the new chunks
            from FORAGER.embedder import initialize_faiss
            initialize_faiss()
            time.sleep(1)
            status_placeholder.success("✅ FAISS initialized!")

            st.session_state["documents_processed"] = True
            status_placeholder.success("✅ Documents processed!")
    # Run question process only if documents have been processed
    if st.session_state.get("documents_processed"):
        # Question input section
        st.markdown("### 💬 Ask a Question")
        user_question = st.text_input("Query the knowledge base: ")

        if st.button("Submit Question") and user_question:
            st.session_state["submitted"] = True
            status_placeholder.info("🤖 Generating answer via LLM...")
            from test_pipeline import full_forager_pipeline
            # Run the first portion of the FORAGER pipeline (function name misleading)
            answer, claim_eval = full_forager_pipeline(user_question)
            # st.markdown(f"full_forager_pipeline() returned `claim_eval`: {claim_eval}")
            st.session_state["answer"] = answer
            st.session_state["claim_eval"] = claim_eval

        
            # Display the first (full) response
            answer = st.session_state.get("answer", "❓ No answer available")
            st.markdown(f"**📝 Final Claim:**\n> {answer}")
            time.sleep(1)

            # Run Prompt Locked Loop
            from pll_controller import prompt_locked_loop
            status_placeholder.info("💾 Initializing Prompt Locked Loop...")
            pll_logs = prompt_locked_loop(user_question, claim_eval, max_retry=3)
            st.session_state["pll_logs"] = pll_logs

        # TODO: Move this to run after the final answer is locked.
        # status_placeholder.success("🎉 Full pipeline completed successfully!")
        # st.balloons()

# Tab 2: Knowledge Base Management
with tab_knowledge_base:
    st.header("📚 Knowledge Base Management")

# Tab 3: Steb-by-step claims breakdown
with tab_claims:
    st.header("📑 Claims Breakdown")

    final_claims = st.session_state.get("claim_eval", {})

    if not final_claims:
        st.info("No claims available. Submit a question in the Chat tab to generate claims.")
    else:
        for claim, info in final_claims.items():
            label = info.get("label", "N/A")
            confidence = info.get("confidence", "N/A")
            supporting_chunks = info.get("supporting_chunks", [])

            st.markdown(f"### 📝 Claim: {claim}")
            st.markdown(f"- **Evaluation:** `{label}`")
            st.markdown(f"- **Confidence:** `{confidence}`")

            with st.expander("📜 Supporting Chunks"):
                for idx, chunk in enumerate(supporting_chunks, 1):
                    st.markdown(f"**Chunk {idx}:**\n> {chunk['text']}")
# Tab 4: Metrics & Performance
with tab_metrics:
    st.header("📊 Metrics & Performance")

    claim_eval = st.session_state.get("claim_eval", {})
    pll_logs = st.session_state.get("pll_logs", [])

    total_claims = len(claim_eval)
    total_pll_rounds = len(pll_logs)

    st.metric(label="Total Claims Generated", value=total_claims)
    st.metric(label="Total PLL Rounds Executed", value=total_pll_rounds)

    # Count labels
    label_counts = {}
    for info in claim_eval.values():
        label = info.get("label", "N/A")
        label_counts[label] = label_counts.get(label, 0) + 1

    st.subheader("📊 CLaim Label Distribution")
    st.bar_chart(label_counts)

    # Optional: Show PLL decision paths
    st.subheader("🪵 PLL Rounds Breakdown")
    for round_log in pll_logs:
        st.markdown(f"**PLL Round {round_log['pll_round']}** — {len(round_log['claims'])} claims processed")

# Tab 5: PLL Logs + Developer's View
with tab_logs:
    st.header("🪵 PLL Logs")

    pll_logs = st.session_state.get("pll_logs", [])

    if not pll_logs:
        st.info("No PLL logs available. Submit a question in the Chat tab to generate logs.")
    else:
        for round_log in pll_logs:
            with st.expander(f"PLL Round {round_log['pll_round']}"):
                if not round_log["claims"]:
                    st.info("No claims were processed this round.")
                    continue
            
                for claim_info in round_log["claims"]:
                    claim = claim_info["claim"]
                    decision = claim_info["pll_decision"]
                    reason = claim_info["reason"]

                    st.markdown(f"""
                    - **Claim:** {claim}
                    - **Decision:** `{decision}`
                    - **Reason:** {reason}         
                    """)
# === INPUT INGESTION PROCESS === #


