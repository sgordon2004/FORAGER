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

# === FINAL RESULT COLOR CODING DISPLAY LOGIC ===
def get_color_for_value(label_type, value):
    if label_type == "confidence":
        return {
            "High": "#2ECC71",      # Green
            "Medium": "#F1C40F",    # Yellow
            "Low": "#E67E22",       # Orange
            "Zero": "#E74C3C",      # Red
        }.get(value, "#BDC3C7")     # Default grey

    if label_type == "bs":
        return {
            "Supported": "#2ECC71",
            "Unsupported": "#E74C3C",
            "Contradicted": "#8E44AD",
        }.get(value, "#BDC3C7")

    if label_type == "similarity":
        try:
            score = float(value)
            if score >= 0.7:
                return "#2ECC71"
            elif score >= 0.4:
                return "#F1C40F"
            else:
                return "#E74C3C"
        except:
            return "#BDC3C7"  # default gray for "N/A"

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
                    from ingestor import clean_html, html_text_dir, json_dir
                    # Temporarily save uploaded file
                    html_upload_dir = base_dir / "htmls"
                    input_path = html_upload_dir / file.name
                    html_upload_dir.mkdir(parents=True, exist_ok=True)
                    with open(input_path, "wb") as f:
                        f.write(file_bytes)
                    clean_html(input_path, html_text_dir, json_dir)

                elif file_ext == "pdf":
                    # Extract text from PDF
                    from extractor import extract_pdf
                    from ingestor import dump_pdf_text
                    pdf_upload_dir = base_dir / "pdfs"
                    input_path = pdf_upload_dir / file.name
                    pdf_upload_dir.mkdir(parents=True, exist_ok=True)
                    with open(input_path, "wb") as f:
                        f.write(file_bytes)
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
            # Initialize FAISS with the new chunks - this will hopefully be the only embedder
            # object that needs to be created
            from FORAGER.embedder import FAISSEmbedder
            embedder = FAISSEmbedder.create_default()
            embedder.initialize_faiss()
            st.session_state["embedder"] = embedder
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
            embedder = st.session_state.get("embedder")
            if embedder is None:
                st.warning("⚠️ Please process documents first to initialize the embedder.")
            else:
                from test_pipeline import full_forager_pipeline
                # Run the first portion of the FORAGER pipeline (function name misleading)
                answer, claim_eval = full_forager_pipeline(embedder, user_question)
                # st.markdown(f"full_forager_pipeline() returned `claim_eval`: {claim_eval}")
                st.session_state["answer"] = answer
                st.session_state["claim_eval"] = claim_eval

                # Display the summary cards
                answer = st.session_state.get("answer", "❓ No answer available")
                claim_eval = st.session_state.get("claim_eval", {})

                if claim_eval:
                    final_claim, info = list(claim_eval.items())[0]
                    label = info.get("label", "N/A")
                    confidence = info.get("confidence", "N/A")
                    similarity = info.get("similarity", "N/A")

                    st.markdown("## 🧾 Answer Summary")

                    # Dynamic tag colors
                    bs_color = get_color_for_value("bs", label)
                    conf_color = get_color_for_value("confidence", confidence)
                    sim_color = get_color_for_value("similarity", similarity)

                    # === Tag cards row ===
                    col1, col2, col3 = st.columns(3)

                    tag_style = """
                        background-color: #2c2f33;
                        padding: 15px;
                        border-radius: 10px;
                        box-shadow: 0 0 8px rgba(0,0,0,0.15);
                        margin-bottom: 10px;
                        color: #f5f5f5;
                        text-align: center;
                        transition: all 0.3s ease;
                    """

                    with col1:
                        st.markdown(f"""
                            <div style="{tag_style}">
                                <h4 style="color:{bs_color};">🧪 BS Label</h4>
                                <p><b>{label}</b></p>
                            </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                            <div style="{tag_style}">
                                <h4 style="color:{conf_color};">🔐 Confidence</h4>
                                <p><b>{confidence}</b></p>
                            </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
                            <div style="{tag_style}">
                                <h4 style="color:{sim_color};">📈 Similarity</h4>
                                <p><b>{similarity}</b></p>
                            </div>
                        """, unsafe_allow_html=True)

                    # === Final Claim full-width card ===
                    st.markdown(f"""
                        <div style="background-color: #2c2f33;
                                    padding: 20px;
                                    border-radius: 10px;
                                    box-shadow: 0 0 10px rgba(0,0,0,0.2);
                                    margin-top: 10px;
                                    margin-bottom: 20px;
                                    color: #f5f5f5;">
                            <h4 style="color:#9be7ff;">📝 Final Claim</h4>
                            <p style="font-size: 14px;">{answer}</p>
                        </div>
                    """, unsafe_allow_html=True)
            
                time.sleep(1)

                # Run Prompt Locked Loop as long as question was not marked as irrelevant
                if not (answer == "This question is unrelated to the documents in the knowledge base and cannot be answered using them."):
                    from pll_controller import prompt_locked_loop
                    status_placeholder.info("💾 Initializing Prompt Locked Loop...")
                    pll_logs = prompt_locked_loop(embedder, user_question, claim_eval, max_retry=3)
                    st.session_state["pll_logs"] = pll_logs

        # TODO: Move this to run after the final answer is locked.
        # status_placeholder.success("🎉 Full pipeline completed successfully!")
        # st.balloons()

# Tab 2: Knowledge Base Management
with tab_knowledge_base:
    st.header("📚 Knowledge Base Management")
    
    html_upload_dir = base_dir / "htmls"
    pdf_upload_dir = base_dir / "pdfs"

    html_upload_dir.mkdir(parents=True, exist_ok=True)
    pdf_upload_dir.mkdir(parents=True, exist_ok=True)

    st.markdown("### 📂 Existing Uploaded Documents")

    # Show HTML files
    html_files = list(html_upload_dir.glob("*.html"))
    if html_files:
        st.markdown("#### 🟣 HTML Files")
        for file_path in html_files:
            with st.expander(f"{file_path.name}"):
                st.code(file_path.read_text(encoding="utf-8")[:1000] + "..." if file_path.stat().st_size > 1000 else file_path.read_text(encoding="utf-8"), language="html")
                if st.button(f"🗑️ Delete {file_path.name}"):
                    file_path.unlink()
                    st.success(f"Deleted {file_path.name}")
                    st.experimental_rerun()

    # Show PDF files
    pdf_files = list(pdf_upload_dir.glob("*.pdf"))
    if pdf_files:
        st.markdown("#### 🔵 PDF Files")
        for file_path in pdf_files:
            with st.expander(f"{file_path.name}"):
                st.write(f"Size: {file_path.stat().st_size / 1024:.2f} KB")
                if st.button(f"🗑️ Delete {file_path.name}"):
                    file_path.unlink()
                    st.success(f"Deleted {file_path.name}")
                    st.experimental_rerun()

    st.markdown("---")
    st.markdown("### ➕ Upload More Files")
    more_files = st.file_uploader("Upload additional documents (PDF or HTML)", type=["pdf", "html"], accept_multiple_files=True, key="additional_uploads")

    if more_files:
        for file in more_files:
            file_ext = file.name.split(".")[-1].lower()
            file_bytes = file.read()
            if file_ext == "html":
                save_path = html_upload_dir / file.name
            elif file_ext == "pdf":
                save_path = pdf_upload_dir / file.name
            else:
                st.warning(f"Unsupported file type: {file.name}")
                continue

            with open(save_path, "wb") as f:
                f.write(file_bytes)
            st.success(f"✅ Uploaded {file.name}")
        st.experimental_rerun()

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
                    formatted_chunk = chunk["text"].replace("\n", "\n> ")
                    st.markdown(f"> **Chunk {idx}:**  \n> {formatted_chunk}")
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


