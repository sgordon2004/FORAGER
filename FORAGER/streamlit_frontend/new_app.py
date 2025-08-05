import streamlit as st
import os
from dotenv import load_dotenv
import time
import sys
import os
import base64
import plotly.express as px
import pandas as pd
from pathlib import Path
from PIL import Image
from PIL import ImageOps
from bs4 import BeautifulSoup
from streamlit_pdf_viewer import pdf_viewer

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
# st.title("⚛︎ Fact-Oriented Responsible AI-Guided Engineering Research (FORAGER) ⚛︎")
st.markdown("<h1 style='font-size: 40px;'>⚛︎ Fact-Oriented Responsible AI-Guided Engineering Research ⚛︎</h1>", unsafe_allow_html=True)

# Create 4 tabs for the app
tab_chat, tab_knowledge_base, tab_claims, tab_metrics, tab_logs= st.tabs(
    ["💬 Ask Chat", "📚 Document Database", "📑 Claims Breakdown", "📉 Metrics & Visualizations", "📜 PLL Logs"])

# Sidebar for status updates
# with st.sidebar:
    # st.markdown("### 🚦 Pipeline Status")
    # status_placeholder = st.empty()

# Tab 1: Chat Tab (LLM Interaction)
with tab_chat:
    # === Custom CSS for pipeline status cards ===
    st.markdown("""
        <style>
        .status-card {
            background-color: #1e1e1e;
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 10px;
            font-size: 14px;
            color: #f5f5f5;
            display: flex;
            align-items: center;
            gap: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.4);
        }
        .status-icon {
            font-size: 18px;
        }
        </style>
    """, unsafe_allow_html=True)

    # === Logo ===
    logo = Image.open("FORAGER/streamlit_frontend/finallogo.png")
    space1, space2, space3 = st.columns(3)
    # with space1:
    #     st.image(booz_padded, use_container_width=True)
    with space2:
        st.image(logo, width=420)
    # with space3:
    #     st.image(allen_padded, use_container_width=True)
    st.markdown(
    "<h1 style='text-align:center; margin-top: 10px';>What's on your mind today?</h1>", 
    unsafe_allow_html=True)

    # === Instructional Steps ===
    st.markdown("""
    <div class="steps-container">
        <div class="step-card">
            <div class="step-title">Upload</div>
            <div class="step-description">Upload your PDFs, HTML, or TXT files to get started.</div>
        </div>
        <div class="step-card">
            <div class="step-title">Process</div>
            <div class="step-description">Extract and chunk content for search and retrieval.</div>
        </div>
        <div class="step-card">
            <div class="step-title">Ask</div>
            <div class="step-description">Submit a question and get an answer with supporting evidence.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # === Pipeline State Initialization ===
    if "pipeline_complete" not in st.session_state:
        st.session_state["pipeline_complete"] = False

    # === File Upload ===
    uploaded_files = None
    if not st.session_state["pipeline_complete"]:
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "html", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

    with st.container():
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        process_clicked = st.button("Process", key="process_button")
        st.markdown('</div>', unsafe_allow_html=True)

    # === Pipeline Status Below Uploader ===
    st.markdown("### 🚦 Pipeline Status")
    status_placeholder = st.empty()

    # Default idle state message
    if not st.session_state["pipeline_complete"]:
        status_placeholder.markdown(
            "<div class='status-card'><span class='status-icon'>📂</span>Upload documents to get started.</div>",
            unsafe_allow_html=True
        )

    # === Paths ===
    base_dir = Path("FORAGER_corpus/heterogenous_integration")
    html_upload_dir = base_dir / "htmls"
    pdf_upload_dir = base_dir / "pdfs"
    txt_upload_dir = base_dir / "txts"

    # === Process Pipeline ===
    if process_clicked:
        if not uploaded_files:
            status_placeholder.markdown("<div class='status-card'><span class='status-icon'>⚠️</span>No documents uploaded.</div>", unsafe_allow_html=True)
        else:
            st.session_state["pipeline_complete"] = False
            status_placeholder.markdown("<div class='status-card'><span class='status-icon'>📄</span>Starting text extraction...</div>", unsafe_allow_html=True)

            for file in uploaded_files:
                file_ext = file.name.split(".")[-1].lower()
                file_bytes = file.read()

                if file_ext == "html":
                    from ingestor import clean_html, html_text_dir, json_dir
                    input_path = html_upload_dir / file.name
                    html_upload_dir.mkdir(parents=True, exist_ok=True)
                    with open(input_path, "wb") as f:
                        f.write(file_bytes)
                    clean_html(input_path, html_text_dir, json_dir)

                elif file_ext == "pdf":
                    from extractor import extract_pdf
                    from ingestor import dump_pdf_text
                    input_path = pdf_upload_dir / file.name
                    pdf_upload_dir.mkdir(parents=True, exist_ok=True)
                    with open(input_path, "wb") as f:
                        f.write(file_bytes)
                    text = extract_pdf(file.name)
                    dump_pdf_text(file.name, text)

                elif file_ext == "txt":
                    from ingestor import extract_all_txt
                    input_path = txt_upload_dir / file.name
                    txt_upload_dir.mkdir(parents=True, exist_ok=True)
                    with open(input_path, "wb") as f:
                        f.write(file_bytes)
                    extract_all_txt()

            time.sleep(1)
            status_placeholder.markdown("<div class='status-card'><span class='status-icon'>✅</span>Text extraction complete!</div>", unsafe_allow_html=True)

            # Chunk documents
            status_placeholder.markdown("<div class='status-card'><span class='status-icon'>🔗</span>Chunking documents...</div>", unsafe_allow_html=True)
            from chunker import main as chunker_main
            chunker_main()
            time.sleep(1)
            status_placeholder.markdown("<div class='status-card'><span class='status-icon'>✅</span>Chunking complete!</div>", unsafe_allow_html=True)

            # Initialize FAISS
            status_placeholder.markdown("<div class='status-card'><span class='status-icon'>💾</span>Initializing FAISS...</div>", unsafe_allow_html=True)
            from FORAGER.embedder import FAISSEmbedder
            embedder = FAISSEmbedder.create_default()
            embedder.initialize_faiss()
            st.session_state["embedder"] = embedder
            time.sleep(1)
            status_placeholder.markdown("<div class='status-card'><span class='status-icon'>✅</span>FAISS initialized!</div>", unsafe_allow_html=True)

            st.session_state["documents_processed"] = True
            st.session_state["documents_ready"] = True
            st.session_state["pipeline_complete"] = True
            status_placeholder.markdown("<div class='status-card'><span class='status-icon'>🎉</span>Pipeline completed successfully!</div>", unsafe_allow_html=True)
            st.rerun()
    
    # === Question Input Section ===
    if st.session_state.get("documents_ready", False) and any([
        list(html_upload_dir.glob("*.html")),
        list(pdf_upload_dir.glob("*.pdf")),
        list(txt_upload_dir.glob("*.txt"))
    ]):
        st.markdown("### 💬 Ask a Question")
        user_question = st.text_input("Query the knowledge base:")

        if st.button("Submit Question") and user_question:
            st.session_state["submitted"] = True
            status_placeholder.markdown(
                "<div class='status-card'><span class='status-icon'>🤖</span> Generating answer via LLM...</div>",
                unsafe_allow_html=True
            )

            embedder = st.session_state.get("embedder")
            if embedder is None:
                from FORAGER.embedder import FAISSEmbedder
                embedder = FAISSEmbedder.create_default()
                try:
                    embedder.initialize_faiss()
                    st.session_state["embedder"] = embedder
                except Exception:
                    st.warning("⚠️ Could not initialize FAISS index. Please upload and process documents first.")
                    st.stop()

            # === Generate Claims and Evaluate ===
            else:
                from test_pipeline import generate_and_evaluate_claims
                answer, claim_eval = generate_and_evaluate_claims(embedder, user_question)
                st.session_state["answer"] = answer
                st.session_state["claim_eval"] = claim_eval

                answer = st.session_state.get("answer", "❓ No answer available")
                claim_eval = st.session_state.get("claim_eval", {})

                results = []

                for final_claim, info in claim_eval.items():
                    claim_label = info.get("label", "N/A")
                    claim_confidence = info.get("confidence", "N/A")
                    chunks = info.get("supporting_chunks", [])
                    scores = [doc.get("score", 0) for doc in chunks if "score" in doc]
                    claim_similarity = round(sum(scores) / len(scores), 3) if scores else "N/A"
                    results.append({
                        "claim": final_claim,
                        "bs_label": claim_label,
                        "similarity_score": claim_similarity,
                        "confidence": claim_confidence
                    })
                    print(f"Results dictionary: {results}")


                # Get stats for BS Label, Confidence Label, and Similarity Score cards
                label = ""
                confidence = ""
                similarity = 0

                if results:
                    # Get % Supported
                    supported_count = sum(1 for item in results if item["bs_label"] == "Supported")
                    total_count = len(results)
                    supported_percent = round((supported_count / total_count) * 100)
                    label = f"{supported_percent}%"

                    # Get % Confidence
                    high_conf_count = sum(1 for item in results if item["confidence"] == "High")
                    high_conf_percent = round((high_conf_count / total_count) * 100)
                    confidence = f"{high_conf_percent}%"

                    # Get average similarity score
                    total_score = sum(item["similarity_score"] for item in results)
                    average_score = round((total_score / len(results)) * 100)
                    similarity = f"{average_score}%"

                # Avoid errors if there are no results
                else:
                    label = "N/A"
                    confidence = "N/A"
                    similarity = 0


                # Answer summary and stat cards
                st.markdown("## 🧾 Answer Summary")

                # Dynamic tag colors
                bs_class = f"bs-{label}"
                conf_class = f"confidence-{confidence}"
                try:
                    sim_class = (
                        "similarity-high" if float(similarity) >= 70 else
                        "similarity-medium" if float(similarity) >= 40 else
                        "similarity-low"
                    )
                except:
                    sim_class = "similarity-low"

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"""
                        <div class="tag-card {bs_class}">
                            <div style="display: inline-flex; align-items: center;">
                                <h4 style="margin: 0;">🧪 Support Level</h4>
                                <div class="tooltip" style="margin-left: -14px; cursor: pointer; 
                                    font-size: 12px; line-height: 1; position: relative; top: -4px;">
                                    ℹ️
                                    <span class="tooltiptext">
                                        The support level indicates how well the LLM's claims are supported 
                                        by the provided knowledge base. This is determined by the number of
                                        claims that initially had a BS Label of "Supported," which indicates
                                        that the LLM's claim was supported by the documents in the knowledge 
                                        base.
                                    </span>
                                </div>
                            </div>
                            <p><b>{label}</b></p>
                        </div>
                    """, unsafe_allow_html=True)


                with col2:
                    st.markdown(f"""
                        <div class="tag-card {conf_class}">
                            <div style="display: inline-flex; align-items: center;">
                                <h4 style="margin: 0;">🔐 Confidence Level</h4>
                                <div class="tooltip" style="margin-left: -14px; cursor: pointer; 
                                font-size: 12px; line-height: 1; position: relative; top: -4px;">
                                    ℹ️
                                    <span class="tooltiptext">
                                        The confidence level uses the BS label and the similarity score to
                                        gauge how confident the LLM is in its answer. It is determined by
                                        the percentage of initial claims that were marked as high confidence.
                                    </span>
                                </div>
                            </div>
                            <p><b>{confidence}</b></p>
                        </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                        <div class="tag-card {sim_class}">
                             <div style="display: inline-flex; align-items: center;">
                                <h4 style="margin: 0;">📈 Similarity</h4>
                                <div class="tooltip" style="margin-left: -14px; cursor: pointer; 
                                font-size: 12px; line-height: 1; position: relative; top: -4px;">
                                    ℹ️
                                    <span class="tooltiptext">
                                        The similarity score rates the semantic similarity of the LLM's 
                                        claim to the language in the supporting documents pulled from 
                                        the knowledge base. A 0 indicates no similarity, and a 1 indicates 
                                        perfect similarity.
                                    </span>
                                </div>
                            </div>
                            <p><b>{similarity}</b></p>
                        </div>
                    """, unsafe_allow_html=True)
                        
                st.markdown(f"""
                    <div class="final-claim-card">
                        <div style="display: inline-flex; align-items: center;">
                            <h4 style="margin: 0;">📝 Initial Claim</h4>
                            <div class="tooltip" style="margin-left: -14px; cursor: pointer; 
                            font-size: 12px; line-height: 1; position: relative; top: -2px;">
                                ℹ️
                                <span class="tooltiptext">
                                    This is the LLM's initial response to the user's prompt 
                                    before the Prompt-Locked Loop runs.
                                </span>
                            </div>
                        </div>
                        <p style="font-size: 14px;">{answer}</p>
                    </div>
                """, unsafe_allow_html=True)

            # === Display Results ===
            from pll_controller import prompt_locked_loop
            # ... (paste your answer summary, PLL loop, final synthesized answer, and medium confidence claims display here exactly as before)
            import time
            status_placeholder.markdown("<div class='status-card'><span class='status-icon'>📄</span>🔁 Executing Prompt Locked Loop...</div>", unsafe_allow_html=True)
            # status_placeholder.info("🔁 Executing Prompt Locked Loop...")
            start_time = time.perf_counter()
            pll_logs, locked_claims, medium_confidence_claims = prompt_locked_loop(embedder, user_question, claim_eval, max_retry=3)
            pipeline_runtime = time.perf_counter() - start_time
            st.session_state["pipeline_runtime"] = pipeline_runtime
            st.session_state["pll_logs"] = pll_logs
            st.session_state["locked_claims"] = locked_claims
            st.session_state["medium_confidence_claims"] = medium_confidence_claims

            last_round_claims = pll_logs[-1]["claims"]
            

            # Synthesize final answer
            from pll_controller import synthesize_final_answer
            locked_claims = st.session_state.get("locked_claims", [])
            human_answer = synthesize_final_answer(user_question, [c["claim"] for c in locked_claims])
            if human_answer:
                st.markdown(f"""
                            <div class="final-claim-card">
                                <div style="display: inline-flex; align-items: center;">
                                    <h3 style="margin: 0;">✅ Final Synthesized Answer</h3>
                                    <div class="tooltip" style="margin-left: -14px; cursor: pointer; 
                                    font-size: 12px; line-height: 1; position: relative; top: 2px;">
                                        ℹ️
                                        <span class="tooltiptext">
                                            All LLM claims in the answer below were marked as 
                                            Supported and High Confidence.
                                        </span>
                                    </div>
                                </div>
                                <p style="font-size: 16px;">{human_answer}</p>
                            </div>
                        """, unsafe_allow_html=True)
            
            # Show medium confidence claims if there are any
            medium_confidence_claims = st.session_state.get("medium_confidence_claims", [])
            if medium_confidence_claims:
                bullet_points_html = "".join(
                    f"<li>{claim['claim']}</li>" for claim in medium_confidence_claims
                )
                st.markdown(f"""
                        <div class="final-claim-card">
                            <div style="display: inline-flex; align-items: center;">
                                <h4 style="margin: 0;">🤔 The following claims were also found to likely be true, 
                                    though with slightly lower confidence: </h4>
                            </div>
                            <p style="font-size: 16px;">{bullet_points_html}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
            status_placeholder.success("🎉 Full pipeline completed successfully!")
            st.session_state["pipeline_complete"] = True
            # st.balloons()


# === Directories ===
base_dir = Path("FORAGER_corpus/heterogenous_integration")
html_upload_dir = base_dir / "htmls"
pdf_upload_dir = base_dir / "pdfs"
html_upload_dir.mkdir(parents=True, exist_ok=True)
pdf_upload_dir.mkdir(parents=True, exist_ok=True)

# === Session State ===
if "selected_file" not in st.session_state:
    st.session_state.selected_file = None
if "selected_files" not in st.session_state:
    st.session_state.selected_files = set()

with tab_knowledge_base:
    # === Header ===
    st.header("📚 Knowledge Base Management")

    # === Upload Files ===
    st.markdown("### ➕ Upload More Files")
    uploaded_files = st.file_uploader("Upload documents (PDF or HTML)", type=["pdf", "html"], accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            file_ext = file.name.split(".")[-1].lower()
            save_dir = html_upload_dir if file_ext == "html" else pdf_upload_dir
            with open(save_dir / file.name, "wb") as f:
                f.write(file.read())
            st.success(f"✅ Uploaded {file.name}")
        st.session_state.selected_file = None
        st.session_state.selected_files.clear()
        st.rerun()

    # === File List (HTML + PDF) ===
    html_files = list(html_upload_dir.glob("*.html"))
    pdf_files = list(pdf_upload_dir.glob("*.pdf"))
    all_files = [{"name": f.name, "path": f, "type": "HTML"} for f in html_files] + \
                [{"name": f.name, "path": f, "type": "PDF"} for f in pdf_files]

    # === Search Bar ===
    search_query = st.text_input("🔍 Search files", placeholder="Type to search by filename...")
    if search_query:
        all_files = [f for f in all_files if search_query.lower() in f["name"].lower()]

    # === File List ===
    if all_files:
        for file in all_files:
            cols = st.columns([0.06, 0.55, 0.2, 0.2])
            with cols[0]:
                st.markdown(
                    """
                    <style>
                    div[data-testid="stCheckbox"] {
                        display: flex;
                        align-items: flex-start;
                        margin-top: -15px;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                checked = st.checkbox("", value=file["name"] in st.session_state.selected_files, key=f"chk_{file['name']}")
                if checked:
                    st.session_state.selected_files.add(file["name"])
                else:
                    st.session_state.selected_files.discard(file["name"])
            with cols[1]:
                st.markdown(f"📄 {file['name']}")
            with cols[2]:
                if st.button("👁 Preview", key=f"preview_{file['name']}"):
                    st.session_state.selected_file = file
            with cols[3]:
                if st.button("🗑 Delete", key=f"delete_{file['name']}"):
                    file["path"].unlink()
                    st.success(f"🗑 Deleted {file['name']}")
                    if st.session_state.selected_file and st.session_state.selected_file["name"] == file["name"]:
                        st.session_state.selected_file = None
                    st.rerun()
    else:
        st.info("No files found. Upload or adjust your search.")
        # ✅ Reset state if no files remain
        st.session_state.selected_file = None
        st.session_state.selected_files.clear()

    # === Bulk Delete Button ===
    if st.session_state.selected_files:
        if st.button("🗑 Delete Selected Files"):
            for file_name in list(st.session_state.selected_files):
                file_to_delete = next((f for f in all_files if f["name"] == file_name), None)
                if file_to_delete:
                    file_to_delete["path"].unlink()
                    if st.session_state.selected_file and st.session_state.selected_file["name"] == file_name:
                        st.session_state.selected_file = None
            st.session_state.selected_files.clear()
            st.success("🗑 Selected files deleted successfully!")
            st.rerun()

    # === File Preview Section ===
    if st.session_state.selected_file:
        file = st.session_state.selected_file
        st.markdown(f"### 👁 Preview: {file['name']}")

        if file["type"] == "HTML":
            html_content = file["path"].read_text(encoding="utf-8")
            soup = BeautifulSoup(html_content, "html.parser")
            html_preview = f"""
            <div style="background-color:white; color:black; padding:20px; border-radius:8px; height:600px; overflow:auto;">
                {soup}
            </div>
            """
            st.components.v1.html(html_preview, height=600, scrolling=True)

        elif file["type"] == "PDF":
            with open(file["path"], "rb") as f:
                pdf_bytes = f.read()
                pdf_viewer(input=pdf_bytes, width=800, height=900)
        
# Tab 3: Step-by-step claims breakdown
with tab_claims:
    st.header("📑 Claims Breakdown")

    initial_claim_eval = st.session_state.get("claim_eval", {})
    pll_logs = st.session_state.get("pll_logs", [])

    if not initial_claim_eval:
        st.info("No claims available. Submit a question in the Chat tab to generate claims.")
    else:
        for original_claim, eval_info in initial_claim_eval.items():
            final_status = "Unknown❓"
            final_rephrased = None
            was_locked_pre_pll = False
            was_discarded = False
            last_seen_index = None

            # Step 1: Find the last index where the original claim appears
            for idx, round_log in enumerate(pll_logs):
                for claim_info in round_log["claims"]:
                    if claim_info["claim"] == original_claim:
                        last_seen_index = idx
                        if round_log.get("pll_round") == "Pre-PLL Lock":
                            was_locked_pre_pll = True
                        break

            # Step 2: Determine final status
            if was_locked_pre_pll:
                final_status = "🔒 Locked before PLL"
            elif last_seen_index is not None:
                current_version = original_claim
                for j in range(last_seen_index + 1, len(pll_logs)):
                    next_log = pll_logs[j]
                    if next_log.get("pll_round") == "Pre-PLL Lock":
                        continue
                    for claim_info in next_log["claims"]:
                        if claim_info["claim"] != current_version:
                            current_version = claim_info["claim"]
                            final_rephrased = current_version
                            decision = claim_info.get("pll_decision", "N/A")
                            round_label = next_log.get("pll_round", f"index {j}")
                            final_status = f"🔁 Rephrased → `{decision}` in round {round_label}"
                            if decision.lower().startswith("discarded"):
                                was_discarded = True
                            break
                        elif claim_info["claim"] == current_version:
                            decision = claim_info.get("pll_decision", "")
                            if decision.lower().startswith("discarded"):
                                was_discarded = True
                                round_label = next_log.get("pll_round", f"index {j}")
                                final_status = f"🗑️ Discarded in round {round_label}"
                            break

            # --- Render UI ---
            st.markdown(f"### 📝 Claim: {original_claim}")

            # Grab labels
            label = eval_info.get("label", "N/A")
            confidence = eval_info.get("confidence", "N/A")

            # Map label to color class
            eval_class = "status-red" if label.lower() == "contradicted" else \
                        "status-orange" if label.lower() == "unsupported" else \
                        "status-green" if label.lower() == "supported" else "status-red"

            conf_class = "status-red" if confidence.lower() == "zero" else \
                        "status-orange" if confidence.lower() == "medium" else \
                        "status-green" if confidence.lower() == "high" else "status-red"

            pll_class = "status-blue"  # Always blue

            # Render cards
            st.markdown(
                f"""
                <div class="stats-row">
                    <div class="stat-card">
                        <h4>Evaluation</h4>
                        <span class="{eval_class}">{label}</span>
                    </div>
                    <div class="stat-card">
                        <h4>Confidence</h4>
                        <span class="{conf_class}">{confidence}</span>
                    </div>
                    <div class="stat-card">
                        <h4>Final PLL Status</h4>
                        <span class="{pll_class}">{final_status}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if was_discarded:
                st.markdown("❌ **Final Rephrased Claim was discarded by the LLM.**")
            elif final_rephrased:
                st.markdown(f"""<div><b>• Final Rephrased Claim:</b> {final_rephrased}</div>""", unsafe_allow_html=True)

            # PLL Trace
            with st.expander("🔎 Claim Evolution"):
                lineage, seen, queue = [], set(), [original_claim]
                while queue:
                    current = queue.pop(0)
                    if current in seen:
                        continue
                    seen.add(current)
                    lineage.append(current)
                    for round_log in pll_logs:
                        for claim_info in round_log["claims"]:
                            if claim_info.get("original_claim") == current:
                                queue.append(claim_info["claim"])

                if was_locked_pre_pll:
                    st.markdown("""
                    > 🔒 **Locked Before PLL**
                    > This claim was finalized during the initial evaluation and did not participate in any rephrasing or confidence-based updates.
                    """)
                else:
                    for round_log in pll_logs:
                        round_label = round_log.get("pll_round", "N/A")
                        for claim_info in round_log["claims"]:
                            if claim_info["claim"] in lineage:
                                if str(round_label) in ["0", "Pre-PLL Lock"]:
                                    continue

                                bs_label = claim_info.get("eval_label", "N/A")
                                confidence = claim_info.get("confidence_label", "N/A")
                                decision = claim_info.get("pll_decision", "N/A")

                                # Define colors
                                label_colors = {
                                    "Contradicted": "red",
                                    "Unsupported": "orange",
                                    "Supported": "green",
                                    "Zero": "red",
                                    "Medium": "orange",
                                    "High": "green",
                                    "Discarded by LLM": "red"
                                }
                            
                                bs_color = label_colors.get(bs_label, "white")
                                conf_color = label_colors.get(confidence, "white")
                                dec_color = label_colors.get(decision, "white")
                                
                                st.markdown(f"**📈 PLL Round {round_label}**")
                                st.markdown("---")
                                st.markdown(f"- **Rephrased:** {claim_info['claim']}")
                                st.markdown(f"- **BS Label:** <span style='background-color:#171717; color:{bs_color}; padding:2px 6px; border-radius:4px; font-weight:600;'>{bs_label}</span>", unsafe_allow_html=True)
                                st.markdown(f"- **Confidence:** <span style='background-color:#171717; color:{conf_color}; padding:2px 6px; border-radius:4px; font-weight:600;'>{confidence}</span>", unsafe_allow_html=True)
                                st.markdown(f"- **Outcome:** <span style='background-color:#171717; color:{dec_color}; padding:2px 6px; border-radius:4px; font-weight:600;'>{decision}</span>", unsafe_allow_html=True)
                                        
            # Supporting Chunks
            with st.expander("📥 Supporting Chunks"):
                for idx, chunk in enumerate(eval_info.get("supporting_chunks", []), 1):
                    title = chunk.get("source_filename", "")
                    text = chunk.get("text", "").replace("\n", "\n> ")
                    st.markdown(f"**Source {idx}:** {title}")
                    st.markdown(f">{text}")


# Tab 4: Metrics & Visualizations

with tab_metrics:
    # Get data
    claim_eval = st.session_state.get("claim_eval", {})
    pll_logs = st.session_state.get("pll_logs", [])

    total_claims = len(claim_eval)
    total_pll_rounds = len(pll_logs) - 2
    total_claims_in_rounds = sum(len(log["claims"]) for log in pll_logs)

    # Ensure displayed rounds never go negative
    display_pll_rounds = max(total_pll_rounds, 0)

    # Pipeline runtime: show 0 if not set
    pipeline_runtime = st.session_state.get("pipeline_runtime", None)
    display_runtime = round(pipeline_runtime, 2) if pipeline_runtime is not None else 0

    # Visualization data prep
    label_counts = {}
    confidence_counts = {"High": 0, "Medium": 0, "Low": 0, "Zero": 0}
    round_label_counts = []

    for round_log in pll_logs:
        label_count = {"Round": round_log["pll_round"], "Supported": 0, "Unsupported": 0, "Contradicted": 0}
        for claim in round_log["claims"]:
            # Use eval_label & confidence_label if present
            label = claim.get("eval_label") or claim.get("label", "N/A")
            confidence = claim.get("confidence_label") or claim.get("confidence", "N/A")

            label_counts[label] = label_counts.get(label, 0) + 1
            confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1

            if label in ["Supported", "Unsupported", "Contradicted"]:
                label_count[label] += 1
        round_label_counts.append(label_count)

    df_labels = pd.DataFrame.from_dict(
        {k: v for k, v in label_counts.items() if k in ["Supported", "Unsupported", "Contradicted"]},
        orient='index',
        columns=["Count"]
    )
    df_conf = pd.DataFrame.from_dict(confidence_counts, orient='index', columns=["Count"])
    df_rounds = pd.DataFrame(round_label_counts)

    # Safeguard against missing columns
    expected_cols = {"Round", "Supported", "Unsupported", "Contradicted"}
    if expected_cols.issubset(df_rounds.columns):
        df_melted = df_rounds.melt(
            id_vars="Round", 
            value_vars=["Supported", "Unsupported", "Contradicted"],
            var_name="Label", value_name="Count"
        )
    else:
        df_melted = pd.DataFrame(columns=["Round", "Label", "Count"])  # Empty fallback
    
    if df_labels.empty:
        st.info("No claim label data available for visualization.")
    if df_melted.empty:
        st.info("No PLL round data available for visualization.")
    
    # === Debugging: Show Raw DataFrames ===
    st.markdown("### 📶 Raw Data for Visualizations")

    # Metrics cards
    st.markdown(f"""
        <div>
            <div class='metric-card-container'>
                <div class='metric-card'>
                    <div><span class="icon">📄</span><h4 style='display:inline;'>Total Claims Generated</h4></div>
                    <div class='value'>{total_claims}</div>
                </div>
                <div class='metric-card'>
                    <div><span class="icon">🔁</span><h4 style='display:inline;'>Total PLL Rounds</h4></div>
                    <div class='value'>{display_pll_rounds}</div>
                </div>
                <div class='metric-card'>
                    <div><span class="icon">📋</span><h4 style='display:inline;'>Total Claims in Rounds</h4></div>
                    <div class='value'>{total_claims_in_rounds}</div>
                </div>
                <div class='metric-card'>
                    <div><span class="icon">⏱️</span><h4 style='display:inline;'>Total Pipeline Runtime</h4></div>
                    <div class='value'>{display_runtime}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Scoped CSS: Rounded corners and padding for the visualization containers
    st.markdown("""
        <style>
        div[data-testid="stPlotlyChart"] {
            background-color: #1e1e1e;
            border-radius: 10px;
            padding: 15px;
            margin-top: 15px;
            margin-bottom: 15px;
            overflow: hidden; /* Ensures chart respects rounded edges */
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # === Pie Chart (Donut) ===
    with col1:

        st.markdown("### 📑 Claim Label Distribution")
        st.caption("This chart shows the overall distribution of claim labels (Supported, Unsupported, Contradicted) across all PLL rounds combined.")

        if not df_labels.empty:
            fig_pie = px.pie(
                df_labels.reset_index(),
                names="index",
                values="Count",
                color="index",
                color_discrete_map={
                    "Supported": "#2ecc71",
                    "Unsupported": "#f39c12",
                    "Contradicted": "#e74c3c"
                }
            )
            fig_pie.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",   # Transparent background
                paper_bgcolor="rgba(0,0,0,0)",  # Transparent to match container
                font=dict(color="white"),
                margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(
                    orientation="h",
                    y=-0.15,
                    x=0.5,
                    xanchor="center"
                )
            )
            fig_pie.update_traces(
                textfont=dict(color="white"),
                marker=dict(line=dict(color="#1e1e1e", width=2)),
                textposition="inside",
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    # === Bar Chart ===
    with col2:

        st.markdown("### 📊 Claim Label Trends by PLL Round")
        st.caption("This chart shows how claim labels (Supported, Unsupported, Contradicted) evolve across each PLL round, highlighting changes over time.")

        if not df_melted.empty:
            fig_bar = px.bar(
                df_melted,
                x="Round",
                y="Count",
                color="Label",
                barmode="group",
                text="Count",
                color_discrete_map={
                    "Supported": "#2ecc71",
                    "Unsupported": "#f39c12",
                    "Contradicted": "#e74c3c"
                }
            )
            fig_bar.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                margin=dict(t=20, b=20, l=20, r=20),
                xaxis=dict(title="PLL Round"),
                yaxis=dict(title="Claim Count")
            )
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
    
    # === Confidence Breakdown Table ===
    st.markdown("### 🗂️ Confidence Breakdown by PLL Round")
    st.caption("This table summarizes the count of claims at each confidence level (High, Medium, Low, Zero) across all PLL rounds, providing insight into confidence progression during the pipeline.")

    confidence_data = []
    for round_log in pll_logs:
        conf_counts = {"High": 0, "Medium": 0, "Low": 0, "Zero": 0}
        for claim in round_log["claims"]:
            conf = claim.get("confidence_label", "N/A")
            if conf in conf_counts:
                conf_counts[conf] += 1
        confidence_data.append({
            "Round": round_log["pll_round"],
            **conf_counts
        })

    df_confidence = pd.DataFrame(confidence_data)
    st.dataframe(df_confidence, use_container_width=True)


    # # === Prepare Heatmap Data ===
    # heatmap_data = []
    # confidence_levels = ["High", "Medium", "Low", "Zero"]

    # for round_log in pll_logs:
    #     round_num = round_log["pll_round"]
    #     conf_count = {level: 0 for level in confidence_levels}
    #     for claim in round_log["claims"]:
    #         conf_label = claim.get("confidence_label", "N/A")
    #         if conf_label in confidence_levels:
    #             conf_count[conf_label] += 1
    #     for level, count in conf_count.items():
    #         heatmap_data.append({"Round": round_num, "Confidence": level, "Count": count})

    # df_heatmap = pd.DataFrame(heatmap_data)

    # # === Render Heatmap ===
    # if not df_heatmap.empty:
    #     fig_heatmap = px.density_heatmap(
    #         df_heatmap,
    #         x="Round",
    #         y="Confidence",
    #         z="Count",
    #         color_continuous_scale="Blues",  # or "Viridis" for a multi-color scale
    #         text_auto=True
    #     )
    #     fig_heatmap.update_layout(
    #         plot_bgcolor="rgba(0,0,0,0)",
    #         paper_bgcolor="rgba(0,0,0,0)",
    #         font=dict(color="white"),
    #         margin=dict(t=30, b=30, l=30, r=30),
    #         xaxis=dict(title="PLL Round"),
    #         yaxis=dict(title="Confidence Level")
    #     )
    #     st.plotly_chart(fig_heatmap, use_container_width=True, config={"displayModeBar": False})
    # else:
    #     st.info("No confidence data available for heatmap visualization.")

    # === PLL Rounds Breakdown ===
    st.markdown("### 📖 PLL Rounds Breakdown")

    if not pll_logs:
        st.info("No PLL logs available.")
    else:
        # Create one column per round dynamically
        cols = st.columns(len(pll_logs))

        for col, round_log in zip(cols, pll_logs):
            with col:
                st.markdown(f"""
                    <div style='background-color: #1e1e1e; 
                                padding: 15px; 
                                border-radius: 10px; 
                                text-align: left;
                                margin-bottom: 15px;>
                        <span style='color: white; font-weight: bold; font-size:16px;'>
                            PLL Round {round_log['pll_round']}
                        </span>
                        <br>
                        <span style='color: #bbb; font-size:14px;'>
                            {len(round_log['claims'])} claims processed
                        </span>
                    </div>
                """, unsafe_allow_html=True)
    

# Tab 5: PLL Logs
with tab_logs:
    st.header("✍️ PLL Logs")

    pll_logs = st.session_state.get("pll_logs", [])

    if not pll_logs:
        st.info("No PLL logs available. Submit a question in the Chat tab to generate logs.")
    else:
        for round_log in pll_logs:
            if not round_log["claims"]:
                continue  # Skip rounds with no claims

            round_label = round_log["pll_round"]
            expander_title = "Initial Claim Evaluation (Pre-PLL)" if round_label == 0 else f"PLL Round {round_label}"
            with st.expander(f"📈 {expander_title}", expanded=False):
                for claim_info in round_log["claims"]:
                    claim_text = claim_info.get("claim", "❓")
                    decision = claim_info.get("pll_decision", "N/A")
                    confidence = claim_info.get("confidence_label", "N/A")
                    label = claim_info.get("eval_label", "N/A")
                    reason = claim_info.get("reason", "No reason provided.")

                    rephrased_from = claim_info.get("original_claim")
                    was_rephrased = bool(rephrased_from and rephrased_from != claim_text)

                    # 🔹 Color mapping
                    label_colors = {
                        "Contradicted": "red",
                        "Unsupported": "orange",
                        "Supported": "green",
                        "Zero": "red",
                        "Medium": "orange",
                        "High": "green",
                        "Discarded by LLM": "red",
                        "Unknown": "#4da6ff"  # softer blue
                    }

                    dec_color = label_colors.get(decision, "white")
                    conf_color = label_colors.get(confidence, "white")
                    label_color = label_colors.get(label, "white")

                    # --- Rendering ---
                    st.markdown(f"---")
                    st.markdown(f"**📝 Claim:** {claim_text}")
                    st.markdown(f"- **Decision:** <span style='background-color:#171717; color:{dec_color}; padding:2px 6px; border-radius:4px; font-weight:600;'>{decision}</span>", unsafe_allow_html=True)
                    st.markdown(f"- **Confidence:** <span style='background-color:#171717; color:{conf_color}; padding:2px 6px; border-radius:4px; font-weight:600;'>{confidence}</span>", unsafe_allow_html=True)
                    st.markdown(f"- **Label:** <span style='background-color:#171717; color:{label_color}; padding:2px 6px; border-radius:4px; font-weight:600;'>{label}</span>", unsafe_allow_html=True)
                    
                    if was_rephrased:
                        st.markdown(f"- **Rephrased from:** _{rephrased_from}_")
                    st.markdown(f"- **Reason:** {reason}")

                    # Optional metadata
                    metadata = claim_info.get("metadata", {})
                    if metadata:
                        st.markdown("###### 🧬 Metadata:")
                        if "similarity_scores" in metadata:
                            st.markdown(f"- Similarity Scores: {metadata['similarity_scores']}")
                        if "supporting_chunks" in metadata:
                            st.markdown(f"- Supporting Chunk IDs: {metadata['supporting_chunks']}")
                        if "rephrase_count" in metadata:
                            st.markdown(f"- Rephrase Round Count: {metadata['rephrase_count']}")
                        if "rerank_method" in metadata:
                            st.markdown(f"- Rerank Method Used: {metadata['rerank_method']}")
    locked_claims = st.session_state.get("locked_claims", [])
    if locked_claims:
        with st.expander("✅ Final Locked Claims", expanded=False):
            for c in locked_claims:
                st.markdown(f"- {c['claim']}")