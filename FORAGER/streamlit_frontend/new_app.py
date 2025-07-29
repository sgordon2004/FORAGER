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

# Sidebar for status updates
with st.sidebar:
    st.markdown("### 🚦 Pipeline Status")
    status_placeholder = st.empty()

# Tab 1: Chat Tab (LLM Interaction)
with tab_chat:
    st.markdown("""
        <div class="chat-header" style="text-align: center;">
            <img src="https://cdn-icons-png.flaticon.com/512/4712/4712137.png" class="chat-logo" width="250" alt="Chat Logo">
            <h1 class="chat-title">What's on your mind today?</h1>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="steps-container">
        <div class="step-card">
            <div class="step-title">Upload</div>
            <div class="step-description">Upload your PDFs or HTML files to get started.</div>
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

    # === File Upload Section ===
    st.markdown('<div class="centered-uploader">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "html", "txt"],
        accept_multiple_files=True
    )

    # Paths to uploaded files
    base_dir = Path("FORAGER_corpus/heterogenous_integration")
    html_dir = base_dir / "html"
    pdf_dir = base_dir / "pdf"
    html_upload_dir = base_dir / "htmls"
    pdf_upload_dir = base_dir / "pdfs"
    txt_upload_dir = base_dir / "txts"

    if st.button("Process"):
        if not uploaded_files:
            status_placeholder.warning("⚠️ No documents uploaded.")
        else:
            status_placeholder.info("📄 Starting text extraction...")
            for file in uploaded_files:
                file_ext = file.name.split(".")[-1].lower()
                file_bytes = file.read()

                if file_ext == "html":
                    from ingestor import clean_html, html_text_dir, json_dir
                    html_upload_dir = base_dir / "htmls"
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
                else:
                    st.warning(f"Unsupported file type: {file.name}")
                    continue
            time.sleep(1)
            status_placeholder.success("✅ Text extraction complete!")

            status_placeholder.info("🔗 Chunking documents...")
            from chunker import main as chunker_main
            chunker_main()
            time.sleep(1)
            status_placeholder.success("✅ Chunking complete!")

            status_placeholder.info("💾 Initializing FAISS...")
            from FORAGER.embedder import FAISSEmbedder
            embedder = FAISSEmbedder.create_default()
            embedder.initialize_faiss()
            # Find best spot for where we should be checking for new chunks
            # embeddings = embedder.embed_chunks()
            # embedder.add_to_faiss(embeddings)
            st.session_state["embedder"] = embedder
            time.sleep(1)
            status_placeholder.success("✅ FAISS initialized!")

            st.session_state["documents_processed"] = True
            status_placeholder.success("✅ Documents processed!")
    # Run question process only if documents have been processed
    if any([list(html_upload_dir.glob("*.html")), list(pdf_upload_dir.glob("*.pdf"))]):
        # Question input section
        st.markdown("### 💬 Ask a Question")
        user_question = st.text_input("Query the knowledge base:")

        if st.button("Submit Question") and user_question:
            st.session_state["submitted"] = True
            status_placeholder.info("🤖 Generating answer via LLM...")
            embedder = st.session_state.get("embedder")

            if embedder is None:
                from FORAGER.embedder import FAISSEmbedder
                embedder = FAISSEmbedder.create_default()
                try:
                    embedder.initialize_faiss() # Load existing FAISS index if present
                    st.session_state["embedder"] = embedder
                except Exception as e:
                    st.warning("⚠️ Could not initialize FAISS index. Please upload and process documents first.")
                    st.stop()
            else:
                from test_pipeline import generate_and_evaluate_claims
                answer, claim_eval = generate_and_evaluate_claims(embedder, user_question)
                st.session_state["answer"] = answer
                st.session_state["claim_eval"] = claim_eval

                answer = st.session_state.get("answer", "❓ No answer available")
                claim_eval = st.session_state.get("claim_eval", {})

                if claim_eval:
                    final_claim, info = list(claim_eval.items())[0]
                    label = info.get("label", "N/A")
                    confidence = info.get("confidence", "N/A")
                    chunks = info.get("supporting_chunks", [])
                    scores = [doc.get("score", 0) for doc in chunks if "score" in doc]
                    similarity = round(sum(scores) / len(scores), 3) if scores else "N/A"

                    st.markdown("## 🧾 Answer Summary")

                    # Dynamic tag colors
                    bs_class = f"bs-{label}"
                    conf_class = f"confidence-{confidence}"
                    try:
                        sim_class = (
                            "similarity-high" if float(similarity) >= 0.7 else
                            "similarity-medium" if float(similarity) >= 0.4 else
                            "similarity-low"
                        )
                    except:
                        sim_class = "similarity-low"

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown(f"""
                            <div class="tag-card {bs_class}">
                                <div style="display: inline-flex; align-items: center;">
                                    <h4 style="margin: 0;">🧪 BS Label</h4>
                                    <div class="tooltip" style="margin-left: -14px; cursor: pointer; 
                                     font-size: 12px; line-height: 1; position: relative; top: -4px;">
                                        ℹ️
                                        <span class="tooltiptext">
                                            This label indicates how well the LLM's claims are supported 
                                            by the provided knowledge base.
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
                                    <h4 style="margin: 0;">🔐 Confidence</h4>
                                    <div class="tooltip" style="margin-left: -14px; cursor: pointer; 
                                    font-size: 12px; line-height: 1; position: relative; top: -4px;">
                                        ℹ️
                                        <span class="tooltiptext">
                                            This label uses the BS label and the similarity score to
                                            gague how confident the LLM is in its answer.
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
                                            claim to the language in the supporting documents. A 0 indicates 
                                            no similarity, and a 1 indicates perfect similarity.
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
                                font-size: 12px; line-height: 1; position: relative; top: -4px;">
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

                # Run Prompt Locked Loop as long as question was not marked as irrelevant
                # if (answer != "This question cannot be answered by the information in the knowledge base."):
                from pll_controller import prompt_locked_loop
                import time
                status_placeholder.info("🔁 Executing Prompt Locked Loop...")
                start_time = time.perf_counter()
                pll_logs, locked_claims, medium_confidence_claims = prompt_locked_loop(embedder, user_question, claim_eval, max_retry=3)
                pipeline_runtime = time.perf_counter() - start_time
                st.session_state["pipeline_runtime"] = pipeline_runtime
                st.session_state["pll_logs"] = pll_logs
                st.session_state["locked_claims"] = locked_claims
                st.session_state["medium_confidence_claims"] = medium_confidence_claims
                # else:
                #     print(f"Prompt Locked Loop skipped due to unanswerable question.")

                last_round_claims = pll_logs[-1]["claims"]

        # TODO: Move this to run after the final answer is locked.
        
        from pll_controller import synthesize_final_answer
        locked_claims = st.session_state.get("locked_claims", [])
        human_answer = synthesize_final_answer(user_question, [c["claim"] for c in locked_claims])
        if human_answer:
            st.markdown("## Final Synthesized Answer")
            st.markdown(f"""
                        <p style="font-size: 14px;">All LLM claims in the answer below were marked as 
                        Supported and High Confidence.</p>
                        """, unsafe_allow_html=True)
            st.markdown(f"> {human_answer}")
            st.markdown("#### The following claims were also found to likely be true, though with slightly lower confidence: ")
            medium_confidence_claims = st.session_state.get("medium_confidence_claims", [])
            if medium_confidence_claims:
                for claim in medium_confidence_claims:
                    st.markdown(f"- {claim['claim']}")
            status_placeholder.success("🎉 Full pipeline completed successfully!")
            st.balloons()

# Tab 2: Knowledge Base
with tab_knowledge_base:

    st.header("📚 Knowledge Base Management")
    
    html_upload_dir = base_dir / "htmls"
    pdf_upload_dir = base_dir / "pdfs"

    html_upload_dir.mkdir(parents=True, exist_ok=True)
    pdf_upload_dir.mkdir(parents=True, exist_ok=True)

    st.markdown("### ➕ Upload More Files")
    more_files = st.file_uploader("Upload additional documents (PDF or HTML)", type=["pdf", "html"], accept_multiple_files=True, key="additional_uploads")

    if more_files:
        # Read and store all new file data to prevent .read() issues later
        uploaded_file_data = [(file, file.read()) for file in more_files]
        for file, file_bytes in uploaded_file_data:
            file_ext = file.name.split(".")[-1].lower()
            file_bytes = file.read()
            save_dir = html_upload_dir if file_ext == "html" else pdf_upload_dir
            with open(save_dir / file.name, "wb") as f:
                f.write(file_bytes)
            st.success(f"✅ Uploaded {file.name}")
        
        # Process new documents just like in Tab 1
        if st.button("Process New Document(s)"):
            status_placeholder.info("📄 Starting text extraction...")
            # Extract text from uploaded files
            for file, file_bytes in uploaded_file_data:
                # Isolate file extension and read bytes
                file_ext = file.name.split(".")[-1].lower()
                # file_bytes = file.read()

                if file_ext == "html":
                    from ingestor import clean_html, html_text_dir, json_dir
                    # Temporarily save uploaded file
                    
                    input_path = html_upload_dir / file.name
                    html_upload_dir.mkdir(parents=True, exist_ok=True)
                    with open(input_path, "wb") as f:
                        f.write(file_bytes)
                    clean_html(input_path, html_text_dir, json_dir)

                elif file_ext == "pdf":
                    # Extract text from PDF
                    from extractor import extract_pdf
                    from ingestor import dump_pdf_text
                    
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
            status_placeholder.info("💾 Adding to FAISS...")
            # Creating temporary second embedder object because I can't access the first one here
            from embedder import FAISSEmbedder
            embedder2 = FAISSEmbedder.create_default()
            new_embeddings = embedder2.embed_chunks_from_json()
            embedder2.add_to_faiss(new_embeddings)
            st.session_state["embedder2"] = embedder2
            time.sleep(1)
            status_placeholder.success("✅ New documents processed!")
        
        st.rerun()

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
                    st.rerun()

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
                    st.rerun()

    st.markdown("---")

    st.markdown("### 🧹 Clear Entire Knowledge Base")

    if st.button("Delete All Uploaded Files"):
        for file_path in html_upload_dir.glob("*.html"):
            file_path.unlink()
        for file_path in pdf_upload_dir.glob("*.pdf"):
            file_path.unlink()
        st.success("🗑️ All uploaded documents have been deleted.")
        st.rerun()
        

        

# Tab 3: Step-by-step claims breakdown
with tab_claims:
    st.header("📑 Claims Breakdown")

    initial_claim_eval = st.session_state.get("claim_eval", {})
    pll_logs = st.session_state.get("pll_logs", [])

    if not initial_claim_eval:
        st.info("No claims available. Submit a question in the Chat tab to generate claims.")
    else:
        for original_claim, eval_info in initial_claim_eval.items():
            final_status = "❓ Unknown"
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

            # ✅ Render Stat Cards instead of bullet list
            st.markdown(
                f"""
                <div class="stats-row">
                    <div class="stat-card">
                        <h4>Evaluation</h4>
                        <span class="label-tag eval-{label.lower()}">{label}</span>
                    </div>
                    <div class="stat-card">
                        <h4>Confidence</h4>
                        <span class="label-tag conf-{confidence.lower()}">{confidence}</span>
                    </div>
                    <div class="stat-card">
                        <h4>Final PLL Status</h4>
                        <span class="pll-status">{final_status}</span>
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
                                
                                st.markdown(f"**▶️ PLL Round {round_label}**")
                                st.markdown(f"- **Rephrased:** \{claim_info['claim']}")
                                st.markdown(f"- **BS Label:** `{bs_label}`")
                                st.markdown(f"- **Confidence:** `{confidence}`")
                                st.markdown(f"- **Outcome:** `{decision}`")
                                st.markdown("---")

            # Supporting Chunks
            with st.expander("📥 Supporting Chunks"):
                for idx, chunk in enumerate(eval_info.get("supporting_chunks", []), 1):
                    title = chunk.get("source_filename", "")
                    text = chunk.get("text", "").replace("\n", "\n> ")
                    st.markdown(f"> **Chunk {idx}:**\n> {text}")

            st.markdown(f"### 📝 Claim: {original_claim}")
            st.markdown(f"""
                <p><strong>Evaluation:</strong> 
                <span class="label-tag eval-{label}">
                    {label}
                </span></p>
            """, unsafe_allow_html=True)

            st.markdown(f"""
                <p><strong>Confidence:</strong> 
                <span class="label-tag conf-{confidence}">
                    {confidence}
                </span></p>
            """, unsafe_allow_html=True)

            with st.expander("📜 Supporting Chunks"):
                for idx, chunk in enumerate(eval_info.get("supporting_chunks", []), 1):
                    formatted_chunk = chunk["text"].replace("\n", "\n> ")
                    st.markdown(f"> **Chunk {idx}:**  \n> {formatted_chunk}")

# Tab 4: Metrics & Performance
with tab_metrics:
    import plotly.express as px
    import pandas as pd
    # Get data
    claim_eval = st.session_state.get("claim_eval", {})
    pll_logs = st.session_state.get("pll_logs", [])

    total_claims = len(claim_eval)
    total_pll_rounds = len(pll_logs) - 2
    total_claims_in_rounds = sum(len(log["claims"]) for log in pll_logs)

    # Visualization data prep
    label_counts = {}
    confidence_counts = {"High": 0, "Medium": 0, "Low": 0, "Zero": 0}
    round_label_counts = []

    for round_log in pll_logs:
        label_count = {"Round": round_log["pll_round"], "Supported": 0, "Unsupported": 0, "Contradicted": 0}
        for claim in round_log["claims"]:
            label = claim.get("label", "N/A")
            confidence = claim.get("confidence", "N/A")
            label_counts[label] = label_counts.get(label, 0) + 1
            confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
            if label in label_count:
                label_count[label] += 1
        round_label_counts.append(label_count)

    df_labels = pd.DataFrame.from_dict(label_counts, orient='index', columns=["Count"])
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

    # Build charts BEFORE columns
    fig1 = px.bar(
        df_labels.reset_index(), x='index', y='Count', color='index',
        labels={'index': 'Label'},
        color_discrete_sequence=['#62B6CB', '#FFA07A', '#FF6347']
    )
    fig1.update_layout(
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='white')
    )

    fig2 = px.bar(
        df_conf.reset_index(), x='index', y='Count', color='index',
        labels={'index': 'Confidence'},
        color_discrete_sequence=['#90EE90', '#FFD700', '#FFA07A', '#FF4500']
    )
    fig2.update_layout(
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='white')
    )

    fig3 = px.bar(
        df_melted, x='Round', y='Count', color='Label',
        barmode='group',
        color_discrete_sequence=['#00BFFF', '#FFA500', '#DC143C']
    )
    fig3.update_layout(
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='white')
    )

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
                    <div class='value'>{total_pll_rounds}</div>
                </div>
                <div class='metric-card'>
                    <div><span class="icon">📦</span><h4 style='display:inline;'>Total Claims in Rounds</h4></div>
                    <div class='value'>{total_claims_in_rounds}</div>
                </div>
                <div class='metric-card'>
                    <div><span class="icon">⏱️</span><h4 style='display:inline;'>Total Pipeline Runtime</h4></div>
                    <div class='value'>{round(st.session_state.get("pipeline_runtime", "N/A"), 2) if st.session_state.get("pipeline_runtime", "N/A") != "N/A" else "N/A"}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 3-Column Visualization Section
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class='visualization-box'>
                <div class='visualization-title'>🔖 Claim Label Distribution</div>
        """, unsafe_allow_html=True)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class='visualization-box'>
                <div class='visualization-title'>📊 Confidence Level Distribution</div>
        """, unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class='visualization-box'>
                <div class='visualization-title'>🔁 PLL Round Label Breakdown</div>
        """, unsafe_allow_html=True)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # PLL round breakdowns
    st.markdown("### 🪵 PLL Rounds Breakdown")
    if not pll_logs:
        st.info("No PLL logs available.")
    else:
        for round_log in pll_logs:
            st.markdown(f"""
                <div style='background-color: #1e1e1e; padding: 15px; border-radius: 10px; margin-bottom: 10px;'>
                    <span style='color: white; font-weight: bold;'>PLL Round {round_log['pll_round']}</span>
                    <br><span style='color: #bbb;'>{len(round_log['claims'])} claims processed</span>
                </div>
            """, unsafe_allow_html=True)

# Tab 5: PLL Logs
with tab_logs:
    st.header("🪵 PLL Logs")

    pll_logs = st.session_state.get("pll_logs", [])

    if not pll_logs:
        st.info("No PLL logs available. Submit a question in the Chat tab to generate logs.")
    else:
        for round_log in pll_logs:
            if not round_log["claims"]:
                continue  # Skip rounds with no claims

            round_label = round_log["pll_round"]
            expander_title = "Initial Claim Evaluation (Pre-PLL)" if round_label == 0 else f"PLL Round {round_label}"
            with st.expander(f"▶️ {expander_title}", expanded=False):
                for claim_info in round_log["claims"]:
                    claim_text = claim_info.get("claim", "❓")
                    decision = claim_info.get("pll_decision", "N/A")
                    confidence = claim_info.get("confidence_label", "N/A")
                    label = claim_info.get("eval_label", "N/A")
                    reason = claim_info.get("reason", "No reason provided.")

                    rephrased_from = claim_info.get("original_claim")
                    was_rephrased = bool(rephrased_from and rephrased_from != claim_text)

                    st.markdown(f"---")
                    st.markdown(f"**📝 Claim:** {claim_text}")
                    st.markdown(f"- **Decision:** `{decision}`")
                    st.markdown(f"- **Confidence:** `{confidence}`")
                    st.markdown(f"- **Label:** `{label}`")
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