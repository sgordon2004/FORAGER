# 🧠 FORAGER
<p align="center">
  <img src="https://sgordon-demo-s3.s3.us-east-2.amazonaws.com/IMG_6709.jpeg" alt="FORAGER Logo" width="200"/>
</p>

_A trust-first RAG pipeline for validating LLM outputs._

## ❓ What is FORAGER?
**FORAGER** (Fact-Oriented Responsible AI-Guided Engineering Research) is a verification pipeline designed to improve trust in Large Language Models (LLMs). It combines **Prompt-Locked Looping**, **Retrieval-Augmented Generation (RAG)**, and **Natural Language Inference (NLI)** techniques to validate AI-generated claims against reliable evidence.

FORAGER is built for **domains where factual accuracy is non-negotiable**, like engineering research, scientific documentation, and techincal reports. LLM output is often treated as a *black box*, where what you see is what you get, and the process that derived the output is mysterious to the user. FORAGER serves to make the reasoning process completely transparent by indentifying, scoring, and verifying each claim the LLM makes.

Its modular design gives developers and users the freedom to:
- Plug in custom document sets (PDF & HTML)
    - *More formats to come!*
- *(Eventually)* swap in different LLM or NLI models
- Trace claim evolution across multiple verification rounds

FORAGER is well-suited to **high-stakes, document-grounded question answering**, where citations, traceability, and accuracy matter as much as fluency.

## 💡 Why FORAGER Matters
The use of LLMs is quickly becoming, if it has not already become, the norm in today's world. Whether you are a freshman in high school, a doctoral student, or the CEO of a Fortune 500 company, you have more than likely implemented LLMs in your day-to-day in some form. While this often makes our lives more convenient, LLMs are far from perfect. They consistently fail to output what you wanted, [hallucinate false information](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence)), and flat-out lie. Even though almost everyone uses LLMs, almost nobody actually understands how the LLM comes up with their answers. This unreliability prevents LLMs from being utilized to their fullest potential, especially in high-stakes and mission-critical settings, where accuracy is of utmost importance.

FORAGER enters to engender trust into AI. FORAGER enables LLMs to frame their answers with hard evidence and showcase their reasoning. This is more than *Chain-of-Thought* reasoning, where all reasoning is based solely on the model's training data, internal and unverified. FORAGER adds a layer of **real-world verification** on top of LLM reasoning. By validating each **atomic claim** the LLM makes, FORAGER unlocks a level of auditability that has yet to be seen with LLMs.

## 🧱 FORAGER Tech Stack
FORAGER is built with a modular, research-focused stack that balances LLM flexibility with rigorous verification.

### 📦 Backend
- **Python 3.13** – Core implementation language
- **FAISS** – Efficient vector similarity search for chunk retrieval
- **PyMuPDF** – PDF parsing and layout-aware text extraction
- **BeautifulSoup4** – HTML parsing for web documents
- **GroqAPI** – LLM-driven claim generation and rephrasing
- **transformers (Hugging Face)** – BS Detection using `facebook/bart-large-mnli` for NLI

### 📊 Embeddings & Retrieval
- **SentenceTransformers** – Semantic embeddings for document chunks
- **FAISS Index** – Local in-memory vector store for chunk retrieval
- **Custom reranking** – Chunk reordering based on similarity and prior evaluation

### 🧠 Claim Evaluation & Reasoning
- **LLM-based extractors** – Atomic claim parsing from natural language
- **BS Detector** – Natural Language Inference (NLI) engine for factual alignment
- **Confidence Checker** – Heuristic scoring of evidence strength and alignment certainty
- **Prompt-Locked Loop Controller** – Core feedback loop enforcing claim trustworthiness

### 🖥️ Frontend
- **Streamlit** – Lightweight, fast-deploy UI for document upload, question input, and claim evaluation display
- **JSONL export** – Structured output for easy downstream parsing

### 🔧 Dev & Tooling
- **pytest** – Unit testing
- **black** / **flake8** – Code formatting and linting
- **Markdown / YAML** – Configurable prompts and evaluation logs

## ⚙️ How FORAGER Works
FORAGER operates through three main subprocesses: Document Ingestion, Claim Evaluation, and the Prompt-Locked Loop.

<p align="center">
  <img src="https://sgordon-demo-s3.s3.us-east-2.amazonaws.com/High-Level+FORAGER+Flowchart.png" alt="FORAGER Pipeline" width="400"/>
</p>

### 📥 Ingestion
FORAGER begins by ingesting source documents. These are PDFs and HTML files that the user has collected, all about a specific topic of interest to the user. For example, a professor may choose to upload her lecture notes and textbook, or a computer engineer may choose to upload all of his project requirements. These documents are stored locally on the user's machine.

### ✂️ Chunking
Once a corpus has been uploaded, FORAGER chunks all of the documents into sections of 300 [**tokens**](https://learn.microsoft.com/en-us/dotnet/ai/conceptual/understanding-tokens) each, with an overlap of 50 tokens between each chunk. This overlap prevents context from being lost between chunks.

### 🧬 Embedding
Each chunk is then embedded into a [**vector**](https://www.pinecone.io/learn/vector-embeddings/), which is a list of 700+ numbers that represents the **semantic content** of the chunk. Each vector represents what the chunk actually *means*, as opposed to what it says. For example, "cut" and "slice", despite being different words, generally mean the same thing (to divide into pieces with a knife), and can be said to have similar semantic content.

Each vector embedding is stored in the FAISS index, a database that allows vectors to be queried and retrieved.

### 🧠 Claim Extraction
When the LLM generates a response to a user's question, the generated response is parsed to isolate **atomic claims**. Each atomic claim must express only one complete fact in a standalone full sentence that is understandable without additonal context. This step is crucial to enable full traceability. When interacting with LLMs traditionally, the answer to your question may be an 8 sentence paragraph, but there may only be 3 or 4 actual atomic claims within that paragraph.

<p align="center">
  <img src="https://sgordon-demo-s3.s3.us-east-2.amazonaws.com/Doc+Ingestion+FORAGER+Flowchart.png" alt="FORAGER Pipeline" width="400"/>
</p>

### 🔍 BS Detection
This is the first step in FORAGER's claim evaluation process. It is meant to, well—detect BS. The BS Detector uses [**Natural Language Inference (NLI)**](https://en.wikipedia.org/wiki/Textual_entailment) to check whether a claim is entailed, contradicted, or neutral with respect to the document(s) it is based in. In cases where NLI is neutral or uncertain, the BS Detector falls back to **vector similarity**. The primary goal of this step is to validate factual alignment between LLM-generated claims and the retrieved documents. The BS Detector assigns each claim a label of "Supported", "Unsupported", or "Contradicted", depending on the degree of entailment between the claim and its supporting document(s). The BS Detector can be thought of as a fact-checking judge asking "Is this claim logically and semantically supported by the source documents?".

### 📊 Confidence Checking
If the BS Detector is a fact-checking judge, the Confidence Checker is an assistant that says "I am X% sure that the judge made the right call here". It assesses how reliable or certain the BS Detector's label is by incorporating the length/clarity of the supporting evidence, agreement among multiple supporting chunks, and how far the similarity score is from the thresholds. It determines whether a claim's status is trustworthy enough to lock in the Prompt Locked Loop.

<p align="center">
  <img src="https://sgordon-demo-s3.s3.us-east-2.amazonaws.com/Claim+Evaluation+FORAGER+Flowchart.png" alt="FORAGER Pipeline" width="400"/>
</p>

### 🔒 Locking / Rephrasing
This is the Prompt-Locked Loop, the main engine of the pipeline. Claims enter the Prompt Lock Loop and exit when they are "locked," meaning that they are deemed "Supported" and "High Confidence" by the BS Detector and Confidence Checker, respectively. If a claim does not lock (it is not reliable enough to exit yet), FORAGER will either:
- Rephrase the original claim using the LLM to better reflect its supporting document(s), or
- Rerank the retrieved chunks and re-evaluate with the same claim.

This process repeats, refining and retrying, until the claim locks, or the system gives up after a specific number of iterations, since the claim can no longer be improved. This mechanism enforces stability and trustworthiness, rather than simply accepting the LLM's first response. It ensures that the final claims are not only plausible but backed by evidence and verfiably correct.

<p align="center">
  <img src="https://sgordon-demo-s3.s3.us-east-2.amazonaws.com/Prompt+Locked+Loop+FORAGER+Flowchart.png" alt="FORAGER Pipeline" width="400"/>
</p>

## 🔁 What is the Prompt-Locked Loop?
**Prompt-locked Loop** is a term coined by our team to refer to the main control mechanism behind FORAGER that iteratively refines AI-generated claims until they "lock" into a stable, supported, and high-confidence output. Computer engineers may notice the obvious inspiration we took from [**Phase-locked loop**](https://en.wikipedia.org/wiki/Phase-locked_loop), which synchronizes an output signal with a reference signal in computer electronics. Much how a Phase Lock Loop aligns those signals, our Prompt Locked Loop aligns an AI-generated claim with ground-truth documents.

## 🚀 Features
- 🔍 Claim-by-Claim Verification
    - FORAGER splits the LLM's answer into standalone atomic claims and verifies each one individually, rather than treating the response as a whole.
- 🤖 BS Detection with NLI
    - Uses facebook/bart-large-mnli to evaluate whether claims are supported, unsupported, or contradicted by the retrieved evidence.
- 🔁 Prompt-Locked Loop
    - Iteratively rephrases or re-evaluates claims until they meet both factual alignment and confidence thresholds, ensuring only trustworthy outputs are locked.
- 🧩 Modular Pipeline
    - Swap components in and out (LLMs, NLI models, document sources, or chunking strategies) with little friction.
- 📄 Structured Outputs
    - Export results as JSONL for downstream use, including full claim metadata and traces.
🧪 Built for High-Stakes QA
    - Designed with scientific, engineering, and technical domains in mind, where factual precision and auditability is needed.

## 📘 Usage
FORAGER is accessible via a simple front-end interface; no coding is required.

1. **Start the App**
- Run the following command from the root `FORAGER` folder:
```bash
streamlit run FORAGER/streamlit_frontend/new_app.py
```

2. **Upload Source Documents**
- Supports: PDF, HTML (more formats coming soon)
- Documents are stored locally and embedded automatically once the "Process Document(s)" button is clicked

3. **Submit a Question**
- Enter your research or task-related question (e.g., "How does chiplet integration affect power efficiency?")

4. **Inspect Verified Claims**
- View atomic claims extracted from the LLM's answer in the Claims Breakdown tab.
- Each claim is labeled as Supported, Unsupported, or Contradicted.
- Confidence level and supporting document chunks are shown.

5. **Trace Claim Evolution**
- For claims that are refined or rephrased, the full Prompt-Locked Loop trace is shown, round-by-round, in the Claims Breakdown tab.

6. **Export Results**
- Save the final claims and metadata as .jsonl for downstream use.

## 🛠️ Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/FORAGER.git
cd FORAGER
pip install -r requirements.txt
```

## 📖 API Documentation
Looking for function-level documentation?  
Check out the full [FORAGER API Reference](https://sgordon2004.github.io/FORAGER/FORAGER.html) 📚

## 🚧 Roadmap
- [ ] Support for .txt, hyperlink, and .md ingestion
- [ ] User-specified model selection
- [ ] NLP claim extraction using spaCy
- [ ] Exportable JSONL showing results