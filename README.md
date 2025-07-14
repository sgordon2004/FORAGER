# FORAGER

**FORAGER** (Fact-Oriented Responsible AI-Guided Engineering Research) is a verification pipeline designed to improve trust in Large Language Models (LLMs). It combines **Prompt Lock Looping**, **Retrieval-Augmented Generation (RAG)**, and **Natural Language Inference (NLI)** techniques to validate AI-generated claims against reliable evidence.

---

## 🚀 Features
- ✅ Multi-stage claim verification
- ✅ BS Detector leveraging `facebook/bart-large-mnli`
- ✅ Modular ingestion, chunking, and embedding pipelines
- ✅ Structured JSONL formatting and claim aggregation
- ✅ Supports engineering research domains (e.g., semiconductors, HI)

---

## 🛠️ Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/FORAGER.git
cd FORAGER
pip install -r requirements.txt
```

## 📘 Usage

FORAGER is accessible via a simple front-end interface; no coding is required.

### Typical Flow via UI
1. **Upload** your documents (PDF, HTML, or Markdown).
2. **Submit a query** to the LLM via the app interface.
3. **View extracted claims** and their verification status directly in the dashboard.
4. **Download** or **export** results for further use.

---

### Backend Workflow (For Reference Only)
Internally, FORAGER follows this flow:
- Ingestion → Chunking → Embedding → Claim Evaluation

This process is fully automated through the UI.