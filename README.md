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