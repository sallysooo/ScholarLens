# 🔍ScholarLens

*A lightweight research assistant that helps you **read**, **discover**, and **question** academic papers—all in a single notebook.*

---

## Table of Contents
1. [Overview](#overview)  
2. [Key Features](#key-features)  
3. [Demo Walkthrough](#demo-walkthrough)  
4. [Tech Stack](#tech-stack)  
5. [Example Usage](#example-usage)  
6. [Acknowledgments](#acknowledgments)  

---

## Overview

**ScholarLens** transforms a Jupyter notebook into a **three-in-one research companion**:

| # | Capability     | Description                                               |
|---|----------------|-----------------------------------------------------------|
| 1 | **Summarize**  | Generates concise, section-wise summaries of academic papers. |
| 2 | **Find Similar** | Retrieves related papers from arXiv via dense-vector search. |
| 3 | **Ask & Answer** | Answers natural-language questions using the paper as context. |

> Built for fast prototyping—no backend server, no database, just Python and GPUs.

---

## Key Features

- ✂️ **Section-aware summarization** with *DistilBART (CNN/DM)*  
- 🔍 **Semantic search** using *BGE-large-en v1.5* embeddings + *FAISS* FlatIP index  
- 🤖 **Extractive QA** via *deepset/roberta-base-squad2* and a sliding-window context chunker  
- 🧠 **Fully in-memory**: no external storage or cloud APIs required  
- 🔧 **Modular and extensible**: swap out models or integrate RAG pipelines easily  

---

## Demo Walkthrough

```
flowchart LR
    A[Upload PDF] --> B[PyMuPDF Text Extractor]
    B --> C[Summarizer]
    B --> D[Q&A]
    E[User Query] --> F[Embed (BGE)]
    F --> G[FAISS Index]
    G --> H[Return Similar Papers]
```

1. Open `ScholarLens.ipynb` in Jupyter.  
2. Run the first three cells to install dependencies.  
3. Upload a PDF or enter an arXiv ID.  
4. Try the widgets or CLI prompts:  
   - 📝 *Summarize* → quick abstracts and conclusions  
   - 🔎 *Search* → top-k related papers with links  
   - ❓ *Ask* → free-form Q&A on the content  

---

## Tech Stack

| Layer             | Library / Model                  | Description                                 |
|------------------|----------------------------------|---------------------------------------------|
| **Language Models** | 🤗 *transformers*               | DistilBART‑CNN‑12‑6, RoBERTa‑SQuAD2          |
| **Embeddings**     | *sentence-transformers*         | BGE-large-en v1.5 (1536-d)                   |
| **Vector Index**   | *FAISS*                         | `IndexFlatIP` (can switch to IVF/HNSW)       |
| **PDF Parsing**    | *PyMuPDF (fitz)*                | Fast and layout-aware                       |
| **Data Source**    | *arXiv API*                     | Metadata + direct PDF links                 |
| **Runtime**        | Python 3.9+, Jupyter Notebook   | GPU optional, recommended for speed         |

---

## Example Usage

```python
from src.summarizer import summarize_pdf
from src.search import ArxivPaperSearch
from src.qa import answer_question

# 1. Summarize a paper
summary = summarize_pdf("./papers/attention_is_all_you_need.pdf")
print(summary["Conclusion"])

# 2. Build & query the index
searcher = ArxivPaperSearch(category="cs.LG", max_results=500)
searcher.build_index()
results = searcher.search("Graph Attention Networks", top_k=5)
for r in results: print(r.title, r.pdf_url)

# 3. Ask a question
answer = answer_question("./papers/gan.pdf", "What loss function is used?")
print(answer)
```


---

## Acknowledgments

- **Hugging Face** for open models & datasets  
- **FAISS** team for lightning-fast vector search  
- **arXiv** for accessible paper metadata  
- Inspired by `papers-with-code` and the academic ML tooling community  

---

*Happy researching!* :)
