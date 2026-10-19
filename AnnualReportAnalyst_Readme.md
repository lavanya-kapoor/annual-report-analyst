# Annual Report Analyst: Multimodal RAG over Financial Reports

## 1. Problem statement

Annual reports run to 150-250 pages, and much of the important information sits in **tables and charts**, not running text. Analysts spend hours finding a single figure, and many reports are partly **scanned**, so a plain text search can't read them at all.

Standard RAG (retrieval-augmented generation) systems only index text. They miss table values and chart trends, mix up figures from different years, and can produce numbers with a citation that looks right but isn't supported by the source.

**Goal:** a question-answering system over annual reports that:
- answers from **text, tables and charts**
- gives a **page citation** for every figure
- **refuses** when the answer isn't in the reports
- runs entirely on a **free stack**

## 2. What I did

I built an end-to-end multimodal RAG pipeline, from PDF parsing to a working web app, and measured every design choice instead of assuming it helped.

- **Parsed** digital and scanned PDFs, extracting text, tables and charts separately
- **Made tables and charts searchable** by turning them into text summaries, while sending the original table or chart image to the LLM for the answer
- **Built a test set** of 56 questions and evaluated 7 retrieval setups
- **Added** keyword search, a reranker, visual page retrieval and a fine-tuned embedding model, keeping only what improved the results
- **Shipped a Gradio app** that answers in about 1–3 seconds with citations, sources and the charts it used

**Result:** judged retrieval Hit@5 rose from **0.82 → 0.98**, and fully correct answers from **76.5% → 84.3%**.

## 3. Skills used

| Area | Skills and tools |
|---|---|
| Document AI | PDF parsing (Docling, PyMuPDF), OCR for scanned pages (EasyOCR), table structure extraction (TableFormer) |
| Retrieval | Embeddings (bge-small), vector search (ChromaDB), BM25 keyword search, Reciprocal Rank Fusion, cross-encoder reranking, metadata filtering |
| Multimodal | CLIP zero-shot image classification, perceptual-hash dedup, visual page retrieval (ColQwen2), vision LLM (Gemini) |
| LLMs | Prompt design for grounding and citations, batched LLM calls, LLM-as-judge evaluation, rate-limit handling and model fallback |
| ML | Fine-tuning embeddings on synthetic data (sentence-transformers, contrastive loss), leak-free train/test splits |
| Evaluation | Building a test set, Hit@k, MRR, answer grading, ablation studies, error analysis |
| Engineering | Python, numpy, pandas, Gradio, Google Colab (T4 GPU), caching and checkpointing, memory management |

## 4. How it was built: six stages, one folder each

Each folder has its own notebook and a **README** with the method, results and caveats.

| Step | What happens | Folder |
|---|---|---|
| 1. Ingest | Parse PDFs, extract text, tables and charts | `01_ingestion` |
| 2. Index & answer | Summarize tables and charts, build the index, generate cited answers | `02_rag_pipeline` |
| 3. Evaluate & improve search | Test set, keyword + hybrid search, reranker, answer grading | `03_evaluation_and_re-ranking` |
| 4. Add visual search | Retrieve whole page images and add them to the answer context | `04_Visual_Retrieval` |
| 5. Fine-tune | Train the embedding model on this domain | `05_fine_tune_embedding` |
| 6. Ship | Package the best setup into a web app | `06_app` |

### [01 · Ingestion](01_ingestion/)
Classifies every page as digital, scanned or mixed, and runs OCR only where it's needed. Docling extracts text in structure-aware chunks, tables as Markdown and charts as images, each tagged with company, year and page. Reports are processed in 40-page batches with checkpoints so Colab doesn't run out of memory.
*Details in [`01_ingestion/01_readme.md`](01_ingestion/README.md).*

### [02 · RAG pipeline](02_rag_pipeline/)
A CLIP filter keeps real charts and drops logos and photos, and near-duplicate images are removed. Gemini writes summaries of tables and charts in batches, cutting about 1,400 API calls to about 150. Everything is embedded into ChromaDB, with the company and fiscal year parsed from the question used as filters. Gemini answers from the original tables and chart images with page citations. This stage also found and fixed a hallucinated figure.
*Details in [`02_rag_pipeline/02__readme.md`](02_rag_pipeline/README.md).*

### [03 · Evaluation and re-ranking](03_evaluation_and_re-ranking/)
Builds a 56-question test set (tables, text, charts, hand-written and off-topic questions). Compares 7 retrieval setups: dense, BM25, hybrid (RRF), cross-encoder reranker and fusions. An LLM judge scores relevance, because many "misses" turned out to be valid answers on other pages. Answers are graded end to end, including whether the system correctly refuses.
*Details in [`03_evaluation_and_re-ranking/03_README`](03_evaluation_and_re-ranking/README.md).*

### [04 · Visual retrieval](04_Visual_Retrieval/)
Embeds every page as an image with ColQwen2, so layout and charts can be searched without any text extraction. On its own it scored lower than text search, but adding the top page images to the answer context raised answer accuracy.
*Details in [`04_Visual_Retrieval/04_readme.md`](04_Visual_Retrieval/README.md).*

### [05 · Fine-tuning embeddings](05_fine_tune_embedding/)
Generates question–passage training pairs from the reports themselves, with no manual labelling. Fine-tunes bge-small in 1 minute on a free T4 GPU, holding out one full report and every test answer page to keep the evaluation fair.
*Details in [`05_fine_tune_embedding/05_README.md`](05_fine_tune_embedding/README.md).*

### [06 · App](06_app/)
A Gradio app using the best fast setup: fine-tuned hybrid search, company and year filters, and Gemini with chart images in the context. It shows the answer, its sources and the charts used, and falls back to another model if one is busy.
*Details in [`06_app/README.md`](06_app/06_readme.md).*

## 5. Findings

**Retrieval** (LLM-judged: a retrieved passage actually answers the question)

| Setup | Judged Hit@5 | Judged MRR |
|---|---|---|
| Dense search + company/year filters | 0.824 | 0.611 |
| + BM25 hybrid search (RRF) | 0.922 | 0.755 |
| **+ fine-tuned embeddings** | **0.980** | **0.821** |

**Answers** (same model and prompt, only the retrieval changes)

| Retrieval | Fully correct | Wrongly refused | Off-topic refused |
|---|---|---|---|
| Dense + filters | 76.5% | 13.7% | 5/5 |
| Hybrid (RRF) | 80.4% | 11.8% | 5/5 |
| **Hybrid + visual page images** | **84.3%** | **7.8%** | **5/5** |

*51 answerable + 5 off-topic questions; one question ≈ 2 points.*

- **Metadata filters were the biggest single win.** Parsing company and year from the question lifted strict Hit@5 from 0.53 to 0.73, because consecutive reports repeat the same topics with different numbers.
- **Keyword search matters for finance.** BM25 beat embeddings on table questions (0.90 vs 0.76), because they hinge on exact terms and figures.
- **The reranker wasn't worth it.** It only won under strict labels and costs about 35 s per query on CPU, so the app uses hybrid search.
- **Many "failures" were test-set errors:** ambiguous questions, labels pointing at prior-year tables, and valid answers on other pages. That's why LLM-judged relevance was added.
- **Fine-tuning gave the largest retrieval gain.** 1 minute of training took judged Hit@5 from 0.92 to 0.98, and from 0.85 to 0.95 on a report held out of training. It fixed 5 questions and broke none.
- **Visual retrieval works best as a supplement.** It was weaker alone (0.77 vs 0.94), but adding two page images raised answer accuracy from 80.4% to 84.3%.
- **A citation doesn't prove a number is supported.** The model once back-calculated a figure that appeared in no source and still cited a page. A prompt rule fixed it.

## 6. Moving forward

The pipeline is built to scale: nothing is specific to one company, and every stage is cached and modular. The next steps build on what already works:

- **Scale to more companies and sectors.** The ingestion and indexing stages are company-agnostic, so new reports can be added by dropping PDFs into a folder. The next test is a company the fine-tuned model has never seen, to confirm the 0.85 → 0.95 gain on the held-out report carries across companies.
- **Test with real analyst questions.** The 56-question test set and LLM judge are already in place, so adding analyst-written questions is a direct extension of the current evaluation.
- **Measure scanned-page accuracy.** Scanned and mixed pages are already detected and routed to OCR; a separate score for them would show how well this part holds up.
- **Cut wrong refusals further.** Better retrieval has already brought them down from 13.7% to 7.8%. Adding the fine-tuned model to the answer step, with fewer, more relevant sources, should bring them lower.
- **Label the data period in every summary.** Year filters already prevent cross-report mix-ups; stating the period of each table (e.g. "FY2022-23 comparatives") would handle prior-year figures inside the same report.
- **Host it permanently.** `06_app/app.py` and `requirements.txt` are deployment-ready, and search runs in under 0.5 s on CPU, so the app can run on a small, low-cost server.
---

**Run it yourself:** add a free Gemini API key to Colab Secrets as `GEMINI_API_KEY`, put PDFs in Google Drive at `annual-report-rag/reports/<company>/<year>.pdf`, and run the notebooks in folder order on a T4 runtime. Notebook 6 launches the app with a public link. Large files (PDFs, indexes, the fine-tuned model) stay on Google Drive, not in this repo.

**License:** MIT
