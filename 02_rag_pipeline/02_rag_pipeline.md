# 02 – Multimodal RAG Pipeline

`02_rag_pipeline.ipynb` turns parsed annual-report elements (text, tables and images) into a question-answering system that answers from text, tables **and charts**, cites the exact page for every fact, and refuses when the answer isn't in the reports.

Data: three years (FY2023–FY2025) of AU Small Finance Bank annual reports.

## Example

**Q: What was the gross NPA closing balance in FY 2023-24?**

> The gross NPA closing balance in FY 2023-24 was **₹ 1,237.40 Crore** [AU Small Finance Bank FY2024, p.148].
> - Opening balance ₹ 981.31 Crore, additions ₹ 1,698.19 Crore, reductions ₹ 1,442.10 Crore [FY2024, p.148]
> - Gross NPA ratio of 1.67% of gross advances [FY2024, p.122, p.149]

**Q: Who is the CEO of Google?**

> I couldn't find this in the provided reports.

## What the notebook does

1. **Cleans text.** Custom PDF fonts store the ₹ symbol as "H"; targeted regexes restore it without touching words like "Holding".
2. **Filters images with CLIP.** Zero-shot classification keeps charts and drops photos, logos and icons (537 → 208 images).
3. **Removes duplicates.** Perceptual hashing removes repeated images within each report (208 → 207).
4. **Summarizes tables and charts with Gemini.** 6 items per call, returned as JSON. Digital tables are sent as Markdown (exact numbers); charts are sent as images. Summaries are cached, and the job survives per-minute and daily quota limits with a fallback chain across models.
5. **Embeds and indexes.** bge-small embeddings stored in ChromaDB using a multi-vector pattern: search runs on summaries, but answers use the original table or chart image.
6. **Retrieves with metadata filters.** Company and fiscal year are parsed from the question ("FY 2023-24", "FY24", "2024") and applied before similarity search, so an FY24 question can't pull FY23 figures.
7. **Generates grounded answers.** Gemini answers only from retrieved sources, cites every fact by page, and sees retrieved charts as images.

## Numbers

| Stage | Result |
|---|---|
| Input elements | 3,127 text · 857 tables · 537 images |
| CLIP chart filter | 537 → 208 images |
| Perceptual-hash dedup | 208 → 207 |
| Tables summarized | 693 of 857 (tiny tables embedded directly) |
| Gemini summarization calls | 1,394 → 151 (~89% fewer) |
| Indexed in ChromaDB | 4,059 items: 3,127 text · 802 tables · 130 charts |

## A failure worth noting

An early answer to *"How did total advances change between FY23 and FY24?"* included a "total credit exposure" the model had back-calculated from a percentage. The figure appeared in no source, yet it still carried a citation. **A citation doesn't prove a number is supported.** The answer prompt was tightened to allow only stated figures (plus simple differences and % changes with the calculation shown), and the re-run answer contains only numbers that appear in the cited pages.

## Running it

**Input:** the parsed elements from notebook 01, as `processed/*.json` in `MyDrive/annual-report-rag/`.

1. Open the notebook in Colab with a **T4 GPU** runtime.
2. Add `GEMINI_API_KEY` to Colab Secrets.
3. Run all cells. CLIP scores, summaries and embeddings are cached to Drive, so reruns are fast and quota interruptions resume where they stopped.
4. Ask questions with `answer("your question")`.

**Libraries:** `google-genai`, `transformers` (CLIP), `sentence-transformers` (BAAI/bge-small-en-v1.5), `chromadb`, `imagehash`, `pandas`

## Limitations and next steps

- **Evaluation is manual so far.** Next: a labeled question set to measure retrieval hit@k, numeric answer accuracy and refusal rate.
- **Single company.** The company filter is implemented but untested across multiple issuers.
- **Automated number checking.** Verify that every number in an answer appears in its cited source, rather than relying on the prompt alone.
- **Relevance threshold.** Refuse before calling the model when no retrieved source scores above a cutoff.
