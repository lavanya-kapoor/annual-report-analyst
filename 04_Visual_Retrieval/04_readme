# 04 - Visual Retrieval (ColQwen2) and Fusion

`04_visual_retrieval_colpali.ipynb` adds a second search path that skips parsing entirely: it searches **images of whole report pages** with ColQwen2, then combines that with the hybrid text search from Notebook 3 and measures both retrieval and answer accuracy on the same 56-question test set.

- **Path A (text):** hybrid dense + BM25 search over parsed text, tables and chart summaries, the best setup from Notebook 3.
- **Path B (visual):** every page rendered as an image and searched directly. No OCR, no parsing, no summaries.

## Headline results

| | Hybrid text (Notebook 3) | Hybrid + visual pages |
|---|---|---|
| Answer accuracy | 80.4% | **84.3%** |
| Wrongly refused | 11.8% | **7.8%** |
| Out-of-scope questions refused | 5/5 | 5/5 |

Visual search was weaker than text search on its own, but adding its top page images to the model's context improved the final answers.

## How Path B works

1. Each page of the 3 reports (692 pages) is rendered at 96 DPI with PyMuPDF.
2. **ColQwen2** (`vidore/colqwen2-v1.0`, built on Qwen2-VL-2B) turns each page into a set of patch-level vectors.
3. A question is encoded the same way, and pages are ranked by **late interaction**: each question token finds its best-matching patch on the page, and those scores are summed.
4. The same company and fiscal-year filters as Path A are applied before scoring.

Embedding all 692 pages took about **11 minutes on a T4 GPU** and produced 134 MB of embeddings, saved to Drive in 40-page chunks so a disconnect only loses one chunk.

## Retrieval comparison (page level)

Both paths are compared on pages, since Path B returns pages rather than individual tables or chunks.

| Variant | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|---|
| A. Text hybrid | **0.59** | **0.71** | **0.82** | 0.90 | **0.67** |
| B. Visual pages | 0.37 | 0.53 | 0.59 | 0.73 | 0.48 |
| A+B fused (RRF) | 0.49 | **0.71** | 0.80 | **0.94** | 0.62 |

**LLM-judged** (does the page actually answer the question?):

| Variant | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| A. Text hybrid | **0.77** | **0.92** | **0.94** | **0.84** |
| B. Visual pages | 0.59 | 0.71 | 0.77 | 0.65 |
| A+B fused | 0.67 | 0.86 | 0.92 | 0.77 |

New pages from the visual path were judged by Gemini from the page image (407 page judgments, cached).

- **Text search wins on its own.** These reports are digital PDFs with cleanly parsed tables, which plays to text search's strengths.
- **Fusion trades ranking for coverage.** It found more correct pages in the top 10 but pushed some below wrong ones at the top.
- **The visual path rescued 3 questions** that text search missed entirely, including the adjusted profit-after-tax table (FY2024, p.69).

## Answer accuracy

The fused setup sends the model the top 6 items from Path A plus up to 2 extra **full page images** from Path B (pages not already covered). All setups use the same model (`gemini-3.5-flash-lite`) and prompt, and are graded by Gemini against the reference answers.

| Retrieval | Correct | Correct + partial | Wrongly refused | Unanswerable refused |
|---|---|---|---|---|
| Dense + filters | 76.5% | 78.4% | 13.7% | 5/5 |
| Hybrid (RRF) | 80.4% | 80.4% | 11.8% | 5/5 |
| **Hybrid + visual pages** | **84.3%** | **84.3%** | **7.8%** | 5/5 |

**Correct by question type:**

| Retrieval | Hand-written | Chart | Table | Text |
|---|---|---|---|---|
| Dense + filters | 0.75 | 1.00 | 0.71 | 0.69 |
| Hybrid (RRF) | 0.75 | 1.00 | 0.76 | 0.75 |
| Hybrid + visual pages | 0.75 | 1.00 | 0.76 | **0.88** |

Refusals here are detected from the answer text itself ("I couldn't find this in the provided reports"), which independently confirms the 5/5 refusal figure from Notebook 3.

## Caveats

- The accuracy gain over hybrid is **2 questions out of 51** (41 to 43).
- The setups don't send identical context: fused sends 6 text items + 2 page images, hybrid sends 8 text items.
- Visual-path pages were judged from page images, while text-path pages mostly reused Notebook 3's item-level judgments, so the judged comparison isn't perfectly like-for-like.
- Test questions were generated from the parsed content, which may favour text search.

## Running it

**Input:** `docstore.json`, the ChromaDB index, `metadata.csv` (report paths and page counts) and `eval/golden.csv` from Notebooks 1–3, in `MyDrive/annual-report-rag/`.

1. Open the notebook in Colab with a **T4 GPU** runtime (required: ColQwen2 uses about 9 GB of GPU memory).
2. Add `GEMINI_API_KEY` to Colab Secrets.
3. Run all cells. Page embeddings are saved to `page_embeddings/`, and page judgments, answers and grades are cached in `eval/`, so reruns skip completed work.

**Outputs saved to `eval/`:** `page_retrieval_strict.csv`, `page_retrieval_judged.csv`, `answer_accuracy_final.csv`

**Libraries:** `colpali-engine` (ColQwen2), `pymupdf`, `chromadb`, `sentence-transformers`, `rank_bm25`, `google-genai`, `pandas`
