# 05 - Fine-tuning the Embedding Model

`05_finetune_embeddings.ipynb` fine-tunes `BAAI/bge-small-en-v1.5`, the embedding model behind the text search in Notebooks 2–4, on question to passage pairs generated from the annual reports themselves. No manual labelling was needed.

## Headline results

| Search | Base model | Fine-tuned |
|---|---|---|
| Dense, Hit@5 (gold page) | 0.73 | **0.94** |
| Hybrid, Hit@5 (gold page) | 0.82 | **0.90** |
| Hybrid, Hit@5 (LLM-judged) | 0.92 | **0.98** |
| Dense, Hit@5 on the **unseen FY2024 report** | 0.65 | **0.90** |

Fine-tuning fixed 6 questions the base model missed and broke none. Training took **1.2 minutes on a T4 GPU**.

## Keeping the test fair

The model must not simply memorise the answers to the test questions, so the training data was built with two hold-outs:

1. **No FY2024 data at all.** Training uses only the FY2023 and FY2025 reports, so the 20 test questions about FY2024 measure how well the model carries over to a report it has never trained on.
2. **No answer pages.** Every page that answers a test question, or that the LLM judge marked relevant in Notebook 3, is excluded from training (81 pages).

## Training data

| Step | Result |
|---|---|
| Usable items (FY2023 + FY2025, not held out) | 1,680 |
| Randomly sampled | 900 (687 text · 184 tables · 29 charts) |
| Items with generated questions | 885 |
| Questions generated | 1,720 |
| Train / validation pairs (split by item) | 1,548 / 172 |

For each item, Gemini (`gemini-3.5-flash-lite`) wrote two questions it answers: a natural analyst question and a short keyword-style query. Generation took 113 calls and is cached.

## Training

- **Loss:** Multiple Negatives Ranking Loss. Every other passage in a batch acts as a wrong answer, so no negative examples have to be mined.
- **Batching:** a batch never holds the same passage twice, since both questions for an item share one passage.
- **Settings:** 3 epochs, batch size 32, learning rate 3e-5, 10% warmup, fp16.

| Epoch | Training loss | Validation loss |
|---|---|---|
| 1 | 0.265 | 0.052 |
| 2 | 0.133 | 0.046 |
| 3 | 0.079 | 0.048 |

Validation loss bottomed out at epoch 2 and rose slightly at epoch 3, an early sign that more epochs would start to overfit.

## Results

Both models search with exact similarity (numpy), using the same fiscal-year filters, so the comparison is like for like.

**All 51 answerable questions**

| Setup | Hit@5 | MRR | Judged Hit@1 | Judged Hit@5 | Judged MRR |
|---|---|---|---|---|---|
| Dense (base) | 0.725 | 0.554 | 0.490 | 0.824 | 0.611 |
| Dense (fine-tuned) | **0.941** | **0.728** | 0.667 | 0.941 | 0.776 |
| Hybrid (base) | 0.824 | 0.666 | 0.667 | 0.922 | 0.755 |
| Hybrid (fine-tuned) | 0.902 | 0.717 | **0.706** | **0.980** | **0.815** |

**Unseen FY2024 report only (20 questions)**

| Setup | Hit@5 | MRR | Judged Hit@1 | Judged Hit@5 | Judged MRR |
|---|---|---|---|---|---|
| Dense (base) | 0.65 | 0.589 | 0.55 | 0.75 | 0.625 |
| Dense (fine-tuned) | **0.90** | 0.754 | 0.65 | 0.90 | 0.754 |
| Hybrid (base) | 0.75 | 0.693 | **0.75** | 0.85 | 0.785 |
| Hybrid (fine-tuned) | **0.90** | **0.758** | **0.75** | **0.95** | **0.829** |

**Judged Hit@5 by question type**

| Setup | Hand-written | Chart | Table | Text |
|---|---|---|---|---|
| Dense (base) | 0.75 | 1.00 | 0.76 | 0.81 |
| Dense (fine-tuned) | 0.75 | 1.00 | **1.00** | 0.88 |
| Hybrid (base) | 0.75 | 1.00 | 0.95 | 0.88 |
| Hybrid (fine-tuned) | **1.00** | 1.00 | **1.00** | **0.94** |

- **Tables gained most.** Dense search on table questions went from 0.76 to 1.00: the model learned how analysts phrase questions about financial tables.
- **The gain holds on the unseen report.** Dense Hit@5 on FY2024 rose from 0.65 to 0.90, so the model learned the domain, not just the training pages.
- **Fine-tuned dense search now beats base hybrid search** (Hit@5 0.94 vs 0.82), and fine-tuning still adds on top of hybrid.
- **Questions fixed include** "total reduction in gross non-performing advances for FY 2022-23" and "adjusted profit after tax… FY 2023-24", which text search had missed in Notebooks 3 and 4.

## Caveats

- **Same question style.** Training and test questions were both written by Gemini with similar prompts, so part of the gain may come from learning that style rather than real analyst phrasing.
- **FY2024 isn't completely unseen in content.** The FY2025 report repeats many FY2024 figures as comparatives.
- **Small samples:** 51 questions overall and 20 for FY2024; one question is about 2–5 points.
- **Search quality only.** Answer accuracy with the fine-tuned model wasn't measured here.

## Outputs

- **Model:** `models/bge-small-annual-reports/` on Drive
- **Embeddings:** `cache/vecs_finetuned.npy`
- **Index:** a new `annual_reports_ft` collection in the Chroma index, alongside the original `annual_reports`
- **Results:** `eval/finetune_results.csv`; training questions in `eval/train_questions.json`

## Running it

**Input:** `docstore.json`, the Chroma index, `eval/golden.csv` and `eval/relevance_cache.json` from Notebooks 2–3, in `MyDrive/annual-report-rag/`.

1. Open the notebook in Colab with a **T4 GPU** runtime.
2. Add `GEMINI_API_KEY` to Colab Secrets.
3. Run all cells. Training questions and relevance judgments are cached, so reruns make no new question-generation calls.

**Libraries:** `sentence-transformers`, `datasets`, `accelerate`, `chromadb`, `rank_bm25`, `google-genai`, `pandas`
