# Annual Report Analyst: multimodal RAG over company annual reports
import os, json, re, time
from pathlib import Path
import numpy as np
import gradio as gr
from PIL import Image
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from google import genai
from google.genai import types

HERE = Path(__file__).parent
DATA = HERE / "data"

# loading the index (fine-tuned embeddings, original content, chart images)
docstore = json.loads((DATA / "docstore.json").read_text())
ids = json.loads((DATA / "ids.json").read_text())
doc_text = json.loads((DATA / "doc_text.json").read_text())
M = np.load(DATA / "embeddings.npy")
embedder = SentenceTransformer(str(HERE / "model"), device="cpu")

# building keyword search
TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
def tok(s): return TOKEN.findall(s.lower())
bm25 = BM25Okapi([tok(doc_text[i]) for i in ids])
comp_arr = np.array([docstore[i]["company"] for i in ids])
year_arr = np.array([docstore[i]["year"] for i in ids])

COMPANIES = sorted(set(comp_arr.tolist()))
YEARS = sorted(set(year_arr.tolist()))
ALIASES = {c: [c.replace("_", " ")] for c in COMPANIES}
ALIASES.setdefault("au_small_finance_bank", []).extend(["au bank", "au sfb"])
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

def pretty_company(c): return c.replace("_", " ").title()
def fy_label(y): return f"FY {int(y) - 1}-{y[2:]}"

# parsing company and fiscal year from the question
def find_years(q):
    ys, q = set(), q.lower()
    for m in re.finditer(r"(20\d{2})\s*[-–/]\s*(\d{2,4})", q):
        end = m.group(2); ys.add(end if len(end) == 4 else "20" + end)
    q = re.sub(r"(20\d{2})\s*[-–/]\s*(\d{2,4})", " ", q)
    for m in re.finditer(r"fy\s*'?(\d{2,4})\b", q):
        y = m.group(1); ys.add(y if len(y) == 4 else "20" + y)
    q = re.sub(r"fy\s*'?(\d{2,4})\b", " ", q)
    ys.update(re.findall(r"\b(20\d{2})\b", q))
    return sorted(y for y in ys if y in YEARS)

def parse_query(q):
    ql = q.lower()
    company = next((c for c, names in ALIASES.items() if any(n in ql for n in names)), None)
    return company, find_years(q)

# hybrid search: fine-tuned dense + BM25, fused with reciprocal rank fusion
def filter_mask(company, years):
    m = np.ones(len(ids), bool)
    if company: m &= comp_arr == company
    if years:   m &= np.isin(year_arr, years)
    return m

def ranked(scores, mask, n):
    s = np.where(mask, scores, -np.inf)
    top = np.argpartition(-s, min(n, len(s) - 1))[:n]
    return [ids[j] for j in top[np.argsort(-s[top])] if np.isfinite(s[j])]

def rrf(*lists, k=60):
    s = {}
    for L in lists:
        for r, i in enumerate(L, 1):
            s[i] = s.get(i, 0) + 1 / (k + r)
    return [i for i, _ in sorted(s.items(), key=lambda x: -x[1])]

def search(q, company, years, top=8):
    mask = filter_mask(company, years)
    qv = embedder.encode([QUERY_PREFIX + q], normalize_embeddings=True)[0]
    return rrf(ranked(M @ qv, mask, 30), ranked(bm25.get_scores(tok(q)), mask, 30))[:top]

# answering with Gemini, falling back to other models when one is busy
ANSWER_PROMPT = """You are a financial analyst assistant answering questions about company annual reports.
Rules:
- Use ONLY the sources provided below. Never use outside knowledge.
- Only report numbers that appear in the sources. You MAY compute a simple difference or
  % change between two figures stated in the sources, and must show the calculation.
  NEVER derive new quantities (e.g. back-calculating a total from a percentage).
- Quote exact numbers with their units (₹ crore, %, etc.) and fiscal year.
- Cite every fact like [AU Small Finance Bank FY2024, p.148].
- Include only points that directly answer the question; leave out tangential data.
- If the sources don't contain the answer, reply: "I couldn't find this in the provided reports."
- Keep it concise: a direct answer first, then 1-3 supporting points."""

ANSWER_MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"]  # lite first: fastest, and the model used in evaluation
API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY, http_options=types.HttpOptions(timeout=45_000)) if API_KEY else None  # 45 s max per call

RETRYABLE = ("500", "502", "503", "504", "UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL", "timed out", "timeout")

def generate(contents):
    for model in ANSWER_MODELS:
        t = time.time()
        try:
            r = client.models.generate_content(model=model, contents=contents,
                    config=types.GenerateContentConfig(temperature=0))
            print(f"[answer] {model} ok in {time.time() - t:.1f}s")
            return r.text, model
        except Exception as ex:
            msg = str(ex)
            print(f"[answer] {model} failed after {time.time() - t:.1f}s: {msg[:150]}")
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or any(c in msg for c in RETRYABLE):
                continue                                    # busy, slow or out of quota: next model right away
            raise                                           # real bug: show it
    return "All models are busy right now, please try again in a minute.", "none"

def cite(d):
    return f"[{pretty_company(d['company'])} FY{d['year']}, p.{d['page']}]"

def build_contents(q, hits):
    contents = [ANSWER_PROMPT, f"QUESTION: {q}", "SOURCES:"]
    for k, i in enumerate(hits, 1):
        d = docstore[i]
        if d["type"] == "image" and d.get("image"):
            contents += [f"Source {k} {cite(d)}: chart. Summary: {d.get('summary', '')}",
                         Image.open(DATA / "images" / d["image"])]
        elif d["type"] == "table":
            contents.append(f"Source {k} {cite(d)}: table\n{d['content'][:6000]}")
        else:
            contents.append(f"Source {k} {cite(d)}: text\n{d['content']}")
    return contents

COMPANY_CHOICES = {"Auto-detect": None, **{pretty_company(c): c for c in COMPANIES}}
YEAR_CHOICES = {"Auto-detect": None, **{fy_label(y): y for y in YEARS}}

def ask(question, company_choice="Auto-detect", year_choice="Auto-detect"):
    if not question or not question.strip():
        return "Please type a question.", "", []
    if client is None:
        return "GEMINI_API_KEY is not set. Add it as a secret to use the app.", "", []
    company, years = parse_query(question)
    if COMPANY_CHOICES.get(company_choice): company = COMPANY_CHOICES[company_choice]
    if YEAR_CHOICES.get(year_choice):       years = [YEAR_CHOICES[year_choice]]

    t0 = time.time()
    hits = search(question, company, years)
    t_ret = time.time() - t0
    text, model = generate(build_contents(question, hits))
    t_ans = time.time() - t0 - t_ret

    lines = []
    for k, i in enumerate(hits, 1):
        d = docstore[i]
        preview = (d.get("summary") or d["content"]).replace("\n", " ")[:160]
        lines.append(f"**{k}.** {cite(d)} · *{d['type']}* · {preview}…")
    filt = f"{pretty_company(company) if company else 'any company'}, " \
           f"{', '.join(fy_label(y) for y in years) if years else 'any year'}"
    sources_md = "\n\n".join(lines) + \
        f"\n\n<sub>Filters: {filt} · retrieval {t_ret:.2f}s · answer {t_ans:.1f}s by {model}</sub>"
    gallery = [(str(DATA / "images" / docstore[i]["image"]), cite(docstore[i]))
               for i in hits if docstore[i]["type"] == "image" and docstore[i].get("image")]
    return text, sources_md, gallery

EXAMPLES = [
    "What was the gross NPA closing balance in FY 2023-24?",
    "How did total advances change between FY23 and FY24?",
    "What are the key risks highlighted by management in FY2025?",
    "Show me the business segments of AU Small Finance Bank in FY2023",
    "What was the total deposit amount in FY 2024-25?",
]

with gr.Blocks(title="Annual Report Analyst", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📊 Annual Report Analyst\n"
                "Ask questions about company annual reports. Answers come only from the reports, "
                "with page citations, and draw on text, tables and charts.")
    with gr.Row():
        q = gr.Textbox(label="Your question", placeholder="e.g. What was net profit in FY 2024-25?", scale=4)
        btn = gr.Button("Ask", variant="primary", scale=1)
    with gr.Row():
        comp = gr.Dropdown(list(COMPANY_CHOICES), value="Auto-detect", label="Company")
        yr = gr.Dropdown(list(YEAR_CHOICES), value="Auto-detect", label="Report year")
    ans = gr.Markdown()
    with gr.Accordion("Sources", open=True):
        src = gr.Markdown()
        gal = gr.Gallery(label="Charts used", columns=3, height="auto")
    gr.Examples(EXAMPLES, inputs=q)
    btn.click(ask, [q, comp, yr], [ans, src, gal])
    q.submit(ask, [q, comp, yr], [ans, src, gal])

if __name__ == "__main__":
    demo.launch()
