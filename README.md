# FinSight AI
**Agentic RAG for Financial Document Intelligence & Compliance Screening**

Built for: *The Arch — RAG and Agentic AI Hackathon*, IIT Kharagpur (Finance track)

## Problem Statement

Credit and compliance analysts manually read financial statements, loan files,
and internal suspicious activity reports to answer routine but high-stakes
questions: *Does this borrower meet its loan covenants? Are there AML red
flags in this account history? What's the current ratio for this company?*
This is slow, inconsistent across analysts, and hard to audit after the fact.

FinSight is an agentic RAG assistant that lets an analyst ask these questions
in natural language. It retrieves grounded answers from the actual documents
(not model memory), and uses an agentic tool-calling loop to perform
financial calculations and rule-based AML/compliance screening on top of
what it retrieves — with every step shown in a visible reasoning trace so
the output is auditable, not a black box.

## Why this is "agentic," not just RAG

A plain RAG chatbot retrieves text and summarizes it. FinSight's agent
decides, per query, whether it needs to:
1. **Retrieve** relevant document chunks (`search_documents`)
2. **Calculate** a derived financial metric from retrieved figures (`calculate_financial_ratio`)
3. **Screen** retrieved text against an AML red-flag ruleset (`flag_compliance_risk`)
4. Chain these in sequence (e.g. retrieve → screen → explain) before answering

The model itself chooses which tools to call and in what order via OpenAI
function calling — this is the agentic loop, implemented directly in
`src/agent.py` in under 50 lines, with no framework black box in between.

## Architecture

```
                     ┌─────────────────────┐
   User question ──▶ │   Streamlit Chat UI  │  (app.py)
                     └──────────┬───────────┘
                                ▼
                     ┌─────────────────────┐
                     │   Agent Orchestrator │  (src/agent.py)
                     │  tool-calling loop   │
                     └──────────┬───────────┘
                                │ decides which tool(s) to call
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
     search_documents  calculate_financial   flag_compliance_risk
     (RAG retrieval)        _ratio            (AML rule screen)
              │
              ▼
     ┌─────────────────┐
     │   ChromaDB       │   (src/vectorstore.py)
     │  vector store    │
     └─────────────────┘
              ▲
              │ embeddings (OpenAI text-embedding-3-small)
     ┌─────────────────┐
     │  Ingested docs   │   (data/sample_docs/*.txt → src/ingest.py)
     └─────────────────┘
```

## Tech Stack

- **LLM / function calling**: Groq `llama-3.3-70b-versatile` (free tier, no credit card)
- **Embeddings**: ChromaDB's built-in local model (`all-MiniLM-L6-v2` via onnxruntime) — runs on your machine, no API key, no cost
- **Vector store**: ChromaDB (persistent, local)
- **UI**: Streamlit
- No LangChain/LlamaIndex — the RAG pipeline and agent loop are implemented
  directly so every step is easy to explain in the Q&A round.
- **Cost to run this entire project: $0.** Only the reasoning step calls an
  external API (Groq), and Groq's free tier (no card required) easily
  covers a hackathon demo.

## Project Structure

```
finsight-ai/
├── app.py                  # Streamlit chat UI (deploy this)
├── requirements.txt
├── .env.example
├── data/sample_docs/       # synthetic sample financial documents
├── src/
│   ├── llm_client.py       # OpenAI embeddings + chat wrapper
│   ├── vectorstore.py      # ChromaDB wrapper
│   ├── ingest.py           # chunk + embed + store documents
│   ├── tools.py            # the 3 agent tools + their schemas
│   └── agent.py            # the tool-calling agent loop
└── tests/test_tools.py     # unit tests for deterministic tools
```

## Setup

1. **Clone and install dependencies**
   ```bash
   git clone <your-repo-url>
   cd finsight-ai
   pip install -r requirements.txt
   ```

2. **Add your free Groq API key**
   ```bash
   cp .env.example .env
   # then edit .env and paste your key:
   # GROQ_API_KEY=gsk_...
   ```
   Get a key at https://console.groq.com/keys — sign up with just an email,
   no credit card required. The free tier (30 requests/min, 14,400/day) is
   far more than enough for a hackathon demo.

3. **Ingest the sample documents into the vector store**
   ```bash
   python -m src.ingest
   ```
   You should see `Ingestion complete. Vector store now has N chunks...`

4. **Run the app locally**
   ```bash
   streamlit run app.py
   ```
   Opens at `http://localhost:8501`.

5. **Run the tests** (no API key required)
   ```bash
   python -m pytest tests/
   ```

## Adding your own documents

Drop additional `.txt` files into `data/sample_docs/` and re-run
`python -m src.ingest`. PDF support can be added by extending `src/ingest.py`
with a PDF text extractor (e.g. `pypdf`) before chunking — noted as a clear
next step if asked about extensibility.

## Deploying a live demo link

**Streamlit Community Cloud (recommended, free):**
1. Push this repo to GitHub (see below if you're new to GitHub).
2. Go to https://share.streamlit.io, sign in with GitHub, click "New app."
3. Select this repo, branch `main`, main file `app.py`.
4. Under "Advanced settings → Secrets," add:
   ```
   GROQ_API_KEY = "gsk_..."
   ```
5. Deploy. You'll get a public URL like `https://your-app.streamlit.app` —
   this is your "live working demo link" for the submission.

**Alternative: Hugging Face Spaces** (also free) — create a new Space,
choose "Streamlit" as the SDK, push this repo's contents, and add
`GROQ_API_KEY` under Space Settings → Repository secrets.

## Pushing this project to GitHub (if you're new to GitHub)

1. Create a free account at https://github.com if you haven't already.
2. Click the **+** icon (top right) → **New repository**. Name it
   `finsight-ai`, keep it Public, don't initialize with a README (you
   already have one), then click **Create repository**.
3. On the next page, use the **"uploading an existing file"** link if you'd
   rather drag-and-drop than use git commands — you can upload this whole
   folder's contents directly through the browser.
4. Or, with git installed locally, from inside this folder:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: FinSight AI"
   git branch -M main
   git remote add origin https://github.com/<your-username>/finsight-ai.git
   git push -u origin main
   ```
5. Double-check `.env` is **not** in the repo (it's in `.gitignore`) so your
   API key is never committed.

## Limitations & Next Steps (good Q&A talking points)

- Compliance screening is keyword-based by design for transparency in this
  prototype; a production version would combine this with a learned
  classifier and human-in-the-loop review before any regulatory filing.
- Currently ingests `.txt` only; PDF/scanned-document ingestion (OCR) is the
  natural next extension for real-world filings.
- Single-tenant local vector store; a production deployment would move to a
  managed vector DB with access controls per analyst/team.
