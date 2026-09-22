# AccessClass

**An Agentic AI System for Accessible Technical Learning Materials**

AccessClass converts technical lecture PDFs into an accessible, structured, evidence-grounded learning experience. It is not a generic chatbot or a PDF-to-speech tool — it reads a document, audits it for accessibility barriers, understands its academic content, explains diagrams and code in context, produces chapter-based audio, and answers questions *only* from the uploaded lecture material.

Built as a milestone project during an AI internship, using a six-stage multi-agent pipeline orchestrated with **CrewAI**.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [The Six-Stage Pipeline](#the-six-stage-pipeline)
- [Supported Subjects](#supported-subjects)
- [Tech Stack](#tech-stack)
- [Architecture Notes](#architecture-notes)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Testing](#testing)
- [Known Limitations](#known-limitations)
- [Academic Integrity](#academic-integrity)

---

## Overview

Technical lecture material — scanned handouts, image-only PDFs, dense code listings, diagrams, and mathematical notation — is often inaccessible to students who rely on screen readers, audio learning, or simplified structured content. AccessClass addresses this by running every uploaded lecture through a pipeline of specialized AI agents that:

- Extract clean, structured text and page-level metadata from any PDF (typed or scanned)
- Audit the document for concrete accessibility barriers
- Classify content by academic subject and type
- Generate plain-language explanations of visuals, code, tables, and equations
- Build an accessible HTML study pack with chapter-based audio and a glossary
- Answer student questions using only the processed lecture, with page citations

## Key Features

- 📄 **Universal PDF ingestion** — handles normal selectable text and scanned/image-only pages via OCR fallback
- ♿ **Accessibility auditing** — flags scanned pages, missing headings, undescribed images, headerless tables, unlabeled code, and low-confidence OCR
- 🧠 **Automatic subject & content classification** — no manual tagging required; a single document can mix subjects
- 🖼️ **Multimodal explanations** — Gemini vision for diagrams/images, Groq for code/tables/equations
- 🔊 **Chapter-based audio** — generated per heading section via Edge TTS, not one long recording
- 📖 **Accessible HTML study pack** — proper heading structure, inline visual descriptions, glossary
- 💬 **"Ask This Lecture"** — retrieval-grounded Q&A with page citations; explicitly refuses to answer from outside knowledge
- 📝 **Revision queue** — students can flag confusing pages for later review
- 🛡️ **Academic integrity by design** — never completes assignments, writes exam answers, or produces take-home solutions

## The Six-Stage Pipeline

| Stage | Agent | Purpose |
|---|---|---|
| 1 | **Document Reader** | Extracts text, headings, pages, images, tables, and code blocks from the uploaded PDF |
| 2 | **Accessibility Auditor** | Detects barriers: scanned pages, missing structure, unlabeled visuals, low-confidence OCR |
| 3 | **Subject Interpreter** | Classifies each content element by subject and type (code / table / equation / image / text) |
| 4 | **Explanation Agent** | Produces plain-language descriptions of visuals, code, and equations |
| 5 | **Study-Pack Agent** | Builds structured HTML, chapter audio, and a glossary |
| 6 | **Grounded Learning Agent** | Indexes the lecture for retrieval and answers questions with page citations |

Stages 1–2 are deterministic (rule-based extraction and checks); Stages 3–6 involve genuine LLM reasoning and are implemented as CrewAI agents or direct model calls where appropriate.

## Supported Subjects

Programming Fundamentals & C++ · Data Structures & Algorithms · Calculus · ICT · OOP · Database Systems · Operating Systems · Computer Networks · AI/ML

Classification happens automatically per content element — a single lecture can mix multiple subjects (e.g. a DSA lecture with a Calculus refresher).

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI |
| UI | Server-rendered HTML + CSS (no JavaScript) |
| Agent orchestration | CrewAI (Crews & Flows) |
| Text reasoning | Groq — `openai/gpt-oss-120b` |
| Visual reasoning & embeddings | Gemini — `gemini-3.5-flash` / `gemini-embedding-001` |
| PDF extraction | PyMuPDF |
| OCR | Tesseract |
| Text-to-speech | Edge TTS |
| Vector database | Qdrant |
| Validation | Pydantic |
| Storage & database | Supabase (Storage + Postgres) |
| Testing | pytest |

## Architecture Notes

A few deliberate decisions worth knowing before reading the code:

- **Groq via the "openai" provider.** This CrewAI version has no native Groq integration. Groq exposes an OpenAI-compatible endpoint, so it's called through CrewAI's `openai` provider pointed at Groq's base URL, with the exact model string preserved.
- **Deterministic stages aren't LLM agents.** The Document Reader and Accessibility Auditor need no reasoning, so they run as plain Python steps inside the CrewAI Flow rather than LLM-backed agents — faster, cheaper, and no hallucination risk for work that's already fully solved by direct extraction and rule-checking.
- **One Flow, not six.** All six stages share a single `AccessClassFlow`, so later stages can read earlier stages' output from shared state.
- **Rate-limit aware.** Gemini's free tier caps requests per minute, not per account balance. Every Gemini call retries automatically on 429/500/503 with backoff, and image/embedding calls are deliberately spaced out to avoid bursting the limit in the first place.
- **Qdrant resilience.** If the cloud Qdrant cluster is temporarily unreachable, the app falls back to a temporary in-memory vector store so "Ask This Lecture" still works for that session — a completed accessibility report and study pack are never discarded just because retrieval indexing had a network hiccup.
- **Point IDs are UUID5-derived.** Qdrant only accepts integer or UUID point IDs, so lecture chunk IDs are deterministically converted via `uuid.uuid5`.

## Project Structure

```
accessclass/
├── main.py                     # FastAPI app: routes, HTML rendering
├── config.py                   # Every model name & env var name, in one place
├── flow.py                     # CrewAI Flow wiring all six stages together
├── requirements.txt
├── .env.example
│
├── static/
│   ├── index.css                # Shared design system (upload, results, ask, revision pages)
│   └── style.css                # Legacy stylesheet (unused, kept for reference)
├── templates/
│   ├── index.html               # Upload page
│   ├── results.html             # Pipeline results
│   ├── ask.html                 # Ask This Lecture
│   └── revision_queue.html
│
├── models/                      # Pydantic schemas (one file per pipeline stage's output)
├── agents/                      # One file per pipeline stage's logic
├── tools/                       # PDF/OCR/Gemini/TTS/chunking helpers
├── db/                          # Supabase and Qdrant clients + schema.sql
└── tests/                       # 78 automated tests
```

## Getting Started

### Prerequisites

- Python 3.11 or 3.12 (CrewAI does not yet support 3.14)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed and on your PATH (or configured via `pytesseract.pytesseract.tesseract_cmd`)
- Accounts/API keys for: [Groq](https://console.groq.com), [Google AI Studio (Gemini)](https://aistudio.google.com), [Qdrant Cloud](https://cloud.qdrant.io), [Supabase](https://supabase.com)

### Installation

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate       # macOS/Linux

pip install -r requirements.txt
```

### Supabase setup

1. In the Supabase SQL editor, run the contents of `db/schema.sql` (creates the `documents` and `revision_queue` tables).
2. Create a Storage bucket named `lectures`.

### Qdrant setup

Create a free cluster at [cloud.qdrant.io](https://cloud.qdrant.io) and copy its endpoint URL and API key. The `accessclass_lecture_chunks` collection is created automatically on first use.

### Environment variables

```bash
cp .env.example .env
```

Fill in `GROQ_API_KEY`, `GEMINI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `SUPABASE_URL`, and `SUPABASE_KEY`.

### Running the app

```bash
uvicorn main:app
```

Open `http://127.0.0.1:8000`.

> **Note:** avoid `--reload` if your virtual environment sits inside the project folder — the file watcher can pick up unrelated package installs inside `venv/` and trigger unnecessary restarts.

## Usage

1. **Upload** a lecture PDF from the home page (subject selection is optional — classification is automatic).
2. Processing takes roughly 1–3 minutes depending on document length and image count.
3. Review the **Results** page: extraction summary, accessibility report, subject tags, explanations, and the generated study pack (HTML + audio + glossary).
4. Click **Ask This Lecture** to ask questions grounded in that specific document.
5. Use **Mark a page as confusing** to add pages to the revision queue for later review.

## Testing

```bash
python -m pytest tests/ -v
```

78 tests covering every pipeline stage, error-handling paths, retry/fallback logic, and the FastAPI routes. External services (Groq, Gemini, Qdrant, Supabase, Edge TTS) are mocked in tests; a real end-to-end run requires valid API keys.

## Known Limitations

- **OCR pages lose heading structure.** Scanned/image-only pages are recovered via Tesseract, but OCR text carries no font-size metadata, so heading detection (which relies on relative font size) doesn't apply to OCR'd content.
- **Table detection can false-positive on stylized layouts.** PyMuPDF's table detector looks for grid/border patterns and can mistake colorful card-based slide layouts for real tables. The Explanation Agent flags these with `looks_like_real_table: false` where possible, but Stage 1's initial detection isn't perfect.
- **Gemini free-tier rate limits.** Automatic retry with backoff is implemented, but a very image-heavy document may still take longer to process on the free tier.
- **Qdrant in-memory fallback is session-scoped.** If Qdrant Cloud is unreachable and the app falls back to in-memory retrieval, that index is lost when the server restarts — re-upload the document to restore "Ask This Lecture" for it.

## Academic Integrity

AccessClass is an accessibility and learning-support tool, not an assignment-completion tool. It does not write assignments, generate exam answers, or produce take-home solutions. The Grounded Learning Agent explicitly states when a question cannot be answered from the uploaded lecture material rather than inventing a response.
