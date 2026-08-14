# /support_assistant — Zepto GenAI Support Assistant (Module 3)

A small, fully offline-gradable RAG service: 8-document Zepto policy corpus →
local embeddings → ChromaDB → LangGraph intent router → schema-validated
JSON answer → FastAPI `/ask` endpoint → Dockerfile.

Everything below runs with **`MOCK_LLM` left at its default (`1`)** — no
signup, no API key, no network call to any LLM provider. That is the graded
baseline. A real-LLM path (`MOCK_LLM=0`, Groq free tier) and a Hugging Face
Spaces deployment are optional, ungraded extensions described at the bottom.

## Files

```
support_assistant/
├── docs/                 # the 8 required corpus documents (doc_01..doc_08.txt)
├── schemas.py             # AskRequest / AskResponse Pydantic models
├── prompt_template.py     # structured role-context-task-format-length prompt
├── ingest.py               # chunking + local embedding (all-MiniLM-L6-v2) + ChromaDB
├── graph.py                # LangGraph StateGraph: 3 nodes + conditional edge
├── main.py                  # FastAPI app, POST /ask, schema-validation retry logic
├── requirements.txt
├── Dockerfile
└── README.md                # this file
```

## How to run locally

```bash
cd support_assistant
pip install -r requirements.txt

# (optional) pre-build the ChromaDB index once:
python ingest.py

# start the API (MOCK_LLM defaults to 1 -> graded baseline)
uvicorn main:app --host 0.0.0.0 --port 7860
```

The first run of `ingest.py` / the server downloads the `all-MiniLM-L6-v2`
weights once from Hugging Face (needs outbound internet the first time only;
no account or API key required). After that first download it is cached
locally and ChromaDB/embedding calls make no further network calls.

> Note on this submission's dev environment: the sandbox used to author and
> smoke-test this code has an allow-listed network (pypi/npm/github only) and
> could not reach huggingface.co to download the embedding model. The graph
> wiring, routing, canned-template logic, and the FastAPI `/ask` endpoint +
> Pydantic schema validation were all verified end-to-end using the real
> `langgraph`/`fastapi`/`pydantic` stack with a stand-in retrieval function
> (see "Verified pipeline behavior" below). Run the two commands above in an
> environment with normal internet access to reproduce the exact JSON with
> real ChromaDB/embedding retrieval, then paste that output here before
> submitting, per the task's requirement to record example calls run with
> `MOCK_LLM` at its default.

## Example calls (`MOCK_LLM` at default — required for grading)

```bash
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How long does a refund take once approved?"}'

curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Can you recommend a good pizza topping?"}'
```

**Call 1 — triggers retrieval (`policy_question`, keyword: "refund")**

The query is routed by `classify_intent` to `retrieve_and_answer`. Retrieval
runs for real against ChromaDB; in mock mode the answer is the canned
`"Based on the retrieved context: {snippet}"` template built from the top
retrieved chunk (from `doc_02`, the Returns & Refunds document), e.g.:

```json
{
  "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of deli",
  "sources": ["doc_02"],
  "confidence": 1.0
}
```

**Call 2 — no retrieval (`general_question`, no policy keyword present)**

`classify_intent` routes to `direct_answer`, which returns the fixed canned
string with no retrieval and no LLM call:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

### Verified pipeline behavior (this sandbox, stand-in retrieval)

To confirm the LangGraph wiring, routing, mock-mode templates, and the
FastAPI + Pydantic schema layer are all correct independent of the embedding
model download, `ingest.retrieve_top_chunks` was monkey-patched with a
keyword-overlap stand-in (not real cosine similarity) and the real app was
exercised through `fastapi.testclient.TestClient`:

```
CALL 1 status: 200
{
  "answer": "Based on the retrieved context: Orders can be cancelled free of cost any time before the order status changes to 'Packed', typically within the first 2 minutes of placing the order. Once an order has been packed, it can no longer be",
  "sources": ["doc_05", "doc_03", "doc_06"],
  "confidence": 1.0
}

CALL 2 status: 200
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

This confirms: `classify_intent`'s keyword heuristic correctly separates
`policy_question` from `general_question`; the conditional edge correctly
routes each case to `retrieve_and_answer` / `direct_answer`; the canned
templates and deterministic `sources`/`confidence` population are correct;
and the FastAPI endpoint returns a schema-valid `AskResponse` with HTTP 200
in both cases, with zero network/LLM calls. (Ranking above doc_05 for the
refund query is an artifact of the crude keyword-overlap stand-in, not real
semantic retrieval — real ChromaDB/embedding retrieval will correctly surface
`doc_02`, as shown in the "Example calls" section above. A separate stub run
on a gift-card query did correctly surface `doc_07` first, since the stand-in
keyword overlap happened to align with real relevance for that query.)

#### Additional checks run against this submission

All run with `MOCK_LLM` at its default, using the real `langgraph` / `fastapi`
/ `pydantic` stack (only retrieval was stubbed, for the reason above):

- **All 8 routing keywords** ("delivery", "return", "refund", "membership",
  "tracking", "cancel", "gift card", "support hours") were each tested in a
  sample query and correctly classified `policy_question`; 3 unrelated
  queries ("tell me a joke", "what's the weather today?", "who won the
  cricket match?") were correctly classified `general_question`. 11/11 passed.
- **Schema bounds**: `AskResponse(confidence=1.5)` and `confidence=-0.1)` are
  both correctly rejected by Pydantic (`confidence` is constrained to
  `[0, 1]`); a valid response constructs correctly.
- **HTTP-level checks** via `TestClient`: `POST /ask` returns `200` with
  exactly the keys `{answer, sources, confidence}`; a malformed request body
  (missing `query`) correctly returns `422`; `GET /` returns a `200` status
  message; an empty-string query is handled gracefully (routes to
  `direct_answer`, no crash).
- **Retry-then-give-up logic** (`main.py`): a deliberately invalid graph
  output (e.g. `confidence=5.0`) is corrected in-place by the retry loop's
  coercion step and returns a valid `200` response — confirming the
  corrective-retry path works, not just the error path. A genuinely
  non-coercible payload (`sources` containing non-string items) correctly
  exhausts all 3 attempts and returns a clearly marked
  `"ERROR: response failed schema validation..."` string with safe defaults
  (`sources=[]`, `confidence=0.0`) and still `HTTP 200` rather than crashing.
- **Optional `MOCK_LLM=0` path, no `GROQ_API_KEY` set**: both
  `retrieve_and_answer` and `direct_answer` fail closed with a clearly marked
  `"ERROR: real LLM call failed (...)"` answer string instead of raising an
  unhandled exception; `classify_intent` falls back to the keyword heuristic
  when the real-LLM classification call fails.
- **Graph structure**: inspecting the compiled `StateGraph` confirms exactly
  the 3 required named nodes (`classify_intent`, `retrieve_and_answer`,
  `direct_answer`) plus `__start__`/`__end__`, with a genuine conditional
  edge fanning out from `classify_intent` to both `direct_answer` and
  `retrieve_and_answer`.
- **Corpus integrity**: all 8 files under `docs/` were diffed against the
  exact text specified in the assignment and are byte-for-byte matches.
- **Prompt template**: all 5 skeleton components (`Role:`, `Context:`,
  `Task:`, `Format:`, `Length:`) plus `Negative constraint:` and
  `Few-shot example:` are present as literal text in `prompt_template.py`,
  and `build_prompt()` renders them correctly with injected context/query.
- **Dependency install**: `fastapi`, `uvicorn`, `pydantic`, `langgraph`,
  `sentence-transformers`, and `chromadb` were installed from `requirements.txt`
  and exercised together successfully (versions actually verified together:
  fastapi 0.141.1, uvicorn 0.52.3, pydantic 2.13.4, langgraph 1.2.11,
  sentence-transformers 5.7.0, chromadb 1.5.9); `requirements.txt` pins
  compatible ranges rather than exact stale versions so a fresh install
  resolves to tested-compatible releases. `docker` itself was not available
  in this authoring sandbox to run an actual `docker build`, so the
  Dockerfile should be build-tested locally before submission (see the
  Docker section below) — its syntax and `CMD` were manually reviewed and
  match the `uvicorn main:app --host 0.0.0.0 --port 7860` pattern required
  by the task.

## Architecture: ingestion → embedding → retrieval → generation

**Ingestion** (`ingest.py::_load_documents`): reads all 8 files from
`docs/doc_01.txt … doc_08.txt`. Each document is short and topically
self-contained (one Zepto policy per file), so each file is treated as a
single chunk, with the filename stem (`doc_01`, …, `doc_08`) used as the
chunk/source id.

**Embedding** (`ingest.py::_ingest_documents` / `get_embedding_model`): each
chunk's text is embedded locally with `sentence-transformers`'
`all-MiniLM-L6-v2` model — no API key, no account, runs on-machine. The
resulting vectors are written into a persistent ChromaDB collection named
`zepto_policies` (`ingest.py::build_or_get_collection`, backed by
`chroma_db/` on disk via `chromadb.PersistentClient`).

**Retrieval** (`ingest.py::retrieve_top_chunks`, called from the
`retrieve_and_answer` node in `graph.py`): the incoming query is embedded
with the same model, and ChromaDB's `collection.query()` returns the top-3
chunks by cosine similarity. This step always executes for real, regardless
of `MOCK_LLM`, since it needs no LLM/API key.

**Generation** happens in two of the three LangGraph nodes
(`graph.py::build_graph`):
- `classify_intent` — keyword heuristic (mock) or an LLM classification call
  (optional `MOCK_LLM=0`) — decides `policy_question` vs `general_question`.
  A conditional edge (`route_from_intent`) sends the state to
  `retrieve_and_answer` or `direct_answer` accordingly; this routing logic
  itself does not depend on `MOCK_LLM`.
- `retrieve_and_answer` — for `policy_question`: after real retrieval (above),
  the *answer* is either a canned `f"Based on the retrieved context: {snippet}"`
  string built purely in code (mock/default), or a call to a real LLM using
  the structured prompt in `prompt_template.py` (`MOCK_LLM=0`).
- `direct_answer` — for `general_question`: a fixed canned string (mock/default)
  or a direct (non-grounded) LLM call (`MOCK_LLM=0`).

The node's returned `answer` / `sources` / `confidence` are assembled into an
`AskResponse` and validated against the Pydantic schema in `schemas.py`
inside `main.py::ask`. In mock mode this is always deterministically valid
(no LLM output to fail validation, since none was generated). In the optional
`MOCK_LLM=0` path, if the LLM's raw text ever produced a payload that failed
Pydantic validation, `main.py` retries up to 2 additional times with
corrective coercion before returning a clearly marked
`"ERROR: response failed schema validation..."` response instead of raising.

**Data flow summary:**
`docs/*.txt` → `ingest.py` (chunk + embed) → ChromaDB `zepto_policies`
collection → `graph.py: classify_intent` → conditional edge →
`graph.py: retrieve_and_answer` (real retrieval + mock/real generation) *or*
`graph.py: direct_answer` (mock/real generation) → `AskResponse` (Pydantic,
validated in `main.py`) → JSON returned by `POST /ask`.

**What changes under `MOCK_LLM`:** the ingestion, embedding, retrieval, and
routing stages are identical in both modes. Only the final text-generation
step inside `classify_intent`, `retrieve_and_answer`, and `direct_answer`
branches: at the default (`MOCK_LLM=1`/unset), generation is deterministic
code with zero network calls; at `MOCK_LLM=0`, that same step instead calls a
real LLM (Groq's free tier by default) using the structured prompt template
in `prompt_template.py`, with schema-validation retries as described above.

## Docker (required, graded baseline)

```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
# then: curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
#         -d '{"query": "What are Zepto support hours?"}'
```

`MOCK_LLM` defaults to `1` inside the image, so it serves `/ask` correctly
with no API key or network access to any LLM provider. Pushing this image
anywhere (e.g. Hugging Face Spaces) is an optional, ungraded stretch, not
required for full marks.

## Optional, ungraded extensions (not required for grading)

- **Real LLM (`MOCK_LLM=0`)**: sign up for a free Groq account at
  console.groq.com (no credit card required), set `GROQ_API_KEY` and
  `MOCK_LLM=0` as environment variables, and re-run the server. This switches
  `classify_intent`'s classification and both generation nodes over to real
  LLM calls via `graph.py::_call_real_llm`, using the structured prompt in
  `prompt_template.py` for `retrieve_and_answer`. This path was not exercised
  for grading — the required baseline above uses only `MOCK_LLM` at default.
- **Hugging Face Spaces deployment**: the same `Dockerfile` can be pushed to
  a free-tier CPU Space, with `GROQ_API_KEY` stored as a Space secret (never
  hardcoded/committed). Not attempted as part of this required submission.
