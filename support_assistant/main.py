"""
FastAPI wrapper around the LangGraph support-assistant pipeline.

POST /ask
  request:  {"query": "<str>"}
  response: {"answer": "<str>", "sources": ["<str>", ...], "confidence": <float 0-1>}

Run locally (MOCK_LLM left at its default -> graded baseline, no API key needed):
    uvicorn main:app --host 0.0.0.0 --port 7860
"""

from fastapi import FastAPI
from pydantic import ValidationError

from graph import get_graph
from schemas import AskRequest, AskResponse

# Creates the FastAPI application with its name and version.
app = FastAPI(title="Zepto Support Assistant", version="1.0.0")


# Provides a simple health check to confirm that the API is running.
@app.get("/")
def root():
    return {"status": "ok", "message": "Zepto Support Assistant is running. POST a query to /ask."}


# Receives a user query, runs the LangGraph pipeline, and returns a validated response.
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:

    # Gets the compiled LangGraph workflow.
    graph = get_graph()

    # Sends the user's query through the support-assistant pipeline.
    result = graph.invoke({"query": request.query})

    # Builds the response data using the values returned by the graph.
    payload = {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "confidence": result.get("confidence", 0.0),
    }

    # Schema-validation retry loop. In mock mode this always succeeds on the
    # first pass since the payload is built deterministically in code. This
    # logic exists for the OPTIONAL MOCK_LLM=0 path, where a raw LLM output
    # could fail validation: retry up to 2 additional times with a corrective
    # coercion before giving up and returning a clearly marked error response.
    attempts = 0
    last_error = None
    max_attempts = 3  # 1 initial + 2 retries

    # Tries to validate and correct the response before returning it.
    while attempts < max_attempts:
        try:
            return AskResponse(**payload)
        except ValidationError as exc:
            last_error = exc
            attempts += 1

            # Corrective coercion before retrying.
            payload["answer"] = str(payload.get("answer", ""))

            try:
                # Converts confidence to a float and keeps it between 0 and 1.
                payload["confidence"] = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
            except (TypeError, ValueError):
                # Uses a safe default when confidence cannot be converted.
                payload["confidence"] = 0.0

            # Replaces an invalid sources value with an empty list.
            if not isinstance(payload.get("sources"), list):
                payload["sources"] = []

    # Returns a clear error response if validation fails after all attempts.
    return AskResponse(
        answer=f"ERROR: response failed schema validation after {attempts} attempts: {last_error}",
        sources=[],
        confidence=0.0,
    )