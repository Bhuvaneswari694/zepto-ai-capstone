"""
LangGraph orchestration for the Zepto support assistant.

State: SupportState (TypedDict)
Nodes (3, as required):
  - classify_intent      -> routes to policy_question / general_question
  - retrieve_and_answer  -> retrieval (always real) + generation (branches on MOCK_LLM)
  - direct_answer        -> generation only (branches on MOCK_LLM)

Every node's *generation* step branches on the MOCK_LLM environment variable:
  - MOCK_LLM unset or "1" (default, GRADED baseline): deterministic, rule-based
    logic only. No LLM call, no network call.
  - MOCK_LLM="0" (OPTIONAL, ungraded extension): calls a real LLM (Groq's
    OpenAI-compatible free-tier API) using the structured prompt template.

The conditional edge out of classify_intent is routing logic and does NOT
depend on MOCK_LLM -- it only depends on the classified intent.
"""

import os
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from ingest import retrieve_top_chunks
from prompt_template import build_prompt

# Keywords used to identify questions related to Zepto policies.
POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


# Defines the shared data passed between all LangGraph nodes.
class SupportState(TypedDict, total=False):
    query: str
    intent: str
    retrieved: List[Dict[str, Any]]
    answer: str
    sources: List[str]
    confidence: float


# Checks whether the required deterministic mock mode is enabled.
def _mock_llm_enabled() -> bool:
    """MOCK_LLM unset or '1' -> mock (graded baseline). '0' -> real LLM."""
    return os.environ.get("MOCK_LLM", "1") != "0"


# Calls the optional real LLM when MOCK_LLM=0 is selected.
def _call_real_llm(prompt: str) -> str:
    """OPTIONAL MOCK_LLM=0 extension. Uses Groq's free-tier, OpenAI-compatible API.
    Requires GROQ_API_KEY. Never called when MOCK_LLM is left at its default."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set; cannot use the optional MOCK_LLM=0 path.")

    from groq import Groq  # imported lazily so it's not a hard dependency in mock mode

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------
# Classifies the query and stores the detected intent in the graph state.
def classify_intent(state: SupportState) -> SupportState:
    query = state["query"]

    if _mock_llm_enabled():
        # Graded baseline: keyword heuristic, no LLM call.
        lowered = query.lower()
        intent = "policy_question" if any(kw in lowered for kw in POLICY_KEYWORDS) else "general_question"
    else:
        # Optional extension: ask the LLM to classify instead.
        prompt = (
            "Classify the following user query as exactly one label: "
            "'policy_question' if it relates to Zepto's delivery, returns, membership, "
            "order tracking, cancellation, gift cards, or support hours, otherwise "
            "'general_question'. Reply with only the single label and nothing else.\n\n"
            f"Query: {query}"
        )
        try:
            label = _call_real_llm(prompt).strip().lower()
            intent = "policy_question" if "policy" in label else "general_question"
        except Exception:
            # Fall back to the heuristic if the real-LLM call fails.
            lowered = query.lower()
            intent = "policy_question" if any(kw in lowered for kw in POLICY_KEYWORDS) else "general_question"

    return {**state, "intent": intent}


# ---------------------------------------------------------------------------
# Conditional routing (does NOT depend on MOCK_LLM)
# ---------------------------------------------------------------------------
# Selects the next graph node based on the classified intent.
def route_from_intent(state: SupportState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


# ---------------------------------------------------------------------------
# Node 2: retrieve_and_answer
# ---------------------------------------------------------------------------
# Retrieves relevant policy chunks and generates a grounded answer.
def retrieve_and_answer(state: SupportState) -> SupportState:
    query = state["query"]

    # Retrieval always runs for real in both modes (no API key needed).
    chunks = retrieve_top_chunks(query, k=3)
    sources = [c["id"] for c in chunks]

    if _mock_llm_enabled():
        # Graded baseline: canned templated answer, no LLM call.
        top_chunk_snippet = chunks[0]["content"][:200] if chunks else ""
        answer = f"Based on the retrieved context: {top_chunk_snippet}"
        confidence = 1.0
    else:
        # Optional extension: prompt the real LLM, grounded only in retrieved chunks.
        context = "\n\n".join(f"[{c['id']}] {c['content']}" for c in chunks)
        prompt = build_prompt(query, context)
        try:
            answer = _call_real_llm(prompt)
            confidence = 0.85
        except Exception as exc:
            answer = f"ERROR: real LLM call failed ({exc})"
            confidence = 0.0

    return {**state, "retrieved": chunks, "answer": answer, "sources": sources, "confidence": confidence}


# ---------------------------------------------------------------------------
# Node 3: direct_answer
# ---------------------------------------------------------------------------
# Gives a direct response when the query is not a policy question.
def direct_answer(state: SupportState) -> SupportState:
    if _mock_llm_enabled():
        # Graded baseline: fixed canned string, no LLM call.
        answer = "I can only answer questions about Zepto policies right now."
        confidence = 1.0
    else:
        prompt = (
            "You are Zepto's support assistant. The user's question is unrelated to "
            "Zepto's policies. Politely explain that you can currently only help with "
            f"Zepto policy questions.\n\nQuestion: {state['query']}"
        )
        try:
            answer = _call_real_llm(prompt)
            confidence = 0.7
        except Exception as exc:
            answer = f"ERROR: real LLM call failed ({exc})"
            confidence = 0.0

    return {**state, "retrieved": [], "answer": answer, "sources": [], "confidence": confidence}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
# Builds the LangGraph workflow by connecting the three required nodes.
def build_graph():
    workflow = StateGraph(SupportState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("retrieve_and_answer", retrieve_and_answer)
    workflow.add_node("direct_answer", direct_answer)

    # Starts the workflow with intent classification.
    workflow.set_entry_point("classify_intent")

    # Routes policy questions to retrieval and other questions to direct answers.
    workflow.add_conditional_edges(
        "classify_intent",
        route_from_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )

    # Ends the graph after either answer node completes.
    workflow.add_edge("retrieve_and_answer", END)
    workflow.add_edge("direct_answer", END)

    return workflow.compile()


# Stores the compiled graph so it can be reused instead of rebuilt each time.
_graph = None


# Returns the compiled graph and creates it only on the first call.
def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph