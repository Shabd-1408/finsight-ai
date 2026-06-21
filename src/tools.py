"""
tools.py

Defines the three tools available to the agent, plus their JSON schemas for
OpenAI function calling. Each tool is a plain, transparent Python function on
purpose: in a compliance/finance context, every step the agent takes should
be auditable rather than hidden inside a black-box library call.
"""

from src.vectorstore import VectorStore

_store = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def search_documents(query: str, n_results: int = 4) -> str:
    """RAG retrieval tool: semantic search over the ingested document knowledge base."""
    store = get_store()
    results = store.query(query, n_results=n_results)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    if not docs:
        return "No relevant documents found in the knowledge base."
    formatted = []
    for doc, meta in zip(docs, metas):
        formatted.append(f"[Source: {meta.get('source')}, chunk {meta.get('chunk_index')}]\n{doc}")
    return "\n\n---\n\n".join(formatted)


_RATIO_INTERPRETATIONS = {
    "current_ratio": "Above 1.0 generally indicates the company can cover short-term liabilities with short-term assets.",
    "quick_ratio": "Above 1.0 indicates strong short-term liquidity without relying on selling inventory.",
    "debt_to_equity": "Above 2.0 is often considered high leverage, though acceptable levels vary by industry.",
}


def calculate_financial_ratio(ratio_name: str, numerator: float, denominator: float) -> str:
    """Calculator tool: computes a named financial ratio with a short interpretation."""
    if denominator == 0:
        return "Error: denominator cannot be zero."
    value = numerator / denominator
    note = _RATIO_INTERPRETATIONS.get(ratio_name, "")
    return f"{ratio_name} = {numerator} / {denominator} = {value:.2f}. {note}".strip()


# Intentionally simple, rule-based AML/compliance red-flag screening.
# Kept as explicit keyword rules (not a model call) so it is fully auditable,
# explainable to a judge in 8 minutes of Q&A, and trivially extensible.
RED_FLAG_PATTERNS = {
    "structuring": "Multiple transactions kept just below standard reporting thresholds.",
    "shell company": "References to a shell entity with no clear business footprint.",
    "round-dollar": "Suspiciously round transaction amounts inconsistent with normal activity.",
    "rapid movement": "Funds moved quickly through accounts shortly after being received.",
    "high-risk jurisdiction": "Transactions tied to a jurisdiction flagged for weak AML enforcement.",
    "cash-intensive": "Unusually high cash activity inconsistent with the stated business type.",
}


def flag_compliance_risk(text: str) -> str:
    """Compliance screening tool: checks text against a red-flag keyword ruleset."""
    text_lower = text.lower()
    flags_found = [
        f"- {keyword.upper()}: {explanation}"
        for keyword, explanation in RED_FLAG_PATTERNS.items()
        if keyword in text_lower
    ]
    if not flags_found:
        return (
            "No predefined compliance red-flag keywords detected in this text. "
            "(Keyword-based screen only -- not a substitute for full AML review.)"
        )
    return "Potential compliance red flags detected:\n" + "\n".join(flags_found)


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the financial document knowledge base (financial statements, "
                "loan agreements, suspicious activity reports) for content relevant to "
                "a query. Call this first for any question about specific facts or figures."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_financial_ratio",
            "description": (
                "Calculate a financial ratio (e.g. current_ratio, quick_ratio, "
                "debt_to_equity) given a numerator and denominator pulled from "
                "retrieved document content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ratio_name": {"type": "string"},
                    "numerator": {"type": "number"},
                    "denominator": {"type": "number"},
                },
                "required": ["ratio_name", "numerator", "denominator"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_compliance_risk",
            "description": (
                "Screen a passage of text for common AML/compliance red-flag patterns "
                "such as structuring, shell companies, or rapid fund movement."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to screen, typically retrieved document content"},
                },
                "required": ["text"],
            },
        },
    },
]

AVAILABLE_FUNCTIONS = {
    "search_documents": search_documents,
    "calculate_financial_ratio": calculate_financial_ratio,
    "flag_compliance_risk": flag_compliance_risk,
}
