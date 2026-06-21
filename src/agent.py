"""
agent.py

The agentic core of the project: a manual ReAct-style tool-calling loop built
directly on the OpenAI chat completions API (no LangChain agent abstraction).
Written this way on purpose -- it makes the entire reasoning loop visible in
~40 lines, which matters both for debugging and for explaining the system
live to judges.

Flow per user turn:
  1. Send the conversation + tool schemas to the model.
  2. If the model requests tool call(s), execute them locally and append the
     results back into the conversation.
  3. Repeat until the model returns a plain text answer (or MAX_TURNS hits).
"""

import json

from src.llm_client import chat_completion
from src.tools import TOOL_SCHEMAS, AVAILABLE_FUNCTIONS

SYSTEM_PROMPT = """You are FinSight, an agentic assistant for financial document \
intelligence and compliance screening, used by credit and compliance analysts.

You have three tools: search_documents (semantic search over a financial document \
knowledge base), calculate_financial_ratio (computes ratios like current_ratio or \
debt_to_equity), and flag_compliance_risk (screens text for AML red flags).

Rules:
- Always call search_documents before making any factual claim about a company, \
loan, or report. Never rely on outside knowledge for document-specific facts.
- When computing a ratio, pull the numerator and denominator from retrieved content, \
not from assumption.
- Always cite the source filename(s) for facts you state, e.g. "(Source: \
company_a_financial_statement.txt)".
- If the knowledge base does not contain the answer, say so plainly instead of \
guessing.
"""

MAX_TURNS = 5


def run_agent(user_query: str, history: list = None) -> dict:
    """
    Runs the agent loop for one user query.
    `history` is a list of prior {"role": ..., "content": ...} messages (excluding
    the system prompt) used to keep multi-turn context.
    Returns {"answer": str, "trace": list} where trace records every tool call
    made, so the UI can show full reasoning transparency.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_query})

    trace = []

    for _ in range(MAX_TURNS):
        response = chat_completion(messages, tools=TOOL_SCHEMAS)
        message = response.choices[0].message

        if message.tool_calls:
            messages.append(message)
            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}
                fn = AVAILABLE_FUNCTIONS.get(fn_name)
                result = fn(**fn_args) if fn else f"Unknown tool requested: {fn_name}"
                trace.append({"tool": fn_name, "args": fn_args, "result": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })
            continue

        return {"answer": message.content, "trace": trace}

    return {
        "answer": "I reached my reasoning step limit without a final answer. Try a more specific question.",
        "trace": trace,
    }
