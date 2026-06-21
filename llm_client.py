"""
llm_client.py

Thin wrapper around Groq's chat completion API (free tier, no credit card
required) for the agentic reasoning step. Groq's API is OpenAI-compatible,
so this uses the official `openai` Python package pointed at Groq's
endpoint rather than a separate SDK.

Embeddings are handled separately in vectorstore.py using ChromaDB's
built-in local embedding model, so no API key is needed for retrieval at
all -- only this chat/reasoning step needs a (free) Groq key.

Get a free key at https://console.groq.com/keys (no credit card required).
"""

import os
from openai import OpenAI

# llama-3.3-70b-versatile supports OpenAI-style function/tool calling on
# Groq's free tier. Check https://console.groq.com/docs/models if this
# model name has been deprecated by the time you read this.
CHAT_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

_client = None


def get_client() -> OpenAI:
    """
    Lazily creates the client on first real use (not at import time). This
    keeps pure-logic modules (like the ratio calculator and compliance
    screener) importable and testable without requiring an API key.
    """
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Copy .env.example to .env and add your "
                "free Groq key (console.groq.com/keys), or export it in your "
                "shell before running."
            )
        _client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    return _client


def chat_completion(messages: list, tools: list = None, tool_choice: str = "auto"):
    """
    Single chat completion call. Returns the raw response object so the
    caller can inspect both `message.content` and `message.tool_calls`.
    """
    kwargs = {"model": CHAT_MODEL, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    return get_client().chat.completions.create(**kwargs)
