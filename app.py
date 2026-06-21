"""
app.py

Streamlit chat interface for FinSight AI. This is the file you deploy to
Streamlit Community Cloud or Hugging Face Spaces to get your live demo link.

Run locally with:
    streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.agent import run_agent  # noqa: E402  (import after load_dotenv on purpose)

st.set_page_config(page_title="FinSight AI", page_icon="\U0001F4CA", layout="wide")

st.title("FinSight AI")
st.caption("Agentic RAG assistant for financial document intelligence & compliance screening")

if "history" not in st.session_state:
    st.session_state.history = []
if "display_history" not in st.session_state:
    st.session_state.display_history = []

with st.sidebar:
    st.header("About this system")
    st.write(
        "FinSight retrieves grounded answers from a financial document knowledge "
        "base (RAG) and uses an agentic tool-calling loop to compute financial "
        "ratios and screen text for AML/compliance red flags, with full source "
        "citation and a visible reasoning trace."
    )
    st.subheader("Architecture")
    st.markdown(
        "- **Retriever**: local embeddings (ChromaDB built-in) + ChromaDB\n"
        "- **Agent loop**: manual tool-calling cycle on Groq's Llama 3.3 70B (free tier)\n"
        "- **Tools**: document search, ratio calculator, compliance screener"
    )
    st.subheader("Try asking")
    st.code("What is Company A's current ratio?", language=None)
    st.code("Are there compliance red flags in the suspicious activity report?", language=None)
    st.code("Should Company C get a covenant relaxation?", language=None)

    if st.button("Clear conversation"):
        st.session_state.history = []
        st.session_state.display_history = []
        st.rerun()


def render_trace(trace):
    if not trace:
        return
    with st.expander(f"Agent reasoning trace ({len(trace)} tool call(s))"):
        for step in trace:
            st.markdown(f"**Tool:** `{step['tool']}`")
            st.json(step["args"])
            st.text(step["result"])
            st.divider()


for entry in st.session_state.display_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        render_trace(entry.get("trace"))

query = st.chat_input("Ask FinSight about your financial documents...")

if query:
    st.session_state.display_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving documents and reasoning..."):
            try:
                result = run_agent(query, st.session_state.history)
            except Exception as e:  # surfaces API/key errors clearly in the demo
                result = {"answer": f"Error: {e}", "trace": []}
        st.markdown(result["answer"])
        render_trace(result["trace"])

    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append({"role": "assistant", "content": result["answer"]})
    st.session_state.display_history.append({
        "role": "assistant",
        "content": result["answer"],
        "trace": result["trace"],
    })
