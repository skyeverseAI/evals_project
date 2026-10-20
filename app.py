"""
app.py — a small Streamlit UI over the RAG pipeline.

    uv run streamlit run app.py

Sidebar sets fetch_k (bi-encoder over-retrieve) and top_k (kept after
reranking). Each answer streams in, and the context chunks it was grounded in
can be expanded underneath it.
"""

import time

import streamlit as st

from src.retriever import TOP_K
from src.reranker import RerankingRetriever, FETCH_K
from src.generator import generate_stream

st.set_page_config(page_title="Course RAG", page_icon="🎓", layout="centered")


# load the vector store + cross-encoder once per server, not on every rerun
@st.cache_resource(show_spinner="Loading vector store and reranker...")
def get_retriever():
    return RerankingRetriever()


# 1. SIDEBAR ---- retrieval knobs
with st.sidebar:
    st.header("Retrieval")
    fetch_k = st.slider("fetch_k", 1, 50, FETCH_K, help="Candidates the bi-encoder brings back before reranking")
    top_k = st.slider("top_k", 1, fetch_k, min(TOP_K, fetch_k), help="Chunks kept after reranking and sent to the LLM")
    show_context = st.toggle("Show fetched context", value=True)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []


def render_context(context):
    with st.expander(f"Fetched context ({len(context)} chunks)"):
        for i, (lecture, text) in enumerate(context, 1):
            st.markdown(f"**[{i}] Lecture {lecture}**")
            st.text(text)


st.title("Course RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. HISTORY ---- replay earlier turns
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            st.caption(msg["stats"])
            if show_context:
                render_context(msg["context"])

# 3. NEW TURN ---- retrieve, stream the answer, show the context it used
if query := st.chat_input("Ask about the course..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        retriever = get_retriever()
        retriever.fetch_k, retriever.top_k = fetch_k, top_k

        t0 = time.perf_counter()
        with st.spinner("Retrieving..."):
            docs = retriever.invoke(query)
        t_retrieve = time.perf_counter() - t0

        context = [(doc.metadata.get("lecture", "?"), doc.page_content) for doc in docs]

        # clock time-to-first-token on the first visible chunk
        timing = {}

        def timed_stream():
            for piece in generate_stream(query, [text for _, text in context]):
                timing.setdefault("ttft", time.perf_counter() - t0)
                yield piece

        answer = st.write_stream(timed_stream())
        total = time.perf_counter() - t0
        ttft = timing.get("ttft", total)

        stats = (
            f"fetch_k={fetch_k} · top_k={top_k} · "
            f"retrieve {t_retrieve:.2f}s · TTFT {ttft:.2f}s · total {total:.2f}s"
        )
        st.caption(stats)
        if show_context:
            render_context(context)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "context": context, "stats": stats}
    )
