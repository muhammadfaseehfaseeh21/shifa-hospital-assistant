"""
Hospital Knowledge Base Assistant
----------------------------------
Streamlit chat app that:
1. Loads a pre-built FAISS index (created by ingest.py)
2. Embeds the user's question and retrieves the most relevant chunks
3. Sends those chunks as context to a Groq-hosted LLM
4. Displays the answer along with which document(s) it came from

The Groq API key is read from Streamlit secrets (st.secrets), never typed
into the UI or shown in a text box.
"""

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"  # Groq-hosted model
TOP_K = 4  # number of chunks to retrieve per question

st.set_page_config(page_title="Hospital Knowledge Base Assistant", page_icon="🏥")


# ---------------------------------------------------------------------------
# Cached resources (loaded once per session, not on every rerun)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_vectorstore(_embeddings):
    return FAISS.load_local(
        INDEX_DIR,
        _embeddings,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource(show_spinner=False)
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error(
            "GROQ_API_KEY not found in secrets. "
            "Add it to `.streamlit/secrets.toml` or your app's Secrets settings."
        )
        st.stop()
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Retrieval + generation
# ---------------------------------------------------------------------------
def retrieve_chunks(vectorstore, question, k=TOP_K):
    """Return the top-k most relevant document chunks for the question."""
    return vectorstore.similarity_search(question, k=k)


def build_context(chunks):
    """Format retrieved chunks into a single context string for the LLM prompt."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source_file", "unknown")
        category = chunk.metadata.get("category", "unknown")
        parts.append(f"[Source {i} | {category} / {source}]\n{chunk.page_content}")
    return "\n\n---\n\n".join(parts)


def generate_answer(client, question, context):
    """Call the Groq LLM with the retrieved context and return the answer text."""
    system_prompt = (
        "You are a helpful hospital knowledge base assistant. "
        "Answer the user's question using ONLY the context provided below. "
        "If the context does not contain the answer, say you don't have that "
        "information in the knowledge base. Be concise and clear."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def unique_sources(chunks):
    """Return a de-duplicated list of (category, source_file) pairs, in order of relevance."""
    seen = []
    for chunk in chunks:
        pair = (chunk.metadata.get("category", "unknown"), chunk.metadata.get("source_file", "unknown"))
        if pair not in seen:
            seen.append(pair)
    return seen


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🏥 Hospital Knowledge Base Assistant")
st.caption("Ask a question about hospital policies, departments, admission, or safety.")

embeddings = load_embeddings()
vectorstore = load_vectorstore(embeddings)
client = get_groq_client()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Sources"):
                for category, source_file in msg["sources"]:
                    st.markdown(f"- **{category}** — {source_file}")

# Chat input
question = st.chat_input("Ask a question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            chunks = retrieve_chunks(vectorstore, question)
            context = build_context(chunks)

        with st.spinner("Generating answer..."):
            answer = generate_answer(client, question, context)

        st.markdown(answer)

        sources = unique_sources(chunks)
        if sources:
            with st.expander("📄 Sources"):
                for category, source_file in sources:
                    st.markdown(f"- **{category}** — {source_file}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
