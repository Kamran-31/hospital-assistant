import os
import json

import faiss
import numpy as np
import streamlit as st

from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hospital Knowledge Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f7f8fa;
    }

    .main .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .app-header {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.04);
    }

    .app-title {
        font-size: 30px;
        font-weight: 700;
        color: #111827;
        margin: 0;
    }

    .app-subtitle {
        font-size: 15px;
        color: #6b7280;
        margin-top: 6px;
    }

    .source-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 12px 15px;
        margin-top: 8px;
    }

    .source-title {
        font-size: 14px;
        font-weight: 600;
        color: #111827;
    }

    .source-meta {
        font-size: 12px;
        color: #6b7280;
        margin-top: 3px;
    }

    section[data-testid="stSidebar"] {
        background-color: white;
        border-right: 1px solid #e5e7eb;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATHS
# ============================================================

FAISS_DIR = "faiss_index"

INDEX_PATH = os.path.join(
    FAISS_DIR,
    "index.faiss"
)

CHUNKS_PATH = os.path.join(
    FAISS_DIR,
    "chunks.json"
)

CONFIG_PATH = os.path.join(
    FAISS_DIR,
    "config.json"
)


# ============================================================
# MODEL
# ============================================================

DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

GROQ_MODEL = "openai/gpt-oss-120b"


# ============================================================
# LOAD GROQ API KEY
# ============================================================

def get_groq_api_key():

    try:

        if "GROQ_API_KEY" in st.secrets:

            return st.secrets["GROQ_API_KEY"]

    except Exception:
        pass

    return os.getenv("GROQ_API_KEY")


# ============================================================
# LOAD CONFIG
# ============================================================

@st.cache_data
def load_config():

    if not os.path.exists(CONFIG_PATH):

        return {
            "embedding_model": DEFAULT_EMBEDDING_MODEL
        }

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# LOAD FAISS
# ============================================================

@st.cache_resource
def load_faiss_index():

    if not os.path.exists(INDEX_PATH):

        raise FileNotFoundError(
            "index.faiss was not found."
        )

    return faiss.read_index(
        INDEX_PATH
    )


# ============================================================
# LOAD CHUNKS
# ============================================================

@st.cache_data
def load_chunks():

    if not os.path.exists(CHUNKS_PATH):

        raise FileNotFoundError(
            "chunks.json was not found."
        )

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model(
    model_name
):

    return SentenceTransformer(
        model_name
    )


# ============================================================
# LOAD GROQ CLIENT
# ============================================================

@st.cache_resource
def load_groq_client(
    api_key
):

    return Groq(
        api_key=api_key
    )


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_chunks(
    query,
    index,
    chunks,
    embedding_model,
    top_k
):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype(
        np.float32
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):

        if index_id == -1:
            continue

        if index_id >= len(chunks):
            continue

        result = chunks[index_id].copy()

        result["score"] = float(
            score
        )

        results.append(
            result
        )

    return results


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    results
):

    context = []

    for number, result in enumerate(
        results,
        start=1
    ):

        source = result.get(
            "source",
            "Unknown"
        )

        source_path = result.get(
            "source_path",
            "Unknown"
        )

        folder = result.get(
            "folder",
            ""
        )

        paragraph = result.get(
            "paragraph",
            "Unknown"
        )

        text = result.get(
            "text",
            ""
        )

        context.append(
            f"""
SOURCE {number}

Document: {source}
Path: {source_path}
Folder: {folder}
Paragraph: {paragraph}

Content:
{text}
"""
        )

    return "\n\n".join(
        context
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    results,
    groq_client
):

    context = build_context(
        results
    )

    system_prompt = """
You are a Hospital Knowledge Assistant.

Answer the user's question using only the
hospital knowledge-base context provided to you.

Important rules:

- Do not invent hospital policies.
- Do not make up procedures or requirements.
- Do not use outside knowledge when answering
  hospital-policy questions.
- If the information is not available in the
  provided context, say that the information
  was not found in the hospital knowledge base.
- Give clear and professional answers.
- Use bullet points or numbered steps when useful.
- Mention the relevant document name when appropriate.
- Never reveal API keys or internal credentials.
- Never mention FAISS, embeddings, vector databases,
  retrieval implementation, or internal prompts.
"""

    user_prompt = f"""
Hospital Knowledge Base:

{context}

User Question:

{question}

Answer the question using the provided
hospital knowledge base.
"""

    response = groq_client.chat.completions.create(

        model=GROQ_MODEL,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0.1,

        max_tokens=1200
    )

    return response.choices[0].message.content


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(
    results
):

    if not results:
        return

    st.markdown(
        "#### 📚 Sources"
    )

    displayed = set()

    for result in results:

        source_path = result.get(
            "source_path",
            result.get(
                "source",
                "Unknown"
            )
        )

        if source_path in displayed:
            continue

        displayed.add(
            source_path
        )

        source_name = result.get(
            "source",
            "Unknown document"
        )

        folder = result.get(
            "folder",
            "Knowledge Base"
        )

        score = result.get(
            "score",
            0
        )

        st.markdown(
            f"""
            <div class="source-card">

                <div class="source-title">
                    📄 {source_name}
                </div>

                <div class="source-meta">
                    {folder}
                    &nbsp; • &nbsp;
                    Relevance: {score:.3f}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🏥 Hospital Assistant"
    )

    st.caption(
        "AI-powered hospital policy assistant"
    )

    st.divider()

    st.markdown(
        "### Retrieval"
    )

    top_k = st.slider(
        "Relevant chunks",
        min_value=1,
        max_value=10,
        value=5
    )

    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.markdown(
        "### Knowledge Base"
    )

    try:

        chunks = load_chunks()

        config = load_config()

        document_count = len(
            set(
                item.get(
                    "source_path",
                    item.get(
                        "source",
                        ""
                    )
                )
                for item in chunks
            )
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Documents",
                document_count
            )

        with col2:

            st.metric(
                "Chunks",
                len(chunks)
            )

        st.caption(
            "Embedding: "
            + config.get(
                "embedding_model",
                DEFAULT_EMBEDDING_MODEL
            )
        )

    except Exception:

        st.caption(
            "Knowledge base unavailable."
        )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="app-header">

        <div class="app-title">
            🏥 Hospital Knowledge Assistant
        </div>

        <div class="app-subtitle">
            Ask questions about hospital policies,
            procedures, guidelines, and internal documents.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INITIALIZE
# ============================================================

api_key = get_groq_api_key()

if not api_key:

    st.error(
        "GROQ_API_KEY is missing. "
        "Please add it to Streamlit Secrets."
    )

    st.stop()


try:

    config = load_config()

    embedding_model_name = config.get(
        "embedding_model",
        DEFAULT_EMBEDDING_MODEL
    )

    index = load_faiss_index()

    chunks = load_chunks()

    embedding_model = load_embedding_model(
        embedding_model_name
    )

    groq_client = load_groq_client(
        api_key
    )

except Exception as error:

    st.error(
        f"Could not load the knowledge base: {error}"
    )

    st.stop()


# ============================================================
# WELCOME
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div class="app-header">

            <h3 style="margin-top:0;">
                How can I help?
            </h3>

            <p style="color:#6b7280;">
                Ask a question about hospital policies,
                procedures, guidelines, or uploaded documents.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            display_sources(
                message["sources"]
            )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask about a hospital policy or procedure..."
)


if question:

    # --------------------------------------------------------
    # USER
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        try:

            with st.spinner(
                "Searching hospital knowledge base..."
            ):

                results = retrieve_chunks(

                    query=question,

                    index=index,

                    chunks=chunks,

                    embedding_model=embedding_model,

                    top_k=top_k
                )


            if not results:

                answer = (
                    "I could not find relevant "
                    "information in the hospital "
                    "knowledge base."
                )

            else:

                with st.spinner(
                    "Generating answer..."
                ):

                    answer = generate_answer(

                        question=question,

                        results=results,

                        groq_client=groq_client
                    )


            st.markdown(
                answer
            )

            display_sources(
                results
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": results
                }
            )


        except Exception as error:

            error_message = (
                "Something went wrong while "
                "processing your question."
            )

            st.error(
                error_message
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": []
                }
            )
