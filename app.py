```python
import os
import json
import faiss
import numpy as np
import streamlit as st

from sentence_transformers import SentenceTransformer
from groq import Groq


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

    /* Main background */
    .stApp {
        background: #f7f8fa;
    }

    /* Main content */
    .main .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
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

    /* Source cards */
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

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid #e5e7eb;
    }

    /* Chat messages */
    div[data-testid="stChatMessage"] {
        border-radius: 14px;
    }

    /* Input */
    div[data-testid="stChatInput"] {
        padding-bottom: 1rem;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid #d1d5db;
        font-weight: 500;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONFIGURATION
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

DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

GROQ_MODEL = "openai/gpt-oss-120b"

DEFAULT_TOP_K = 5


# ============================================================
# LOAD API KEY
# ============================================================

def get_groq_api_key():

    # Streamlit Cloud secrets
    if "GROQ_API_KEY" in st.secrets:
        return st.secrets["GROQ_API_KEY"]

    # Optional local environment fallback
    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        return api_key

    return None


# ============================================================
# LOAD CONFIG
# ============================================================

@st.cache_resource
def load_config():

    if not os.path.exists(CONFIG_PATH):
        return {
            "embedding_model": DEFAULT_EMBEDDING_MODEL
        }

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# LOAD FAISS INDEX
# ============================================================

@st.cache_resource
def load_faiss_index():

    if not os.path.exists(INDEX_PATH):

        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_PATH}"
        )

    return faiss.read_index(
        INDEX_PATH
    )


# ============================================================
# LOAD CHUNKS + METADATA
# ============================================================

@st.cache_data
def load_chunks():

    if not os.path.exists(CHUNKS_PATH):

        raise FileNotFoundError(
            f"chunks.json not found: {CHUNKS_PATH}"
        )

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model(model_name):

    return SentenceTransformer(
        model_name
    )


# ============================================================
# LOAD GROQ CLIENT
# ============================================================

@st.cache_resource
def load_groq_client(api_key):

    return Groq(
        api_key=api_key
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_chunks(
    query,
    index,
    chunks,
    embedding_model,
    top_k=DEFAULT_TOP_K
):

    # Create query embedding
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    # Search FAISS
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        if idx >= len(chunks):
            continue

        result = chunks[idx].copy()

        result["score"] = float(score)

        results.append(result)

    return results


# ============================================================
# BUILD LLM CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    for i, result in enumerate(
        results,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {i}
Document: {result.get("source", "Unknown")}
Path: {result.get("source_path", "Unknown")}
Folder: {result.get("folder", "Unknown")}
Paragraph: {result.get("paragraph", "Unknown")}

CONTENT:
{result.get("text", "")}
"""
        )

    return "\n\n".join(
        context_parts
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

Your job is to answer questions using ONLY the
hospital policy and knowledge-base content provided
in the context.

Rules:

1. Use the retrieved documents as the primary source.
2. Do not invent hospital policies, procedures,
   rules, numbers, names, or requirements.
3. If the answer cannot be found in the provided
   context, clearly say that the information was
   not found in the hospital knowledge base.
4. Give a concise but useful answer.
5. When appropriate, organize procedures using
   numbered steps or bullet points.
6. Do not mention FAISS, embeddings, retrieval,
   chunks, prompts, or internal system details.
7. Do not expose or reproduce the API key.
8. When citing information, mention the document
   name naturally.
"""

    user_prompt = f"""
Hospital Knowledge Base Context:

{context}

User Question:

{question}

Answer the question based strictly on the
provided hospital knowledge base context.
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
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🏥 Hospital Knowledge"
    )

    st.caption(
        "AI-powered policy and knowledge assistant"
    )

    st.divider()

    st.markdown(
        "### Retrieval Settings"
    )

    top_k = st.slider(
        "Relevant documents",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of knowledge-base chunks sent to the LLM."
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

        unique_documents = len(
            set(
                item.get(
                    "source_path",
                    item.get("source", "")
                )
                for item in chunks
            )
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Documents",
                unique_documents
            )

        with col2:

            st.metric(
                "Chunks",
                len(chunks)
            )

        st.caption(
            f"Embedding model: "
            f"{config.get('embedding_model', DEFAULT_EMBEDDING_MODEL)}"
        )

    except Exception:

        st.caption(
            "Knowledge base information unavailable."
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
# INITIALIZE COMPONENTS
# ============================================================

try:

    api_key = get_groq_api_key()

    if not api_key:

        st.error(
            "GROQ_API_KEY is not configured in Streamlit Secrets."
        )

        st.stop()


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


except Exception as e:

    st.error(
        f"Unable to load the knowledge base: {e}"
    )

    st.stop()


# ============================================================
# WELCOME MESSAGE
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div style="
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 20px;
        ">

        <h3 style="margin-top:0;">
        How can I help?
        </h3>

        <p style="color:#6b7280;">
        Ask a question about the hospital's policies,
        procedures, guidelines, or other uploaded documents.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        # Show sources for assistant messages
        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            st.markdown(
                "#### 📚 Sources"
            )

            displayed_sources = set()

            for source in message["sources"]:

                source_path = source.get(
                    "source_path",
                    source.get(
                        "source",
                        "Unknown"
                    )
                )

                if source_path in displayed_sources:
                    continue

                displayed_sources.add(
                    source_path
                )

                source_name = source.get(
                    "source",
                    "Unknown document"
                )

                folder = source.get(
                    "folder",
                    ""
                )

                score = source.get(
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
                    {folder if folder else "Knowledge Base"}
                    &nbsp; • &nbsp;
                    Relevance: {score:.3f}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask about a hospital policy or procedure..."
)


if question:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append({

        "role": "user",

        "content": question

    })


    with st.chat_message("user"):

        st.markdown(
            question
        )


    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching the knowledge base..."
        ):

            try:

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

                    answer = generate_answer(

                        question=question,

                        results=results,

                        groq_client=groq_client

                    )


                st.markdown(
                    answer
                )


                # ------------------------------------------------
                # SOURCES
                # ------------------------------------------------

                if results:

                    st.markdown(
                        "#### 📚 Sources"
                    )

                    displayed_sources = set()

                    for source in results:

                        source_path = source.get(
                            "source_path",
                            source.get(
                                "source",
                                "Unknown"
                            )
                        )

                        if source_path in displayed_sources:
                            continue

                        displayed_sources.add(
                            source_path
                        )

                        source_name = source.get(
                            "source",
                            "Unknown document"
                        )

                        folder = source.get(
                            "folder",
                            ""
                        )

                        score = source.get(
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
                            {folder if folder else "Knowledge Base"}
                            &nbsp; • &nbsp;
                            Relevance: {score:.3f}
                            </div>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                # Save complete assistant response
                st.session_state.messages.append({

                    "role": "assistant",

                    "content": answer,

                    "sources": results

                })


            except Exception as e:

                error_message = (
                    "I encountered an error while "
                    "processing your question."
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append({

                    "role": "assistant",

                    "content": error_message,

                    "sources": []

                })
```

### `requirements.txt`

```text
streamlit
groq
faiss-cpu
sentence-transformers
numpy
```

### Streamlit Secrets

In your Streamlit deployment, add the secret as:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

The key is accessed internally through `st.secrets["GROQ_API_KEY"]`; it is **not placed in the chat interface, sidebar, prompts, or displayed to the user**.

Your deployed/project structure should be:

```text
hospital-rag/
│
├── app.py
├── requirements.txt
│
└── faiss_index/
    ├── index.faiss
    ├── chunks.json
    └── config.json
```

One important point: the app expects the `faiss_index` folder to be **in the same project directory as `app.py`**. The `index.faiss` vector positions correspond directly to the entries in `chunks.json`, so both files must come from the same ingestion run.

The retrieval flow is:

**Question → embedding → FAISS top-K → `chunks.json` text/metadata → GPT-OSS 120B via Groq → answer + source documents**.
