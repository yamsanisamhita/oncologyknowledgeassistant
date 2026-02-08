import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load env
load_dotenv()

st.set_page_config(page_title="Oncology RAG Assistant", page_icon="🧬")

st.title("🧬 Oncology Knowledge Assistant")
st.caption(
    "This tool provides informational answers from oncology guideline PDFs. "
    "It does NOT provide medical advice, diagnosis, or treatment."
)

# Load vector DB
persistant_directory = "vector_store"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(
    persist_directory=persistant_directory,
    embedding_function=embedding_model
)

# Load LLM
hf_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.2,
    max_new_tokens=300
)
chat_model = ChatHuggingFace(llm=hf_llm)

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# User input
user_question = st.chat_input("Ask an oncology-related question...")

if user_question:
    st.chat_message("user").write(user_question)

    # Question rewriting (optional)
    if st.session_state.chat_history:
        rewrite_prompt = [
            SystemMessage(content="Rewrite the question to be standalone and searchable."),
        ] + st.session_state.chat_history + [
            HumanMessage(content=user_question)
        ]
        rewritten = chat_model.invoke(rewrite_prompt).content.strip()
        search_question = rewritten
    else:
        search_question = user_question

    # Retrieve documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    # Build oncology-safe prompt
    context = "\n\n".join(
        [f"[Source: {doc.metadata['source']}]\n{doc.page_content}" for doc in docs]
    )

    final_prompt = f"""
You are an oncology knowledge assistant.

Answer the question ONLY using the information from the documents below.
Do NOT provide medical advice or diagnosis.
Stay factual and guideline-based.

Question: {user_question}

Documents:
{context}

If the answer is not present, say:
"I don't have enough information from the provided documents."
"""

    messages = [
        SystemMessage(content="You answer strictly from provided documents."),
        *st.session_state.chat_history,
        HumanMessage(content=final_prompt)
    ]

    response = chat_model.invoke(messages)
    answer = response.content

    st.chat_message("assistant").write(answer)

    # Save history
    st.session_state.chat_history.append(HumanMessage(content=user_question))
    st.session_state.chat_history.append(AIMessage(content=answer))

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        """
        - **Domain:** Oncology (Guideline-based)
        - **Technique:** Retrieval-Augmented Generation (RAG)
        - **Data:** PDF Guidelines
        - **Safety:** No diagnosis or treatment advice
        """
    )
    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()
