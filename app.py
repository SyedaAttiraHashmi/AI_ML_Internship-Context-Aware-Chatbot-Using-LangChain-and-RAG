import streamlit as st
from main import load_documents, split_documents, get_vectorstore, get_llm, get_chain, DATA_PATH

st.set_page_config(page_title="University Helpdesk", layout="wide", initial_sidebar_state="expanded")
st.title("🎓 University Helpdesk Chatbot")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Settings")
show_context = st.sidebar.checkbox("Show PDF context", value=False)
st.sidebar.markdown("---")


# ---------------- LOAD PDFs ----------------
@st.cache_resource
def init_vectorstore():
    docs = load_documents(DATA_PATH)
    chunks = split_documents(docs)
    return get_vectorstore(chunks)

vector_store = init_vectorstore()
if vector_store is None:
    st.error("❌ No PDF files found in the `data/` folder.")
    st.stop()

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ---------------- LLM & CHAIN ----------------
llm = get_llm()
chain = get_chain(llm)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- CHAT DISPLAY ----------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
            if show_context and "context" in msg:
                st.markdown(f"**Context from PDFs:**\n\n{msg['context'][:1000]}...")

# ---------------- CHAT INPUT ----------------
user_input = st.chat_input("Ask a question ...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            docs = retriever.invoke(user_input)
            context = "\n\n".join([d.page_content for d in docs])

            response = chain.invoke(
                {"input": f"Context:\n{context}\n\nQuestion:\n{user_input}"},
                config={"configurable": {"session_id": "user"}}
            )

            answer = response.content
            st.markdown(answer)

    # Save assistant message with context
    st.session_state.messages.append({"role": "assistant", "content": answer, "context": context})
