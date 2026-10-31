import streamlit as st

from transformers import pipeline

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="College FAQ Chatbot",
    page_icon="🎓"
)

st.title("🎓 College FAQ Chatbot")
st.write("Ask questions about admissions, placements, hostel, scholarships, etc.")

# --------------------------------
# LOAD EMBEDDINGS
# --------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------
# LOAD VECTOR DATABASE
# --------------------------------

db = FAISS.load_local(
    "vectorstore",
    embedding_model,
    allow_dangerous_deserialization=True
)

# --------------------------------
# LOAD OPEN-SOURCE LLM
# --------------------------------

@st.cache_resource
def load_llm():

    llm = pipeline(
    "text-generation",
    model="distilgpt2"
)

    return llm

generator = load_llm()

# --------------------------------
# SIDEBAR SETTINGS
# --------------------------------

st.sidebar.header("⚙️ Settings")

top_k = st.sidebar.slider(
    "Number of retrieved documents",
    1,
    5,
    3
)

temperature = st.sidebar.slider(
    "Temperature",
    0.0,
    1.0,
    0.3
)

# --------------------------------
# USER INPUT
# --------------------------------

query = st.text_input("Enter your question:")

# --------------------------------
# CHATBOT LOGIC
# --------------------------------

if query:

    # Retrieve documents
    retrieved_docs = db.similarity_search(
        query,
        k=top_k
    )

    # Combine context
    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    # Prompt
    prompt = f"""
    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    # Generate response
    response = generator(
    prompt,
    max_new_tokens=100,
    temperature=temperature,
    truncation=True
)

    answer = response[0]["generated_text"]

    # Display answer
    st.subheader("🤖 Chatbot Answer")
    st.write(answer)

    # Show retrieved docs
    with st.expander("📄 Retrieved Documents"):

        for i, doc in enumerate(retrieved_docs):

            st.write(f"### Document {i+1}")
            st.write(doc.page_content)