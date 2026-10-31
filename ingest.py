import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data"

documents = []

# Load all txt files
for file in os.listdir(DATA_PATH):

    if file.endswith(".txt"):

        loader = TextLoader(os.path.join(DATA_PATH, file))

        documents.extend(loader.load())

print(f"Loaded {len(documents)} documents")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

print(f"Created {len(docs)} chunks")

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector database
vectorstore = FAISS.from_documents(
    docs,
    embedding_model
)

# Save vector DB
vectorstore.save_local("vectorstore")

print("FAISS vector database saved successfully!")