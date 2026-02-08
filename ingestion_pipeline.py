import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    print(f"Loading PDF documents from: {docs_path}")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory does not exist: {docs_path}")
    
    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError("No PDF files found")

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)}")
        print(f"Preview: {doc.page_content[:200]}...")

    return documents


def split_documents(documents, chunk_size=800, chunk_overlap=150):
    print("Splitting documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    return chunks


def create_vector_store(chunks, persist_directory="vector_store"):
    print("Creating embeddings and storing in Chroma DB")

    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name="oncology_docs"
    )

    print("Vector store created successfully")
    return vector_store


def main():
    documents = load_documents("docs")
    chunks = split_documents(documents)
    create_vector_store(chunks)


if __name__ == "__main__":
    main()
