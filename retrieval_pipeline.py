from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

persistant_directory="vector_store"

embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db=Chroma(
    persist_directory=persistant_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
    collection_name="company_docs"
)

query="tell me about google?"

retriever=db.as_retriever(search_kwargs={"k":5})

relevant_docs=retriever.invoke(query)

print(f"user query: {query}\n")

print("---Context---")
for i,doc in enumerate(relevant_docs,1):
    print(f"Document {i}:\n{doc.page_content}\n ")

#llm setup
hf_llm=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.2,
    max_new_tokens=300
)
chat_model = ChatHuggingFace(llm=hf_llm)

messages=[
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content=f"Use the following context to answer the question:\n\n{relevant_docs}\n\nQuestion:{query}.Please provide a clear,helpful answer.If you cant find the answer say,i coudn't find the answer from the context." )
]

result=chat_model.invoke(messages)
print("\n---Generated response---")
print("content only:")
print(result.content)

