Oncology RAG Assistant

A Retrieval-Augmented Generation (RAG) based chatbot that answers oncology-related questions using guideline PDFs.  
This system provides informational support only and does not provide medical advice, diagnosis, or treatment.

  Tech Stack
- Python
- LangChain
- ChromaDB
- HuggingFace (Mistral-7B)
- Streamlit

steps to run:
1)python -m venv venv
2)pip install -r requirements.txt
3)Set Environment Variable
4)python ingestion_pipeline.py
5)streamlit run history_aware_generation.py

