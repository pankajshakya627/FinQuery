# Minimal RAG System

Working minimal RAG implementation that supports:
✅ Ollama (local LLMs)
✅ Azure OpenAI
✅ Standard OpenAI
✅ Document-aware chunking with LangChain RecursiveCharacterTextSplitter
✅ Modern LangChain packages (no deprecation warnings)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure in `.env`:
   - For Ollama: `LLM_PROVIDER=ollama` and `LLM_MODEL=llama3.2`
     - First install Ollama: https://ollama.ai/download
     - Then run: `ollama pull llama3.2`
   - For Azure OpenAI: set `LLM_PROVIDER=azure` and add your Azure credentials
   - For OpenAI: `LLM_PROVIDER=openai` and set `OPENAI_API_KEY`

## Run Example Script
```bash
cd minimal-rag
PYTHONPATH=/Users/pankajshakya/Downloads/hdfc-rag/minimal-rag python example.py
```

## Run Streamlit Web UI
```bash
cd minimal-rag
PYTHONPATH=/Users/pankajshakya/Downloads/hdfc-rag/minimal-rag streamlit run app.py
```

This will open a web interface where you can:
- Upload PDF documents
- Ask questions about uploaded documents
- See current configuration status

## Key Technologies

- **LangChain**: RAG orchestration
- **langchain-huggingface**: Modern embeddings (HuggingFaceEmbeddings)
- **langchain-ollama**: Modern Ollama integration (ChatOllama)
- **langchain-chroma**: Vector storage
- **RecursiveCharacterTextSplitter**: Document-aware chunking
- **Streamlit**: Web interface
- **PyPDFLoader/TextLoader**: Document ingestion

## Project Structure
```
minimal-rag/
├── rag_module/
│   ├── __init__.py
│   ├── config.py
│   └── rag.py       # Core RAG with semantic chunking
├── app.py           # Streamlit web UI
├── example.py
├── requirements.txt
├── .env
└── README.md
```

## Usage
```python
from app.rag import MinimalRAG

rag = MinimalRAG()
rag.add_document("Your document text here")
answer = rag.query("Your question here")
```
