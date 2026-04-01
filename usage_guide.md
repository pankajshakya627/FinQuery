# HDFC RAG System — Usage Guide

## 1. Start the System

Every time you want to use the app, open a terminal and run:

```bash
cd /Users/pankajshakya/Downloads/hdfc-rag

# Activate the virtual environment
source venv/bin/activate

# Start the server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

> MongoDB starts automatically on login (it was registered as a brew service). If it ever stops:
> ```bash
> brew services start mongodb/brew/mongodb-community@7.0
> ```

Wait until you see:
```
INFO:     Application startup complete.
```

---

## 2. Open the Web UI

| Page | URL | Purpose |
|---|---|---|
| 💬 Chat / Q&A | http://localhost:8000 | Ask questions about indexed documents |
| 📤 Upload | http://localhost:8000/upload | Upload and index new documents |
| 📊 Dashboard | http://localhost:8000/dashboard | View stats, query logs, document list |
| 📖 API Docs | http://localhost:8000/docs | Interactive Swagger UI for all endpoints |

---

## 3. Upload a Document

### Via the Web UI
1. Go to **http://localhost:8000/upload**
2. Click **Choose File** and pick any supported document
3. Enter a **Title** (e.g. `HDFC MITC 2024`)
4. Optionally add tags (comma-separated) and a description
5. Click **Upload & Index**

### Supported File Types
| Format | Extension |
|---|---|
| PDF | `.pdf` |
| Word | `.docx`, `.doc` |
| Excel | `.xlsx`, `.xls` |
| PowerPoint | `.pptx` |
| CSV | `.csv` |
| JSON | `.json` |
| HTML | `.html` |
| Markdown | `.md` |
| Plain Text | `.txt` |

### Via cURL (API)
```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@/path/to/your/document.pdf" \
  -F "title=My Document" \
  -F "tags=finance,credit-card" \
  -F "chunk_size=512"
```

---

## 4. Ask Questions (RAG Query)

### Via the Web UI
1. Go to **http://localhost:8000**
2. Type your question in the chat box
3. Press **Enter** or click **Send**
4. The system will retrieve relevant chunks and generate an answer with source citations

### Via cURL (API)
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the annual fee for the HDFC Infinia card?",
    "top_k": 5,
    "include_sources": true
  }'
```

### Quick GET Query (for testing)
```bash
curl "http://localhost:8000/api/query/simple?q=What+is+the+late+payment+fee?"
```

---

## 5. View the Dashboard

Go to **http://localhost:8000/dashboard** to see:
- Total documents indexed
- Total chunks in the vector DB
- Total queries run
- Average response time
- Recent query log

---

## 6. Manage Documents via API

```bash
# List all documents
curl http://localhost:8000/api/documents/

# Get a specific document
curl http://localhost:8000/api/documents/<document_id>

# Delete a document (removes from MongoDB + ChromaDB)
curl -X DELETE http://localhost:8000/api/documents/<document_id>
```

---

## 7. Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "postgres_connected": false,
  "mongodb_connected": true,
  "chroma_connected": true,
  "total_documents": 3,
  "total_chunks": 412,
  "database_type": "mongodb"
}
```

---

## 8. How the Pipeline Works (Under the Hood)

```
You Upload a File
      ↓
File is saved to ./data/uploads/
      ↓
Parser extracts text (PDF → pages, XLSX → rows, DOCX → paragraphs...)
      ↓
Text is split into ~512 character chunks with overlap
      ↓
Each chunk is converted to a 384-dimensional vector embedding
      ↓
Embeddings stored in ChromaDB (vector similarity search)
Chunk metadata stored in MongoDB (title, page, keywords...)
      ↓
You Ask a Question
      ↓
Question is embedded → Top-K most similar chunks retrieved from ChromaDB
      ↓
Chunks passed as context to GPT-4o-mini → Answer generated
      ↓
Answer + source citations returned to you
```

---

## 9. Configuration (`.env`)

Key settings you may want to change:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `LLM_MODEL` | `gpt-4o-mini` | Change to `gpt-4o` for better quality |
| `CHUNK_SIZE` | `512` | Characters per chunk (larger = more context) |
| `TOP_K_RESULTS` | `5` | How many chunks to retrieve per query |
| `MONGODB_URL` | `mongodb://localhost:27017/hdfc_rag` | MongoDB connection |

### Azure OpenAI Configuration
To use Azure OpenAI, update your `.env` with:
```env
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_AD_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_API_VERSION=2023-05-15
```

After changing `.env`, restart the server.
