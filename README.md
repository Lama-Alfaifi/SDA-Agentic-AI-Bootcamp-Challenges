# 🏛️ National Transformation Program (NTP) 2025 RAG Assistant

An enterprise-grade Retrieval-Augmented Generation (RAG) assistant for the **National Transformation Program (NTP) Annual Report 2025**.

Built with **FastAPI**, **FAISS Vector Search**, **OpenAI (Embeddings & Chat)**, **Streamlit UI**, and full **Docker** containerization.

---

## 🌟 Key Features

- **Accurate Grounded Q&A**: Answers user questions based strictly on the 92-page NTP 2025 Annual Report.
- **Smart Text Ingestion & Ligature Repair**: PyMuPDF extraction with ligature cleaning (`ﬀ`, `ﬁ`, `ﬂ`, `ﬃ`, `ﬄ`, `/f_`) and semantic sentence-boundary chunking.
- **Dense Vector Search (FAISS)**: 1536-dimensional embeddings with OpenAI's `text-embedding-3-small` and Cosine Similarity search.
- **FastAPI REST Backend**: High-performance async API with `/chat`, `/health`, `/retrieve`, and `/reindex` endpoints.
- **Streamlit Chat Interface**: Clean multi-turn chat with conversation history, suggested questions, and collapsible source citations displaying page numbers and relevance scores.
- **Docker Ready**: One-command containerized deployment with Docker Compose.

---

## 🚀 Quick Start with Docker (Recommended)

### 1. Clone the repository
```bash
git clone https://github.com/aalkhamis09/Nationa-Transformation-Program-Assistant.git
cd Nationa-Transformation-Program-Assistant
```

### 2. Configure environment variables (Optional)
Create a `.env` file from the template:
```bash
# On Linux / macOS
cp .env.example .env

# On Windows (PowerShell)
Copy-Item .env.example .env
```
Open `.env` and insert your OpenAI API key (or you can enter it directly in the Streamlit web sidebar):
```ini
OPENAI_API_KEY=sk-proj-...
```

### 3. Run with Docker Compose
```bash
docker compose up --build
```

That's it!
- 🎨 **Streamlit Web UI**: [http://localhost:8501](http://localhost:8501)
- 📡 **FastAPI Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

> 💡 **Tip**: You can enter your OpenAI API key directly in the Streamlit web UI sidebar settings if you don't want to create a `.env` file.

---

## 💻 Local Setup (Without Docker)

### 1. Prerequisites
- Python 3.10+
- OpenAI API Key (in `.env` or entered in the Streamlit UI)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure `.env` (Optional)
You can set your API key in `.env` or enter it in the web interface:
```ini
OPENAI_API_KEY=sk-proj-...
```

### 4. Run Both Backend & Frontend
```bash
python run.py
```

Or run services individually:
```bash
# Terminal 1: FastAPI Backend
python api.py

# Terminal 2: Streamlit Frontend
streamlit run app.py
```

---

## 📁 Project Structure

```
.
├── .dockerignore                     # Docker build exclusions
├── .env.example                      # Environment variables template
├── .gitignore                        # Git exclusions (protects .env and cache)
├── api.py                            # FastAPI REST API Backend
├── app.py                            # Streamlit Web Application
├── docker-compose.yml                # Docker Compose orchestration
├── Dockerfile                        # Multi-service container specification
├── ntp_en_annual_report_2025.pdf     # Knowledge base source PDF
├── RAG (G5).ipynb                    # Original exploratory notebook
├── rag_engine.py                     # Document ingestion, FAISS index & OpenAI RAG engine
├── requirements.txt                  # Python dependencies
├── run.py                            # Unified concurrent runner script
└── README.md                         # Project documentation
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server status, document metadata, and total chunk count |
| `POST` | `/chat` | Multi-turn conversational RAG query with source citations |
| `POST` | `/retrieve` | Vector search context chunk inspection |
| `POST` | `/reindex` | Background vector index rebuild |
