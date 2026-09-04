import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rag_engine import rag_engine, EMBEDDING_MODEL, DEFAULT_MODEL, PDF_PATH

app = FastAPI(
    title="NTP RAG Assistant API",
    description="FastAPI Backend for National Transformation Program (NTP) RAG Chat Assistant",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Schemas
class MessageItem(BaseModel):
    role: str = Field(..., description="Role of the message author: 'user' or 'assistant'")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User question or prompt")
    history: Optional[List[MessageItem]] = Field(default=[], description="Previous conversation turns")
    top_k: Optional[int] = Field(default=3, ge=1, le=10, description="Number of context chunks to retrieve")
    model: Optional[str] = Field(default=DEFAULT_MODEL, description="OpenAI LLM model identifier")
    temperature: Optional[float] = Field(default=0.2, ge=0.0, le=1.0, description="Sampling temperature")
    api_key: Optional[str] = Field(default=None, description="Optional user-provided OpenAI API key")

class SourceChunk(BaseModel):
    id: int
    page: int
    text: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    query: str
    model: str

class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(default=3, ge=1, le=10)
    api_key: Optional[str] = Field(default=None, description="Optional user-provided OpenAI API key")

class HealthResponse(BaseModel):
    status: str
    document: str
    total_chunks: int
    dimension: int
    embedding_model: str
    default_model: str

@app.on_event("startup")
def startup_event():
    """Ensure RAG engine index is loaded when server starts."""
    try:
        rag_engine.build_or_load_index()
    except Exception as e:
        print(f"Error initializing RAG engine on startup: {e}")

@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check and index statistics."""
    if not rag_engine.is_initialized:
        rag_engine.build_or_load_index()

    return HealthResponse(
        status="online",
        document=PDF_PATH,
        total_chunks=len(rag_engine.chunks),
        dimension=rag_engine.dimension,
        embedding_model=EMBEDDING_MODEL,
        default_model=DEFAULT_MODEL
    )

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """Processes user query, retrieves relevant report context, and returns OpenAI LLM answer."""
    try:
        history_dicts = [{"role": m.role, "content": m.content} for m in request.history] if request.history else []
        result = rag_engine.generate_answer(
            query=request.message,
            history=history_dicts,
            top_k=request.top_k,
            model=request.model,
            temperature=request.temperature,
            api_key=request.api_key
        )
        return ChatResponse(
            answer=result["answer"],
            sources=[
                SourceChunk(
                    id=s["id"],
                    page=s["page"],
                    text=s["text"],
                    score=s["score"]
                ) for s in result["sources"]
            ],
            query=result["query"],
            model=result["model"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.post("/retrieve")
def retrieve_endpoint(request: RetrieveRequest):
    """Directly queries the FAISS vector index to inspect retrieved context chunks and scores."""
    try:
        results = rag_engine.retrieve(query=request.query, top_k=request.top_k, api_key=request.api_key)
        return {"query": request.query, "top_k": request.top_k, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

@app.post("/reindex")
def reindex_endpoint(background_tasks: BackgroundTasks):
    """Triggers background rebuild of the FAISS vector index from the PDF."""
    try:
        background_tasks.add_task(rag_engine.build_or_load_index, force_rebuild=True)
        return {"status": "reindexing_started", "message": "FAISS vector index rebuild initiated in background."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reindex error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api:app", host=host, port=port, reload=False)
