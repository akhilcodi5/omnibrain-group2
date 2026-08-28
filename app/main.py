"""FastAPI application entrypoint for OmniBrain."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_health import router as health_router
from app.api.routes_ingest import router as ingest_router
from app.api.routes_visual import router as visual_router

app = FastAPI(
    title="OmniBrain: Agentic Multi-Modal RAG Orchestrator",
    description="Enterprise-grade Agentic Multi-Modal RAG API for financial document intelligence, visual analytics, and cross-modal reasoning.",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(health_router, prefix="/api/v1", tags=["Health"])
app.include_router(ingest_router)
app.include_router(visual_router)


@app.get("/")
async def root():
    """Root status check."""
    return {
        "project": "OmniBrain",
        "status": "online",
        "role_2b": "Visual Analytics & Multi-Modal Tool Integrator active",
        "docs": "/docs",
    }
