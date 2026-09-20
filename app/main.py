import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes_chat import router as chat_router
from app.api.routes_health import router as health_router
from app.api.routes_ingest import router as ingest_router
from app.api.routes_visual import router as visual_router

from dotenv import load_dotenv
load_dotenv()

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
app.include_router(chat_router)


@app.get("/")
async def root():
    """Root status check."""
    return {
        "project": "OmniBrain",
        "status": "online",
        "role_2b": "Visual Analytics & Multi-Modal Tool Integrator active",
        "docs": "/docs",
        "workspace": "/workspace",
    }


@app.get("/workspace", response_class=FileResponse, tags=["UI"])
@app.get("/ui", response_class=FileResponse, tags=["UI"])
async def serve_workspace():
    """Serve the OmniBrain Quant Workspace 3-Panel frontend."""
    frontend_dist_html = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist", "index.html")
    if os.path.exists(frontend_dist_html):
        return FileResponse(frontend_dist_html, media_type="text/html")
    ui_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui", "index.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path, media_type="text/html")
    return HTMLResponse("<h1>OmniBrain Quant Workspace UI not found</h1>", status_code=404)


# Mount built frontend static assets if available
frontend_dist_assets = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist", "assets")
if os.path.exists(frontend_dist_assets):
    app.mount("/assets", StaticFiles(directory=frontend_dist_assets), name="assets")

# Mount extracted images for frontend artifact viewing
os.makedirs("storage/extracted_images", exist_ok=True)
app.mount("/api/v1/images", StaticFiles(directory="storage/extracted_images"), name="extracted_images")


