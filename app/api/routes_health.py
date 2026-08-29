"""Health check and system status REST endpoints for OmniBrain API."""

from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Health Check")
async def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "service": "OmniBrain API",
        "version": "0.1.0",
    }
