"""
FastAPI application entry point for claim extractor API.
"""

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from routers.claims import router as claims_router
import logging
import traceback

load_dotenv()

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="TRUST Agents API",
    description="API for 4 agents from text using crewAI",
    version="0.1.0",
)

# Middleware for error handling and logging
@app.middleware("http")
async def log_exceptions_middleware(request: Request, call_next):
    """
    Middleware to catch and log all unhandled exceptions.
    This ensures ALL 500 errors are logged with full tracebacks.
    """
    try:
        logger.info(f"Incoming request: {request.method} {request.url.path}")
        response = await call_next(request)
        logger.info(
            f"Response: {request.method} {request.url.path} - Status: {response.status_code}"
        )
        return response
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(
            f"UNHANDLED EXCEPTION in {request.method} {request.url.path}"
        )
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception message: {str(e)}")
        logger.error(f"Full traceback:\n{error_trace}")
        
        # Return JSON error response
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(e),
                "error_type": type(e).__name__,
                "path": request.url.path,
                "traceback": error_trace,
            },
        )

# Include routers
app.include_router(claims_router)

# Health endpoint (default tag)
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

# Export app for FastAPI CLI
__all__ = ["app"]
