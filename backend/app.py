"""HTTP routes and optional serving of the built React frontend."""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.schemas import Transaction
from backend.service import PROJECT_ROOT, assess_transaction, get_metadata, load_bundle

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load the model before requests arrive, then release it on shutdown."""
    application.state.bundle = load_bundle()
    # Computing metadata now also detects a broken model/schema at startup.
    application.state.metadata = get_metadata(application.state.bundle)
    yield
    application.state.bundle = None


app = FastAPI(
    title="FraudLens API",
    description="Explainable banking fraud predictions for a synthetic-data portfolio project.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
def health(request: Request) -> dict:
    """Let the interface check that the model loaded successfully."""
    return {"status": "ready", "model_version": request.app.state.bundle["version"]}


@app.get("/api/metadata")
def metadata(request: Request) -> dict:
    """Provide trained category options, thresholds, and sample transactions."""
    return request.app.state.metadata


@app.post("/api/predict")
def predict(transaction: Transaction, request: Request) -> dict:
    """Validate an input, then run CPU work in FastAPI's sync worker thread."""
    try:
        return assess_transaction(request.app.state.bundle, transaction.to_record())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Model inference failed")
        raise HTTPException(status_code=500, detail="Prediction failed. Check the backend terminal for details.") from exc


# Serve only the frontend build, never the repository or the private model files.
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/", include_in_schema=False)
def index():
    """Show the built React app, or a useful setup message before its first build."""
    if (FRONTEND_DIST / "index.html").is_file():
        return FileResponse(FRONTEND_DIST / "index.html")
    return {"message": "API is running. Start React on port 5173, or build frontend/ first.", "docs": "/docs"}
