"""FastAPI application for serving movie recommendation predictions."""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
import time

from src.api.models_store import load_all_models, get_store, predict_rating, get_top_n
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)

app = FastAPI(
    title="Movie Recommendation API",
    description="Predict movie ratings and get top-N recommendations using SVD, KNN, and NCF models.",
    version="1.0.0",
)


# ── Pydantic Models ────────────────────────────────────────────
class PredictRequest(BaseModel):
    user_id: int = Field(..., gt=0, description="Positive user ID")
    movie_id: int = Field(..., gt=0, description="Positive movie ID")


class PredictResponse(BaseModel):
    user_id: int
    movie_id: int
    model: str
    predicted_rating: float


class TopNResponse(BaseModel):
    user_id: int
    model: str
    recommendations: List[dict]


class StatusResponse(BaseModel):
    status: str
    available_models: List[str]


# ── Startup ────────────────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    logger.info("Starting API server and loading models …")
    load_all_models()
    cfg = load_config()
    host = cfg["api"].get("host", "0.0.0.0")
    port = cfg["api"].get("port", 8000)
    logger.info(f"API ready on {host}:{port}")


# ── Endpoints ──────────────────────────────────────────────────
@app.get("/", response_model=StatusResponse)
def root():
    """API health check and available models."""
    store = get_store()
    return StatusResponse(status="ok", available_models=store["available"])


@app.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    model: Optional[str] = Query("svd", description="Model to use: svd, knn, ncf")
):
    """Predict rating for a user-movie pair."""
    store = get_store()
    if model not in store["available"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not available. Available: {store['available']}"
        )

    try:
        pred = predict_rating(req.user_id, req.movie_id, model)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=503, detail="Model prediction failed")

    return PredictResponse(
        user_id=req.user_id,
        movie_id=req.movie_id,
        model=model,
        predicted_rating=round(pred, 3),
    )


@app.get("/recommend/{user_id}", response_model=TopNResponse)
def recommend(
    user_id: int,
    model: Optional[str] = Query("svd", description="Model to use: svd, knn, ncf"),
    n: Optional[int] = Query(10, ge=1, le=100, description="Number of recommendations (1-100)"),
):
    """Get top-N movie recommendations for a user."""
    store = get_store()
    if model not in store["available"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not available. Available: {store['available']}"
        )

    try:
        recommendations = get_top_n(user_id, model, n=n)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error(f"Recommendation error: {exc}")
        raise HTTPException(status_code=503, detail="Model recommendation failed")

    if not recommendations:
        logger.warning(f"No recommendations generated for user {user_id} with {model}")

    return TopNResponse(
        user_id=user_id,
        model=model,
        recommendations=recommendations,
    )
