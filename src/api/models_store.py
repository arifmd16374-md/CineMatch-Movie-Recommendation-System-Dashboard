"""Model loading and in-memory store for the API."""
import json
import os
import time
import joblib
import torch

from src.utils.logger import get_logger
from src.utils.config import load_config
from src.models.ncf_model import NCFModel

logger = get_logger(__name__)

_STORE = {
    "svd": None,
    "knn": None,
    "ncf": None,
    "meta": None,
    "movies": None,
    "available": [],
}


def load_all_models():
    cfg = load_config()
    models_dir = cfg["data"]["models_dir"]
    data_dir = cfg["data"]["data_dir"]
    ncf_cfg = cfg["ncf"]

    available = []

    # Surprise models
    for name in ("svd", "knn"):
        path = os.path.join(models_dir, f"{name}_model.pkl")
        try:
            _STORE[name] = joblib.load(path)
            available.append(name)
            logger.info(f"Loaded {name.upper()} from {path}")
        except Exception as exc:
            logger.error(f"Failed to load {name}: {exc}")

    # Dataset metadata
    meta_path = os.path.join(models_dir, "dataset_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        _STORE["meta"] = meta
    else:
        logger.error("dataset_meta.json not found — NCF won't be available")

    # NCF model
    ncf_path = os.path.join(models_dir, "ncf_model.pth")
    if os.path.exists(ncf_path) and _STORE["meta"]:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            n_users = _STORE["meta"]["n_users"]
            n_movies = _STORE["meta"]["n_movies"]
            model = NCFModel(
                n_users=n_users,
                n_movies=n_movies,
                embedding_dim=ncf_cfg.get("embedding_dim", 64),
                hidden_layers=ncf_cfg.get("hidden_layers", [256, 128, 64]),
                dropout_rate=ncf_cfg.get("dropout_rate", 0.3),
            )
            model.load_state_dict(torch.load(ncf_path, map_location=device))
            model.eval()
            _STORE["ncf"] = model
            available.append("ncf")
            logger.info(f"Loaded NCF from {ncf_path}")
        except Exception as exc:
            logger.error(f"Failed to load NCF: {exc}")

    # Movie titles
    movies_path = os.path.join(data_dir, "ml-100k", "u.item")
    if os.path.exists(movies_path):
        import pandas as pd
        movies = pd.read_csv(
            movies_path, sep="|", encoding="latin-1", header=None,
            usecols=[0, 1], names=["movie_id", "title"]
        )
        _STORE["movies"] = dict(zip(movies["movie_id"].astype(str), movies["title"]))
        logger.info(f"Loaded {len(_STORE['movies'])} movie titles")

    _STORE["available"] = available
    logger.info(f"Available models: {available}")
    return available


def get_store():
    return _STORE


def predict_rating(user_id: int, movie_id: int, model_name: str) -> float:
    """Predict a single rating. Returns float in [1, 5]."""
    store = _STORE
    start = time.perf_counter()

    if model_name not in store["available"]:
        raise ValueError(f"Model '{model_name}' is not available.")

    if model_name in ("svd", "knn"):
        model = store[model_name]
        pred = float(model.predict(user_id, movie_id).est)
    elif model_name == "ncf":
        meta = store["meta"]
        if str(user_id) not in meta["user_to_idx"]:
            raise ValueError(f"Unknown user_id: {user_id}")
        if str(movie_id) not in meta["movie_to_idx"]:
            raise ValueError(f"Unknown movie_id: {movie_id}")

        u_enc = meta["user_to_idx"][str(user_id)]
        m_enc = meta["movie_to_idx"][str(movie_id)]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        u_t = torch.tensor([u_enc], dtype=torch.long).to(device)
        m_t = torch.tensor([m_enc], dtype=torch.long).to(device)
        with torch.no_grad():
            pred = float(store["ncf"](u_t, m_t).cpu().item())
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pred = max(0.5, min(5.0, pred))
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"Predict [{model_name}] user={user_id} movie={movie_id} → {pred:.3f} ({elapsed_ms:.1f}ms)")
    return pred


def get_top_n(user_id: int, model_name: str, n: int = 10) -> list:
    """Return top-N (movie_id, predicted_rating, title) for a user."""
    store = _STORE
    if model_name not in store["available"]:
        raise ValueError(f"Model '{model_name}' is not available.")

    if model_name in ("svd", "knn"):
        meta = store["meta"]
        if meta is None:
            raise RuntimeError("Dataset metadata not loaded.")
        all_movies = [int(m) for m in meta["movie_to_idx"].keys()]
    else:
        meta = store["meta"]
        if str(user_id) not in meta["user_to_idx"]:
            raise ValueError(f"Unknown user_id: {user_id}")
        all_movies = [int(m) for m in meta["movie_to_idx"].keys()]

    preds = []
    for movie_id in all_movies:
        try:
            pred = predict_rating(user_id, movie_id, model_name)
            preds.append((movie_id, pred))
        except Exception:
            pass

    preds.sort(key=lambda x: x[1], reverse=True)
    top_n = preds[:n]

    movies_map = store.get("movies") or {}
    results = [
        {
            "movie_id": mid,
            "predicted_rating": round(rating, 3),
            "title": movies_map.get(str(mid), f"Movie {mid}"),
        }
        for mid, rating in top_n
    ]
    return results
