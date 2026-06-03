"""Train SVD and KNN using Surprise with grid search."""
import os
import joblib
from surprise import SVD, KNNWithMeans, Dataset, Reader
from surprise.model_selection import GridSearchCV
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_svd(train_df: pd.DataFrame, param_grid: dict, models_dir: str = "models/"):
    """Train SVD with grid search and save best model."""
    os.makedirs(models_dir, exist_ok=True)
    logger.info("Training SVD with grid search …")

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_df[["user_id", "movie_id", "rating"]], reader)

    gs = GridSearchCV(SVD, param_grid, measures=["rmse"], cv=5, n_jobs=-1)
    gs.fit(data)

    best_rmse = gs.best_score["rmse"]
    best_params = gs.best_params["rmse"]
    logger.info(f"SVD best RMSE: {best_rmse:.4f} | Params: {best_params}")

    best_model = gs.best_estimator["rmse"]
    model_path = os.path.join(models_dir, "svd_model.pkl")
    joblib.dump(best_model, model_path)
    logger.info(f"SVD model saved to {model_path}")

    return best_model, best_rmse, best_params


def train_knn(train_df: pd.DataFrame, param_grid: dict, models_dir: str = "models/"):
    """Train KNN with grid search and save best model."""
    os.makedirs(models_dir, exist_ok=True)
    logger.info("Training KNN with grid search …")

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_df[["user_id", "movie_id", "rating"]], reader)

    gs = GridSearchCV(KNNWithMeans, param_grid, measures=["rmse"], cv=5, n_jobs=-1)
    gs.fit(data)

    best_rmse = gs.best_score["rmse"]
    best_params = gs.best_params["rmse"]
    logger.info(f"KNN best RMSE: {best_rmse:.4f} | Params: {best_params}")

    best_model = gs.best_estimator["rmse"]
    model_path = os.path.join(models_dir, "knn_model.pkl")
    joblib.dump(best_model, model_path)
    logger.info(f"KNN model saved to {model_path}")

    return best_model, best_rmse, best_params


def load_surprise_model(model_path: str):
    """Load a saved Surprise model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)
