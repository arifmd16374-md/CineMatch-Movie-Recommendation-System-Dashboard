"""MovieLens 100K data loader with download, validation, and train/test split."""
import os
import urllib.request
import zipfile
import pandas as pd
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
RATINGS_FILE = "u.data"
MOVIES_FILE = "u.item"
COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]


def download_movielens(data_dir: str = "data/") -> Path:
    """Download and extract MovieLens 100K if not present."""
    data_path = Path(data_dir)
    dataset_path = data_path / "ml-100k"

    if dataset_path.exists() and (dataset_path / RATINGS_FILE).exists():
        logger.info("MovieLens 100K already present, skipping download.")
        return dataset_path

    os.makedirs(data_path, exist_ok=True)
    zip_path = data_path / "ml-100k.zip"

    logger.info("Downloading MovieLens 100K …")
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            urllib.request.urlretrieve(ML100K_URL, zip_path)
            break
        except Exception as exc:
            logger.warning(f"Download attempt {attempt} failed: {exc}")
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to download MovieLens after {max_retries} attempts."
                ) from exc

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_path)
    os.remove(zip_path)
    logger.info("Download complete.")
    return dataset_path


def load_ratings(data_dir: str = "data/", test_ratio: float = 0.2, seed: int = 42):
    """
    Load and validate ratings, return (train_df, test_df, full_df).
    """
    if not 0.01 <= test_ratio <= 0.99:
        logger.warning(f"Invalid test_ratio {test_ratio}; using default 0.2")
        test_ratio = 0.2

    dataset_path = download_movielens(data_dir)
    ratings_path = dataset_path / RATINGS_FILE

    df = pd.read_csv(ratings_path, sep="\t", names=COLUMNS)
    original_len = len(df)

    # Validation
    valid_mask = (
        df["user_id"].gt(0)
        & df["movie_id"].gt(0)
        & df["rating"].between(1, 5)
        & df["timestamp"].gt(0)
    )
    invalid_count = (~valid_mask).sum()
    if invalid_count:
        logger.warning(f"Dropping {invalid_count} invalid records.")
    df = df[valid_mask].copy()

    if len(df) < 10:
        raise ValueError("Dataset has fewer than 10 valid records.")

    # Shuffle & split
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    logger.info(
        f"Loaded {original_len} records → {len(df)} valid. "
        f"Train: {len(train_df)}, Test: {len(test_df)}, "
        f"Users: {df['user_id'].nunique()}, Movies: {df['movie_id'].nunique()}"
    )
    return train_df, test_df, df


def load_movies(data_dir: str = "data/") -> pd.DataFrame:
    """Load movie metadata (id + title)."""
    dataset_path = download_movielens(data_dir)
    movies_path = dataset_path / MOVIES_FILE
    movies = pd.read_csv(
        movies_path,
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0, 1],
        names=["movie_id", "title"],
    )
    return movies


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """Return basic dataset statistics."""
    return {
        "total_users": int(df["user_id"].nunique()),
        "total_movies": int(df["movie_id"].nunique()),
        "total_ratings": int(len(df)),
        "sparsity": round(
            1 - len(df) / (df["user_id"].nunique() * df["movie_id"].nunique()), 4
        ),
        "rating_mean": round(float(df["rating"].mean()), 4),
        "rating_std": round(float(df["rating"].std()), 4),
        "rating_median": float(df["rating"].median()),
    }
