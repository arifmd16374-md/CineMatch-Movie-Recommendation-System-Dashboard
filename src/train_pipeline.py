"""
End-to-end training pipeline:
  1. Load data
  2. EDA
  3. Train SVD & KNN (Surprise)
  4. Train NCF (PyTorch)
  5. Evaluate all models
"""
import os
import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.data.loader import load_ratings, load_movies, get_dataset_stats
from src.data.eda import run_eda
from src.data.pytorch_dataset import RatingsDataset, test_round_trip
from src.models.surprise_models import train_svd, train_knn
from src.models.ncf_model import train_ncf, load_ncf_model
from src.evaluation.evaluate import run_evaluation

logger = get_logger(__name__)


def main():
    cfg = load_config()
    data_cfg = cfg["data"]
    ncf_cfg = cfg["ncf"]
    eval_cfg = cfg["evaluation"]

    # ── Phase 2: Data ────────────────────────────────────────────
    train_df, test_df, full_df = load_ratings(
        data_dir=data_cfg["data_dir"],
        test_ratio=data_cfg["test_ratio"],
        seed=data_cfg["random_seed"],
    )
    stats = get_dataset_stats(full_df)
    logger.info(f"Dataset stats: {stats}")

    run_eda(full_df, output_dir=os.path.join(data_cfg["output_dir"], "eda/"))

    # Encode for NCF
    train_dataset = RatingsDataset(train_df)
    test_round_trip(train_dataset)  # Verify encoding property

    # Build a test dataset using train encodings (so IDs are consistent)
    from src.data.pytorch_dataset import RatingsDataset as _D
    # Create test set with filtered known IDs only
    known_users = set(train_dataset.user_to_idx.keys())
    known_movies = set(train_dataset.movie_to_idx.keys())
    test_filtered = test_df[
        test_df["user_id"].isin(known_users) & test_df["movie_id"].isin(known_movies)
    ].copy()

    val_dataset = RatingsDataset.__new__(RatingsDataset)
    val_dataset.user_to_idx = train_dataset.user_to_idx
    val_dataset.movie_to_idx = train_dataset.movie_to_idx
    val_dataset.idx_to_user = train_dataset.idx_to_user
    val_dataset.idx_to_movie = train_dataset.idx_to_movie
    val_dataset.n_users = train_dataset.n_users
    val_dataset.n_movies = train_dataset.n_movies

    import torch as _t
    val_dataset.user_ids = _t.tensor(
        test_filtered["user_id"].map(train_dataset.user_to_idx).values, dtype=_t.long
    )
    val_dataset.movie_ids = _t.tensor(
        test_filtered["movie_id"].map(train_dataset.movie_to_idx).values, dtype=_t.long
    )
    val_dataset.ratings = _t.tensor(test_filtered["rating"].values, dtype=_t.float32)

    bs = ncf_cfg.get("batch_size", 256)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

    # ── Phase 3: Collaborative Filtering ─────────────────────────
    svd_model, svd_cv_rmse, svd_params = train_svd(
        train_df,
        cfg.get("svd", {}).get("param_grid", {"n_factors": [100], "n_epochs": [20]}),
        models_dir=data_cfg["models_dir"],
    )
    knn_model, knn_cv_rmse, knn_params = train_knn(
        train_df,
        cfg.get("knn", {}).get("param_grid", {"k": [40], "sim_options": {"name": ["pearson"], "user_based": [True]}}),
        models_dir=data_cfg["models_dir"],
    )

    # ── Phase 4: Neural CF ───────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    ncf_model, ncf_val_rmse = train_ncf(
        train_loader,
        val_loader,
        n_users=train_dataset.n_users,
        n_movies=train_dataset.n_movies,
        config=ncf_cfg,
        models_dir=data_cfg["models_dir"],
        device=device,
    )

    # ── Phase 5: Evaluation ──────────────────────────────────────
    models = {"knn": knn_model, "svd": svd_model, "ncf": ncf_model}
    comparison = run_evaluation(
        models=models,
        test_df=test_filtered,
        dataset=train_dataset,
        ncf_device=device,
        k_values=eval_cfg.get("k_values", [5, 10, 20]),
        threshold=eval_cfg.get("relevance_threshold", 4.0),
        output_dir=data_cfg["output_dir"],
    )

    print("\n===== FINAL MODEL COMPARISON =====")
    print(comparison.to_string())
    print("==================================\n")

    # Save dataset metadata for API
    import json
    meta = {
        "n_users": int(train_dataset.n_users),
        "n_movies": int(train_dataset.n_movies),
        "user_to_idx": {str(int(k)): int(v) for k, v in train_dataset.user_to_idx.items()},
        "movie_to_idx": {str(int(k)): int(v) for k, v in train_dataset.movie_to_idx.items()},
        "idx_to_user": {str(int(k)): int(v) for k, v in train_dataset.idx_to_user.items()},
        "idx_to_movie": {str(int(k)): int(v) for k, v in train_dataset.idx_to_movie.items()},
    }
    os.makedirs(data_cfg["models_dir"], exist_ok=True)
    with open(os.path.join(data_cfg["models_dir"], "dataset_meta.json"), "w") as f:
        json.dump(meta, f)
    logger.info("Dataset metadata saved.")


if __name__ == "__main__":
    main()
