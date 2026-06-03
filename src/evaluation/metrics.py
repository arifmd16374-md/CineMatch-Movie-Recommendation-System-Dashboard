"""Rating and ranking evaluation metrics."""
import numpy as np
import pandas as pd
from typing import List

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Clipped RMSE (predictions clipped to [1, 5])."""
    y_pred_clipped = np.clip(y_pred, 1.0, 5.0)
    return round(float(np.sqrt(np.mean((y_true - y_pred_clipped) ** 2))), 4)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Clipped MAE."""
    y_pred_clipped = np.clip(y_pred, 1.0, 5.0)
    return round(float(np.mean(np.abs(y_true - y_pred_clipped))), 4)


def dcg_at_k(relevances: List[float], k: int) -> float:
    """Discounted Cumulative Gain at K."""
    relevances = relevances[:k]
    return sum(
        rel / np.log2(idx + 2) for idx, rel in enumerate(relevances)
    )


def ndcg_at_k(y_true_by_user: dict, y_pred_by_user: dict, k: int, threshold: float = 4.0) -> float:
    """NDCG@K averaged over users with at least one relevant item."""
    scores = []
    for user, true_ratings in y_true_by_user.items():
        if user not in y_pred_by_user:
            continue
        pred_ratings = y_pred_by_user[user]
        # Sort by predicted score descending
        sorted_by_pred = sorted(pred_ratings.items(), key=lambda x: x[1], reverse=True)
        predicted_order = [movie for movie, _ in sorted_by_pred]

        # Binary relevance
        relevances_pred = [1.0 if true_ratings.get(m, 0) >= threshold else 0.0
                           for m in predicted_order[:k]]
        dcg = dcg_at_k(relevances_pred, k)

        # Ideal DCG
        ideal_relevances = sorted(
            [1.0 if r >= threshold else 0.0 for r in true_ratings.values()],
            reverse=True
        )
        idcg = dcg_at_k(ideal_relevances, k)

        if idcg == 0:
            continue
        scores.append(dcg / idcg)

    if not scores:
        return 0.0
    return round(float(np.mean(scores)), 4)


def map_at_k(y_true_by_user: dict, y_pred_by_user: dict, k: int, threshold: float = 4.0) -> float:
    """Mean Average Precision @ K."""
    aps = []
    for user, true_ratings in y_true_by_user.items():
        if user not in y_pred_by_user:
            continue
        pred_ratings = y_pred_by_user[user]
        sorted_by_pred = sorted(pred_ratings.items(), key=lambda x: x[1], reverse=True)
        top_k = [m for m, _ in sorted_by_pred[:k]]

        relevant_set = {m for m, r in true_ratings.items() if r >= threshold}
        if not relevant_set:
            continue

        hits = 0
        precision_sum = 0.0
        for i, movie in enumerate(top_k, 1):
            if movie in relevant_set:
                hits += 1
                precision_sum += hits / i

        aps.append(precision_sum / min(len(relevant_set), k))

    if not aps:
        return 0.0
    return round(float(np.mean(aps)), 4)


def build_user_dicts(test_df: pd.DataFrame, pred_series: pd.Series):
    """Build per-user true/pred dicts from test df and predictions."""
    test_df = test_df.copy()
    test_df["predicted"] = pred_series.values

    y_true = (
        test_df.groupby("user_id")
        .apply(lambda x: dict(zip(x["movie_id"], x["rating"])))
        .to_dict()
    )
    y_pred = (
        test_df.groupby("user_id")
        .apply(lambda x: dict(zip(x["movie_id"], x["predicted"])))
        .to_dict()
    )
    return y_true, y_pred


def evaluate_model_on_test(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_df: pd.DataFrame,
    k_values: List[int] = None,
    threshold: float = 4.0,
) -> dict:
    """Full evaluation — RMSE, MAE, NDCG@K, MAP@K."""
    if k_values is None:
        k_values = [5, 10, 20]

    results = {
        "model": model_name,
        "rmse": compute_rmse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
    }

    pred_series = pd.Series(y_pred, index=test_df.index)
    y_true_user, y_pred_user = build_user_dicts(test_df, pred_series)

    for k in k_values:
        results[f"ndcg@{k}"] = ndcg_at_k(y_true_user, y_pred_user, k=k, threshold=threshold)
        results[f"map@{k}"] = map_at_k(y_true_user, y_pred_user, k=k, threshold=threshold)

    logger.info(f"[{model_name}] " + " | ".join(f"{k}: {v}" for k, v in results.items() if k != "model"))
    return results
