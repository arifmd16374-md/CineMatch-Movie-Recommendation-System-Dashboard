"""Evaluate all trained models and save comparison CSV."""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import evaluate_model_on_test
from src.utils.logger import get_logger

logger = get_logger(__name__)


def predict_surprise(model, test_df: pd.DataFrame) -> np.ndarray:
    """Generate predictions from a Surprise model."""
    preds = []
    for _, row in test_df.iterrows():
        try:
            pred = model.predict(int(row["user_id"]), int(row["movie_id"])).est
        except Exception as exc:
            logger.warning(f"Prediction failed for ({row['user_id']}, {row['movie_id']}): {exc}")
            pred = 3.0  # fallback to global mean
        preds.append(pred)
    return np.array(preds)


def predict_ncf(model, dataset, test_df: pd.DataFrame, device: str = "cpu") -> np.ndarray:
    """Generate predictions from the NCF model."""
    model.eval()
    user_ids_enc = []
    movie_ids_enc = []
    valid_rows = []

    for idx, row in test_df.iterrows():
        try:
            u_enc = dataset.encode_user(int(row["user_id"]))
            m_enc = dataset.encode_movie(int(row["movie_id"]))
            user_ids_enc.append(u_enc)
            movie_ids_enc.append(m_enc)
            valid_rows.append(idx)
        except ValueError:
            logger.warning(f"Unknown ID in NCF prediction: user={row['user_id']}, movie={row['movie_id']}")

    user_tensor = torch.tensor(user_ids_enc, dtype=torch.long).to(device)
    movie_tensor = torch.tensor(movie_ids_enc, dtype=torch.long).to(device)

    with torch.no_grad():
        preds = model(user_tensor, movie_tensor).cpu().numpy()

    # Map preds back to full test_df length (fallback 3.0 for unknowns)
    full_preds = np.full(len(test_df), 3.0)
    valid_positions = [list(test_df.index).index(r) for r in valid_rows]
    for pos, pred in zip(valid_positions, preds):
        full_preds[pos] = pred

    return full_preds


def run_evaluation(
    models: dict,
    test_df: pd.DataFrame,
    dataset=None,
    ncf_device: str = "cpu",
    k_values=None,
    threshold: float = 4.0,
    output_dir: str = "outputs/",
) -> pd.DataFrame:
    """Evaluate all models and return comparison dataframe."""
    os.makedirs(output_dir, exist_ok=True)
    if k_values is None:
        k_values = [5, 10, 20]

    y_true = test_df["rating"].values
    all_results = []

    for model_name, model_obj in models.items():
        logger.info(f"Evaluating {model_name} …")

        if model_name == "ncf":
            y_pred = predict_ncf(model_obj, dataset, test_df, device=ncf_device)
        else:
            y_pred = predict_surprise(model_obj, test_df)

        results = evaluate_model_on_test(
            model_name, y_true, y_pred, test_df, k_values=k_values, threshold=threshold
        )
        all_results.append(results)

    comparison_df = pd.DataFrame(all_results).set_index("model")
    csv_path = os.path.join(output_dir, "model_comparison.csv")
    comparison_df.to_csv(csv_path, float_format="%.4f")
    logger.info(f"Model comparison saved to {csv_path}")
    logger.info("\n" + comparison_df.to_string())

    return comparison_df
