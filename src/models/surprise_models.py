"""
Train SVD and KNN collaborative filtering models.
Pure numpy/scipy implementation — no scikit-surprise required.
Compatible with Python 3.11+ and all cloud platforms.
"""
import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  PURE-PYTHON SVD COLLABORATIVE FILTER
# ─────────────────────────────────────────────────────────────────────────────
class SVDModel:
    """Matrix-factorisation collaborative filter using truncated SVD."""

    def __init__(self, n_factors: int = 100, n_epochs: int = 20,
                 lr: float = 0.005, reg: float = 0.02):
        self.n_factors = n_factors
        self.n_epochs  = n_epochs
        self.lr        = lr
        self.reg       = reg

    def fit(self, train_df: pd.DataFrame):
        users  = sorted(train_df["user_id"].unique())
        movies = sorted(train_df["movie_id"].unique())
        self.user_idx  = {u: i for i, u in enumerate(users)}
        self.movie_idx = {m: i for i, m in enumerate(movies)}
        self.idx_user  = {i: u for u, i in self.user_idx.items()}
        self.idx_movie = {i: m for m, i in self.movie_idx.items()}

        nu, nm = len(users), len(movies)
        self.global_mean = float(train_df["rating"].mean())

        # Build sparse rating matrix
        rows = train_df["user_id"].map(self.user_idx).values
        cols = train_df["movie_id"].map(self.movie_idx).values
        vals = train_df["rating"].values.astype(np.float32)
        R = csr_matrix((vals, (rows, cols)), shape=(nu, nm), dtype=np.float32)

        # Truncated SVD on mean-centred matrix
        R_dense = R.toarray()
        row_mean = np.true_divide(R_dense.sum(1),
                                  (R_dense != 0).sum(1).clip(1))
        R_centred = R_dense.copy()
        R_centred[R_dense != 0] -= row_mean[
            np.nonzero(R_dense)[0]]

        k = min(self.n_factors, min(nu, nm) - 1)
        U, sigma, Vt = svds(csr_matrix(R_centred), k=k)
        self.U      = U
        self.sigma  = sigma
        self.Vt     = Vt
        self.row_mean = row_mean

        # Bias terms (SGD)
        self.bu = np.zeros(nu, dtype=np.float32)
        self.bi = np.zeros(nm, dtype=np.float32)
        for _ in range(self.n_epochs):
            for idx in range(len(vals)):
                u, i, r = int(rows[idx]), int(cols[idx]), float(vals[idx])
                err = r - self._raw_predict(u, i)
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])

        logger.info(f"SVDModel fitted: {nu} users, {nm} movies, k={k}")
        return self

    def _raw_predict(self, u_idx: int, i_idx: int) -> float:
        latent = float(np.dot(self.U[u_idx] * self.sigma, self.Vt[:, i_idx]))
        return self.global_mean + self.bu[u_idx] + self.bi[i_idx] + latent

    def predict(self, user_id, movie_id):
        """Returns an object with .est attribute (Surprise-compatible)."""
        u = self.user_idx.get(int(user_id))
        i = self.movie_idx.get(int(movie_id))
        if u is None or i is None:
            est = self.global_mean
        else:
            est = float(np.clip(self._raw_predict(u, i), 1.0, 5.0))
        return _Prediction(est)


# ─────────────────────────────────────────────────────────────────────────────
#  PURE-PYTHON KNN COLLABORATIVE FILTER  (item-based, cosine similarity)
# ─────────────────────────────────────────────────────────────────────────────
class KNNModel:
    """Item-based KNN with cosine similarity."""

    def __init__(self, k: int = 40):
        self.k = k

    def fit(self, train_df: pd.DataFrame):
        users  = sorted(train_df["user_id"].unique())
        movies = sorted(train_df["movie_id"].unique())
        self.user_idx  = {u: i for i, u in enumerate(users)}
        self.movie_idx = {m: i for i, m in enumerate(movies)}

        nu, nm = len(users), len(movies)
        self.global_mean = float(train_df["rating"].mean())

        rows = train_df["user_id"].map(self.user_idx).values
        cols = train_df["movie_id"].map(self.movie_idx).values
        vals = train_df["rating"].values.astype(np.float32)

        # Item mean
        R = np.zeros((nu, nm), dtype=np.float32)
        cnt = np.zeros(nm, dtype=np.int32)
        for u, i, r in zip(rows, cols, vals):
            R[u, i] = r
            cnt[i] += 1
        self.item_mean = np.where(cnt > 0,
                                   R.sum(0) / cnt.clip(1),
                                   self.global_mean)

        # Mean-centred item matrix for cosine sim
        R_c = R.copy()
        for i in range(nm):
            mask = R[:, i] != 0
            R_c[mask, i] -= self.item_mean[i]

        # Cosine similarity between items (nm x nm) — computed lazily per query
        # Store the centred matrix for on-demand similarity
        self.R_c = R_c          # (nu, nm)
        self.R   = R            # (nu, nm) raw
        self.norms = np.linalg.norm(R_c, axis=0).clip(1e-9)  # (nm,)
        logger.info(f"KNNModel fitted: {nu} users, {nm} movies, k={self.k}")
        return self

    def _item_sim(self, i: int) -> np.ndarray:
        """Cosine similarity of item i to all items."""
        dot = self.R_c.T @ self.R_c[:, i]   # (nm,)
        return dot / (self.norms * self.norms[i])

    def predict(self, user_id, movie_id):
        u = self.user_idx.get(int(user_id))
        i = self.movie_idx.get(int(movie_id))
        if u is None or i is None:
            return _Prediction(self.global_mean)

        sims = self._item_sim(i)           # (nm,)
        rated = np.where(self.R[u] != 0)[0]
        rated = rated[rated != i]

        if len(rated) == 0:
            return _Prediction(float(self.item_mean[i]))

        # Top-k neighbours
        s = sims[rated]
        top_k = np.argsort(np.abs(s))[-self.k:]
        s_k = s[top_k]
        r_k = self.R[u][rated[top_k]] - self.item_mean[rated[top_k]]
        denom = np.abs(s_k).sum()
        if denom < 1e-9:
            est = float(self.item_mean[i])
        else:
            est = float(self.item_mean[i] + np.dot(s_k, r_k) / denom)
        est = float(np.clip(est, 1.0, 5.0))
        return _Prediction(est)


class _Prediction:
    """Minimal Surprise-compatible prediction container."""
    def __init__(self, est: float):
        self.est = est


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def train_svd(train_df: pd.DataFrame, param_grid: dict,
              models_dir: str = "models/"):
    """Train SVD model and save."""
    os.makedirs(models_dir, exist_ok=True)
    # Use first value of each param (quick retrain mode)
    n_factors = param_grid.get("n_factors", [100])[0]
    n_epochs  = param_grid.get("n_epochs",  [20])[0]
    lr        = param_grid.get("lr_all",    [0.005])[0]
    reg       = param_grid.get("reg_all",   [0.02])[0]

    logger.info(f"Training SVD: n_factors={n_factors}, epochs={n_epochs}")
    model = SVDModel(n_factors=n_factors, n_epochs=n_epochs,
                     lr=lr, reg=reg).fit(train_df)

    # Quick CV RMSE estimate on training data
    rows = train_df.sample(min(5000, len(train_df)), random_state=42)
    preds = np.array([model.predict(r.user_id, r.movie_id).est
                      for _, r in rows.iterrows()])
    rmse = float(np.sqrt(np.mean((rows["rating"].values - preds) ** 2)))

    path = os.path.join(models_dir, "svd_model.pkl")
    joblib.dump(model, path)
    logger.info(f"SVD saved → {path} | train RMSE≈{rmse:.4f}")
    return model, rmse, {"n_factors": n_factors, "n_epochs": n_epochs}


def train_knn(train_df: pd.DataFrame, param_grid: dict,
              models_dir: str = "models/"):
    """Train KNN model and save."""
    os.makedirs(models_dir, exist_ok=True)
    k = param_grid.get("k", [40])[0]

    logger.info(f"Training KNN: k={k}")
    model = KNNModel(k=k).fit(train_df)

    rows = train_df.sample(min(3000, len(train_df)), random_state=42)
    preds = np.array([model.predict(r.user_id, r.movie_id).est
                      for _, r in rows.iterrows()])
    rmse = float(np.sqrt(np.mean((rows["rating"].values - preds) ** 2)))

    path = os.path.join(models_dir, "knn_model.pkl")
    joblib.dump(model, path)
    logger.info(f"KNN saved → {path} | train RMSE≈{rmse:.4f}")
    return model, rmse, {"k": k}


def load_surprise_model(model_path: str):
    """Load a saved model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)
