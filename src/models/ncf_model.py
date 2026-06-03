"""Neural Collaborative Filtering (NCF) model with PyTorch."""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.logger import get_logger

logger = get_logger(__name__)


class NCFModel(nn.Module):
    """Neural Collaborative Filtering with embeddings + MLP."""

    def __init__(
        self,
        n_users: int,
        n_movies: int,
        embedding_dim: int = 64,
        hidden_layers: list = None,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [256, 128, 64]

        if embedding_dim < 1 or embedding_dim > 512:
            raise ValueError(f"embedding_dim must be 1-512, got {embedding_dim}")
        if dropout_rate < 0 or dropout_rate > 0.9:
            raise ValueError(f"dropout_rate must be 0-0.9, got {dropout_rate}")
        if n_users < 1 or n_movies < 1:
            raise ValueError("n_users and n_movies must be positive")

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)

        # MLP layers
        layers = []
        in_dim = 2 * embedding_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids, movie_ids):
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)
        x = torch.cat([user_emb, movie_emb], dim=1)
        out = self.mlp(x).squeeze()
        return out


def train_ncf(
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_users: int,
    n_movies: int,
    config: dict,
    models_dir: str = "models/",
    device: str = "cpu",
):
    """Train NCF with early stopping."""
    os.makedirs(models_dir, exist_ok=True)
    logger.info(f"Training NCF on {device} …")

    model = NCFModel(
        n_users=n_users,
        n_movies=n_movies,
        embedding_dim=config.get("embedding_dim", 64),
        hidden_layers=config.get("hidden_layers", [256, 128, 64]),
        dropout_rate=config.get("dropout_rate", 0.3),
    ).to(device)

    criterion = nn.MSELoss()
    lr = config.get("learning_rate", 0.001)
    if not 0.0001 <= lr <= 0.1:
        logger.warning(f"Learning rate {lr} out of bounds; clamping to [0.0001, 0.1]")
        lr = max(0.0001, min(0.1, lr))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = config.get("epochs", 30)
    patience = config.get("early_stopping_patience", 5)
    delta = config.get("early_stopping_delta", 0.001)

    best_val_rmse = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for user_ids, movie_ids, ratings in tqdm(
            train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False
        ):
            user_ids, movie_ids, ratings = (
                user_ids.to(device),
                movie_ids.to(device),
                ratings.to(device),
            )
            optimizer.zero_grad()
            preds = model(user_ids, movie_ids)
            loss = criterion(preds, ratings)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(ratings)

        train_loss /= len(train_loader.dataset)
        train_rmse = train_loss**0.5

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for user_ids, movie_ids, ratings in val_loader:
                user_ids, movie_ids, ratings = (
                    user_ids.to(device),
                    movie_ids.to(device),
                    ratings.to(device),
                )
                preds = model(user_ids, movie_ids)
                loss = criterion(preds, ratings)
                val_loss += loss.item() * len(ratings)
        val_loss /= len(val_loader.dataset)
        val_rmse = val_loss**0.5

        logger.info(
            f"Epoch {epoch}/{epochs} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}"
        )

        # Early stopping
        if val_rmse < best_val_rmse - delta:
            best_val_rmse = val_rmse
            patience_counter = 0
            model_path = os.path.join(models_dir, "ncf_model.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"Validation improved → model saved to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

    logger.info(f"Training complete. Best Val RMSE: {best_val_rmse:.4f}")
    return model, best_val_rmse


def load_ncf_model(
    model_path: str, n_users: int, n_movies: int, config: dict, device: str = "cpu"
):
    """Load a saved NCF model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = NCFModel(
        n_users=n_users,
        n_movies=n_movies,
        embedding_dim=config.get("embedding_dim", 64),
        hidden_layers=config.get("hidden_layers", [256, 128, 64]),
        dropout_rate=config.get("dropout_rate", 0.3),
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"NCF model loaded from {model_path}")
    return model
