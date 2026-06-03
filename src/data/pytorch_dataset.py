"""PyTorch Dataset for NCF with user/item encoding."""
import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RatingsDataset(Dataset):
    """PyTorch Dataset with encoded user/movie IDs."""

    def __init__(self, df: pd.DataFrame):
        # Create 0-based contiguous encodings
        unique_users = sorted(df["user_id"].unique())
        unique_movies = sorted(df["movie_id"].unique())

        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.movie_to_idx = {m: i for i, m in enumerate(unique_movies)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.idx_to_movie = {i: m for m, i in self.movie_to_idx.items()}

        # Validate all IDs are encoded
        if not all(df["user_id"].isin(self.user_to_idx)):
            raise ValueError("Some user IDs cannot be encoded.")
        if not all(df["movie_id"].isin(self.movie_to_idx)):
            raise ValueError("Some movie IDs cannot be encoded.")

        self.user_ids = torch.tensor(
            df["user_id"].map(self.user_to_idx).values, dtype=torch.long
        )
        self.movie_ids = torch.tensor(
            df["movie_id"].map(self.movie_to_idx).values, dtype=torch.long
        )
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

        self.n_users = len(self.user_to_idx)
        self.n_movies = len(self.movie_to_idx)

        logger.info(
            f"Encoded dataset — {self.n_users} users, {self.n_movies} movies, "
            f"{len(df)} samples"
        )

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.user_ids[idx], self.movie_ids[idx], self.ratings[idx]

    def decode_user(self, encoded_id: int) -> int:
        """Map encoded user ID back to original."""
        if encoded_id not in self.idx_to_user:
            raise ValueError(f"Invalid encoded user ID: {encoded_id}")
        return self.idx_to_user[encoded_id]

    def decode_movie(self, encoded_id: int) -> int:
        """Map encoded movie ID back to original."""
        if encoded_id not in self.idx_to_movie:
            raise ValueError(f"Invalid encoded movie ID: {encoded_id}")
        return self.idx_to_movie[encoded_id]

    def encode_user(self, user_id: int) -> int:
        """Map original user ID to encoded."""
        if user_id not in self.user_to_idx:
            raise ValueError(f"Unknown user ID: {user_id}")
        return self.user_to_idx[user_id]

    def encode_movie(self, movie_id: int) -> int:
        """Map original movie ID to encoded."""
        if movie_id not in self.movie_to_idx:
            raise ValueError(f"Unknown movie ID: {movie_id}")
        return self.movie_to_idx[movie_id]


def test_round_trip(dataset: RatingsDataset):
    """Verify encoding → decoding produces original IDs."""
    for orig_u in dataset.user_to_idx.keys():
        enc = dataset.encode_user(orig_u)
        dec = dataset.decode_user(enc)
        assert orig_u == dec, f"Round-trip failed for user {orig_u}"

    for orig_m in dataset.movie_to_idx.keys():
        enc = dataset.encode_movie(orig_m)
        dec = dataset.decode_movie(enc)
        assert orig_m == dec, f"Round-trip failed for movie {orig_m}"

    logger.info("Round-trip property test passed for all IDs.")
