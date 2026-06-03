"""One-off script to save dataset_meta.json from trained data."""
import json, os
from src.utils.config import load_config
from src.data.loader import load_ratings
from src.data.pytorch_dataset import RatingsDataset

cfg = load_config()
d = cfg["data"]
train_df, _, _ = load_ratings(d["data_dir"], d["test_ratio"], d["random_seed"])
ds = RatingsDataset(train_df)

meta = {
    "n_users": int(ds.n_users),
    "n_movies": int(ds.n_movies),
    "user_to_idx": {str(int(k)): int(v) for k, v in ds.user_to_idx.items()},
    "movie_to_idx": {str(int(k)): int(v) for k, v in ds.movie_to_idx.items()},
    "idx_to_user": {str(int(k)): int(v) for k, v in ds.idx_to_user.items()},
    "idx_to_movie": {str(int(k)): int(v) for k, v in ds.idx_to_movie.items()},
}
os.makedirs(d["models_dir"], exist_ok=True)
path = os.path.join(d["models_dir"], "dataset_meta.json")
with open(path, "w") as f:
    json.dump(meta, f)
print(f"Saved metadata: {ds.n_users} users, {ds.n_movies} movies → {path}")
