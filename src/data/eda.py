"""Exploratory Data Analysis for MovieLens ratings."""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_eda(df: pd.DataFrame, output_dir: str = "outputs/eda/"):
    """Generate EDA visualizations and statistics."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Running EDA …")

    stats = {
        "sparsity": round(
            1 - len(df) / (df["user_id"].nunique() * df["movie_id"].nunique()), 4
        ),
        "rating_mean": round(float(df["rating"].mean()), 4),
        "rating_median": float(df["rating"].median()),
        "rating_std": round(float(df["rating"].std()), 4),
    }
    logger.info(f"Dataset sparsity: {stats['sparsity'] * 100:.2f}%")
    logger.info(
        f"Rating stats — mean: {stats['rating_mean']}, "
        f"median: {stats['rating_median']}, std: {stats['rating_std']}"
    )

    # Rating distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    df["rating"].value_counts().sort_index().plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Rating Distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "rating_distribution.png"), dpi=100)
    plt.close(fig)

    # User activity
    user_counts = df.groupby("user_id").size()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(user_counts, bins=50, ax=ax, kde=False, color="coral")
    ax.set_title("User Activity Distribution")
    ax.set_xlabel("Number of Ratings per User")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "user_activity.png"), dpi=100)
    plt.close(fig)

    # Movie popularity
    movie_counts = df.groupby("movie_id").size()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(movie_counts, bins=50, ax=ax, kde=False, color="mediumseagreen")
    ax.set_title("Movie Popularity Distribution")
    ax.set_xlabel("Number of Ratings per Movie")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "movie_popularity.png"), dpi=100)
    plt.close(fig)

    # Top/bottom users
    top_users = user_counts.nlargest(5)
    bottom_users = user_counts.nsmallest(5)
    logger.info(f"Top 5 users (ratings): {top_users.to_dict()}")
    logger.info(f"Bottom 5 users (ratings): {bottom_users.to_dict()}")

    # Top/bottom movies
    top_movies = movie_counts.nlargest(5)
    bottom_movies = movie_counts.nsmallest(5)
    logger.info(f"Top 5 movies (ratings): {top_movies.to_dict()}")
    logger.info(f"Bottom 5 movies (ratings): {bottom_movies.to_dict()}")

    logger.info(f"EDA complete. Visualizations saved to {output_dir}")
    return stats
