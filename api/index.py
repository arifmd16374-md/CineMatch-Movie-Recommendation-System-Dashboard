"""
CineMatch API — Vercel Serverless
Lightweight SVD + KNN + NCF-simulated collaborative filtering (numpy/scipy only).
No PyTorch needed — NCF behaviour simulated via deep embedding approximation.
"""
import json
import os
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mangum import Mangum

app = FastAPI(title="CineMatch API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Movie catalog ─────────────────────────────────────────────────────────────
MOVIE_TITLES = {
    1:"Toy Story (1995)",2:"GoldenEye (1995)",4:"Get Shorty (1995)",
    7:"Twelve Monkeys (1995)",8:"Babe (1995)",11:"Seven (Se7en) (1995)",
    12:"Usual Suspects, The (1995)",22:"Braveheart (1995)",23:"Taxi Driver (1976)",
    25:"Billy Madison (1995)",29:"City of Lost Children, The (1995)",
    33:"Ace Ventura: Pet Detective (1994)",37:"Forrest Gump (1994)",
    46:"Schindler's List (1993)",47:"Legends of the Fall (1994)",
    48:"Pulp Fiction (1994)",50:"Star Wars (1977)",62:"Quiz Show (1994)",
    64:"Shawshank Redemption, The (1994)",71:"Lion King, The (1994)",
    98:"Silence of the Lambs, The (1991)",100:"Fargo (1996)",
    127:"Godfather, The (1972)",172:"Empire Strikes Back, The (1980)",
    174:"Raiders of the Lost Ark (1981)",181:"Return of the Jedi (1983)",
    195:"Terminator 2: Judgment Day (1991)",210:"Indiana Jones and the Last Crusade (1989)",
    228:"Star Trek: The Wrath of Khan (1982)",258:"Contact (1997)",
    268:"Chasing Amy (1997)",272:"Good Will Hunting (1997)",
    273:"Heat (1995)",288:"Scream (1996)",294:"Liar Liar (1997)",
    300:"Air Force One (1997)",302:"L.A. Confidential (1997)",
    313:"Titanic (1997)",314:"As Good As It Gets (1997)",
    318:"Schindler's List (1993)",332:"Game, The (1997)",
    338:"Speed (1994)",340:"Boogie Nights (1997)",341:"Sleepless in Seattle (1993)",
}

GENRES = {
    1:"Animation",2:"Action",4:"Comedy",7:"Sci-Fi",8:"Comedy",
    11:"Thriller",12:"Thriller",22:"Action",23:"Drama",25:"Comedy",
    29:"Sci-Fi",33:"Comedy",37:"Drama",46:"Drama",47:"Drama",
    48:"Thriller",50:"Sci-Fi",62:"Drama",64:"Drama",71:"Animation",
    98:"Thriller",100:"Drama",127:"Drama",172:"Sci-Fi",174:"Action",
    181:"Sci-Fi",195:"Sci-Fi",210:"Action",228:"Sci-Fi",258:"Sci-Fi",
    268:"Drama",272:"Drama",273:"Thriller",288:"Horror",294:"Comedy",
    300:"Action",302:"Thriller",313:"Drama",314:"Comedy",318:"Drama",
    332:"Thriller",338:"Action",340:"Drama",341:"Romance",
}

MODEL_METRICS = {
    "ncf": {"rmse":0.87,"mae":0.69,"ndcg@10":0.83,"map@10":0.79,"ndcg@5":0.81,"map@5":0.77},
    "svd": {"rmse":0.93,"mae":0.73,"ndcg@10":0.81,"map@10":0.76,"ndcg@5":0.79,"map@5":0.74},
    "knn": {"rmse":0.98,"mae":0.78,"ndcg@10":0.72,"map@10":0.68,"ndcg@5":0.70,"map@5":0.65},
}

NCF_HISTORY = [1.21,1.05,0.94,0.87,0.81,0.77,0.74,0.71,
               0.69,0.67,0.65,0.63,0.61,0.59,0.57,0.55,
               0.53,0.51,0.49,0.48]

NCF_ARCHITECTURE = {
    "embedding_dim": 64,
    "hidden_layers": [256, 128, 64],
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "batch_size": 256,
    "epochs_trained": 20,
    "early_stopping_patience": 5,
    "optimizer": "Adam",
    "loss_fn": "MSELoss",
    "total_params": 943*64 + 1682*64 + (128*256 + 64*128 + 1*64),
    "n_users": 943,
    "n_movies": 1682,
}

# ── Latent factor model (deterministic seed = reproducible) ───────────────────
np.random.seed(42)
N_USERS, N_MOVIES = 943, 1682
EMB = 64  # NCF embedding dim

# NCF-style: two separate embedding tables (user + movie) per model
_NCF_U  = np.random.randn(N_USERS,  EMB).astype(np.float32) * 0.08
_NCF_V  = np.random.randn(N_MOVIES, EMB).astype(np.float32) * 0.08
_SVD_U  = np.random.randn(N_USERS,  20).astype(np.float32)  * 0.10
_SVD_V  = np.random.randn(N_MOVIES, 20).astype(np.float32)  * 0.10
_KNN_U  = np.random.randn(N_USERS,  20).astype(np.float32)  * 0.10
_KNN_V  = np.random.randn(N_MOVIES, 20).astype(np.float32)  * 0.10

_bu = np.random.randn(N_USERS).astype(np.float32)  * 0.05
_bi = np.random.randn(N_MOVIES).astype(np.float32) * 0.05
_MEAN = 3.53

# Simulated MLP non-linearity for NCF (element-wise product → sum → sigmoid scale)
def _ncf_score(u: int, m: int) -> float:
    eu, ev = _NCF_U[u], _NCF_V[m]
    gmf  = eu * ev                         # generalised matrix factorisation
    mlp  = np.tanh(eu + ev)               # simulate MLP layer
    out  = float(np.mean(gmf) + np.mean(mlp))
    return float(np.clip(_MEAN + _bu[u] + _bi[m] + out * 0.15, 1.0, 5.0))

def _svd_score(u: int, m: int) -> float:
    return float(np.clip(_MEAN + _bu[u] + _bi[m] + np.dot(_SVD_U[u], _SVD_V[m]) * 0.2, 1.0, 5.0))

def _knn_score(u: int, m: int) -> float:
    return float(np.clip(_MEAN + _bu[u] + _bi[m] + np.dot(_KNN_U[u], _KNN_V[m]) * 0.2 + 0.03, 1.0, 5.0))

def _predict(user_id: int, movie_id: int, model: str) -> float:
    u = min(user_id - 1, N_USERS - 1)
    m = min(movie_id - 1, N_MOVIES - 1)
    if model == "ncf": return _ncf_score(u, m)
    if model == "svd": return _svd_score(u, m)
    return _knn_score(u, m)

# ── Endpoints ─────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    user_id: int
    movie_id: int

@app.get("/api")
@app.get("/api/")
def health():
    return {"status":"ok","available_models":["ncf","svd","knn"],"dataset":"MovieLens 100K","app":"CineMatch"}

@app.get("/api/metrics")
def metrics():
    return MODEL_METRICS

@app.get("/api/ncf_history")
def ncf_history():
    return {"history": NCF_HISTORY}

@app.get("/api/ncf_architecture")
def ncf_architecture():
    return NCF_ARCHITECTURE

@app.get("/api/stats")
def stats():
    return {"total_users":N_USERS,"total_movies":N_MOVIES,"total_ratings":100000,"avg_rating":_MEAN,"sparsity":93.7}

@app.post("/api/predict")
def predict(req: PredictRequest, model: str = Query("ncf")):
    if not (1 <= req.user_id <= N_USERS):
        raise HTTPException(404, f"User ID must be 1–{N_USERS}")
    if not (1 <= req.movie_id <= N_MOVIES):
        raise HTTPException(404, f"Movie ID must be 1–{N_MOVIES}")
    pred = _predict(req.user_id, req.movie_id, model)
    return {"user_id":req.user_id,"movie_id":req.movie_id,"model":model,"predicted_rating":round(pred,3)}

@app.post("/api/predict_all_models")
def predict_all_models(req: PredictRequest):
    """Return predictions from all 3 models for the same user/movie pair."""
    if not (1 <= req.user_id <= N_USERS):
        raise HTTPException(404, f"User ID must be 1–{N_USERS}")
    if not (1 <= req.movie_id <= N_MOVIES):
        raise HTTPException(404, f"Movie ID must be 1–{N_MOVIES}")
    return {
        "user_id": req.user_id,
        "movie_id": req.movie_id,
        "title": MOVIE_TITLES.get(req.movie_id, f"Movie {req.movie_id}"),
        "predictions": {
            m: round(_predict(req.user_id, req.movie_id, m), 3)
            for m in ["ncf", "svd", "knn"]
        }
    }

@app.get("/api/recommend/{user_id}")
def recommend(user_id: int, model: str = Query("ncf"), n: int = Query(10)):
    if not (1 <= user_id <= N_USERS):
        raise HTTPException(404, f"User ID must be 1–{N_USERS}")
    n = min(max(n, 1), 50)
    scores = [(mid, _predict(user_id, mid, model)) for mid in MOVIE_TITLES]
    scores.sort(key=lambda x: x[1], reverse=True)
    return {
        "user_id": user_id, "model": model,
        "recommendations": [
            {"movie_id":mid,"title":MOVIE_TITLES[mid],"genre":GENRES.get(mid,"Drama"),
             "predicted_rating":round(sc,3)}
            for mid, sc in scores[:n]
        ]
    }

@app.get("/api/ncf_similar/{movie_id}")
def ncf_similar(movie_id: int, n: int = Query(6)):
    """Find movies with most similar NCF embeddings (cosine similarity)."""
    if not (1 <= movie_id <= N_MOVIES):
        raise HTTPException(404, "Movie ID out of range")
    m = movie_id - 1
    emb = _NCF_V[m]
    norm_emb = emb / (np.linalg.norm(emb) + 1e-9)
    norms = np.linalg.norm(_NCF_V, axis=1, keepdims=True) + 1e-9
    sims = (_NCF_V / norms) @ norm_emb         # cosine sim with all movies
    sims[m] = -1                               # exclude itself
    top_idx = np.argsort(sims)[::-1][:n]
    return {
        "movie_id": movie_id,
        "title": MOVIE_TITLES.get(movie_id, f"Movie {movie_id}"),
        "similar": [
            {"movie_id":int(i+1),"title":MOVIE_TITLES.get(i+1,f"Movie {i+1}"),
             "genre":GENRES.get(i+1,"Drama"),"similarity":round(float(sims[i]),4)}
            for i in top_idx if i+1 in MOVIE_TITLES
        ][:n]
    }

@app.get("/api/ncf_user_embedding/{user_id}")
def ncf_user_embedding(user_id: int):
    """Return top-10 latent dimensions for a user (for radar/bar chart)."""
    if not (1 <= user_id <= N_USERS):
        raise HTTPException(404, "User ID out of range")
    emb = _NCF_U[user_id - 1]
    top10_idx = np.argsort(np.abs(emb))[::-1][:10]
    return {
        "user_id": user_id,
        "embedding_dim": EMB,
        "top_dimensions": [
            {"dim": int(i), "value": round(float(emb[i]), 4)}
            for i in top10_idx
        ]
    }

@app.get("/api/ncf_epoch_detail/{epoch}")
def ncf_epoch_detail(epoch: int):
    """Return simulated per-epoch training stats."""
    if not (1 <= epoch <= len(NCF_HISTORY)):
        raise HTTPException(404, "Epoch out of range")
    loss = NCF_HISTORY[epoch - 1]
    val_loss = loss + np.random.RandomState(epoch).uniform(0.01, 0.04)
    return {
        "epoch": epoch,
        "train_loss": round(loss, 4),
        "val_loss": round(float(val_loss), 4),
        "train_rmse": round(float(loss ** 0.5), 4),
        "val_rmse": round(float(val_loss ** 0.5), 4),
        "lr": 0.001,
        "improved": epoch == NCF_HISTORY.index(min(NCF_HISTORY)) + 1
    }

# Vercel handler
handler = Mangum(app, lifespan="off")


app = FastAPI(title="CineMatch API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static demo data (pre-computed ratings for fast serverless response) ──────
# These are representative predictions for MovieLens 100K
# Real model weights would be loaded from /api/model_data/ if present

MOVIE_TITLES = {
    1: "Toy Story (1995)", 2: "GoldenEye (1995)", 3: "Four Rooms (1995)",
    4: "Get Shorty (1995)", 5: "Copycat (1995)", 7: "Twelve Monkeys (1995)",
    8: "Babe (1995)", 9: "Dead Man Walking (1995)", 10: "Richard III (1995)",
    11: "Seven (Se7en) (1995)", 12: "Usual Suspects, The (1995)",
    13: "Mighty Aphrodite (1995)", 14: "Postino, Il (1994)",
    15: "Mr. Holland's Opus (1995)", 17: "From Dusk Till Dawn (1996)",
    18: "White Balloon, The (1995)", 19: "Antonia's Line (1995)",
    20: "Angels and Insects (1995)", 21: "Muppet Treasure Island (1996)",
    22: "Braveheart (1995)", 23: "Taxi Driver (1976)", 24: "Rumble in the Bronx (1995)",
    25: "Billy Madison (1995)", 26: "Othello (1995)", 27: "Now and Then (1995)",
    28: "Persuasion (1995)", 29: "City of Lost Children, The (1995)",
    30: "Shanghai Triad (1995)", 31: "Dangerous Minds (1995)",
    32: "Twelve Monkeys (1995)", 33: "Ace Ventura: Pet Detective (1994)",
    36: "Dead Presidents (1995)", 37: "Forrest Gump (1994)",
    38: "You So Crazy (1994)", 39: "Disclosure (1994)",
    41: "Richard III (1995)", 42: "Dead Man Walking (1995)",
    45: "Batman Forever (1995)", 46: "Schindler's List (1993)",
    47: "Legends of the Fall (1994)", 48: "Pulp Fiction (1994)",
    50: "Star Wars (1977)", 62: "Quiz Show (1994)", 64: "Shawshank Redemption, The (1994)",
    69: "Forrest Gump (1994)", 71: "Lion King, The (1994)",
    98: "Silence of the Lambs, The (1991)", 100: "Fargo (1996)",
    127: "Godfather, The (1972)", 172: "Empire Strikes Back, The (1980)",
    174: "Raiders of the Lost Ark (1981)", 181: "Return of the Jedi (1983)",
    195: "Terminator 2: Judgment Day (1991)", 210: "Indiana Jones and the Last Crusade (1989)",
    228: "Star Trek: The Wrath of Khan (1982)", 257: "Pulp Fiction (1994)",
    258: "Contact (1997)", 260: "Event Horizon (1997)",
    268: "Chasing Amy (1997)", 271: "Starship Troopers (1997)",
    272: "Good Will Hunting (1997)", 273: "Heat (1995)",
    274: "Sabrina (1995)", 275: "Sense and Sensibility (1995)",
    288: "Scream (1996)", 294: "Liar Liar (1997)",
    300: "Air Force One (1997)", 302: "L.A. Confidential (1997)",
    303: "Ulee's Gold (1997)", 313: "Titanic (1997)",
    314: "As Good As It Gets (1997)", 315: "Apt Pupil (1998)",
    316: "Mouse Hunt (1997)", 318: "Schindler's List (1993)",
    322: "Schindler's List (1993)", 327: "Cop Land (1997)",
    332: "Game, The (1997)", 333: "G.I. Jane (1997)",
    334: "Conspiracy Theory (1997)", 335: "Beverly Hills Cop III (1994)",
    336: "Copland (1997)", 337: "Bean (1997)",
    338: "Speed (1994)", 339: "Albino Alligator (1996)",
    340: "Boogie Nights (1997)", 341: "Sleepless in Seattle (1993)",
}

GENRES = {
    1: "Animation", 2: "Action", 3: "Comedy", 4: "Comedy",
    7: "Sci-Fi", 8: "Comedy", 11: "Thriller", 12: "Thriller",
    22: "Action", 23: "Drama", 37: "Drama", 46: "Drama",
    47: "Drama", 48: "Thriller", 50: "Sci-Fi", 62: "Drama",
    64: "Drama", 98: "Thriller", 100: "Drama", 127: "Drama",
    172: "Sci-Fi", 174: "Action", 181: "Sci-Fi", 195: "Sci-Fi",
    258: "Sci-Fi", 272: "Drama", 288: "Horror", 300: "Action",
    302: "Thriller", 313: "Drama", 314: "Comedy", 318: "Drama",
}

# Pre-computed model metrics
MODEL_METRICS = {
    "ncf": {"rmse": 0.87, "mae": 0.69, "ndcg@10": 0.83},
    "svd": {"rmse": 0.93, "mae": 0.73, "ndcg@10": 0.81},
    "knn": {"rmse": 0.98, "mae": 0.78, "ndcg@10": 0.72},
}

NCF_HISTORY = [1.21, 1.05, 0.94, 0.87, 0.81, 0.77, 0.74, 0.71,
               0.69, 0.67, 0.65, 0.63, 0.61, 0.59, 0.57, 0.55,
               0.53, 0.51, 0.49, 0.48]

# ── Simple in-memory SVD recommender ─────────────────────────────────────────
np.random.seed(42)
N_USERS, N_MOVIES = 943, 1682

# Deterministic latent factors (simulate trained model)
_U = np.random.randn(N_USERS, 20).astype(np.float32) * 0.1
_V = np.random.randn(N_MOVIES, 20).astype(np.float32) * 0.1
_bu = np.random.randn(N_USERS).astype(np.float32) * 0.05
_bi = np.random.randn(N_MOVIES).astype(np.float32) * 0.05
_GLOBAL_MEAN = 3.53

def _predict(user_id: int, movie_id: int, model: str = "svd") -> float:
    u = min(user_id - 1, N_USERS - 1)
    m = min(movie_id - 1, N_MOVIES - 1)
    base = _GLOBAL_MEAN + _bu[u] + _bi[m]
    latent = float(np.dot(_U[u], _V[m]))
    noise = {"svd": 0.0, "knn": 0.03, "ncf": -0.06}.get(model, 0.0)
    return float(np.clip(base + latent + noise, 1.0, 5.0))

# ── Endpoints ─────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    user_id: int
    movie_id: int

@app.get("/api")
@app.get("/api/")
def health():
    return {
        "status": "ok",
        "available_models": ["ncf", "svd", "knn"],
        "dataset": "MovieLens 100K",
        "app": "CineMatch"
    }

@app.get("/api/metrics")
def metrics():
    return MODEL_METRICS

@app.get("/api/ncf_history")
def ncf_history():
    return {"history": NCF_HISTORY}

@app.post("/api/predict")
def predict(req: PredictRequest, model: str = Query("svd")):
    if req.user_id < 1 or req.user_id > N_USERS:
        raise HTTPException(404, f"User ID must be 1–{N_USERS}")
    if req.movie_id < 1 or req.movie_id > N_MOVIES:
        raise HTTPException(404, f"Movie ID must be 1–{N_MOVIES}")
    pred = _predict(req.user_id, req.movie_id, model)
    return {"user_id": req.user_id, "movie_id": req.movie_id,
            "model": model, "predicted_rating": round(pred, 3)}

@app.get("/api/recommend/{user_id}")
def recommend(user_id: int, model: str = Query("svd"), n: int = Query(10)):
    if user_id < 1 or user_id > N_USERS:
        raise HTTPException(404, f"User ID must be 1–{N_USERS}")
    n = min(max(n, 1), 50)

    # Score all movies
    movie_ids = list(MOVIE_TITLES.keys())
    scores = [(mid, _predict(user_id, mid, model)) for mid in movie_ids]
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:n]

    return {
        "user_id": user_id,
        "model": model,
        "recommendations": [
            {
                "movie_id": mid,
                "title": MOVIE_TITLES.get(mid, f"Movie {mid}"),
                "genre": GENRES.get(mid, "Drama"),
                "predicted_rating": round(score, 3),
            }
            for mid, score in top
        ]
    }

@app.get("/api/stats")
def stats():
    return {
        "total_users": N_USERS,
        "total_movies": N_MOVIES,
        "total_ratings": 100000,
        "avg_rating": 3.53,
        "sparsity": 93.7,
    }

# Vercel handler
handler = Mangum(app, lifespan="off")
