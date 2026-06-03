# 🎬 Movie Recommendation System

A production-grade recommendation system using Collaborative Filtering (SVD, KNN) and Neural Collaborative Filtering (NCF) served via FastAPI with a Streamlit UI.

## Project Structure

```
.
├── src/
│   ├── data/
│   │   ├── loader.py          # MovieLens download & preprocessing
│   │   ├── eda.py             # Exploratory data analysis
│   │   └── pytorch_dataset.py # Encoded PyTorch Dataset
│   ├── models/
│   │   ├── surprise_models.py # SVD & KNN with grid search
│   │   └── ncf_model.py       # PyTorch NCF model
│   ├── evaluation/
│   │   ├── metrics.py         # RMSE, MAE, NDCG@K, MAP@K
│   │   └── evaluate.py        # Evaluation pipeline
│   ├── api/
│   │   ├── app.py             # FastAPI application
│   │   └── models_store.py    # Model loading & serving
│   └── train_pipeline.py      # End-to-end training script
├── app_streamlit.py            # Streamlit web UI
├── run_api.py                  # API launcher
├── config.yaml                 # All configuration
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Quick Start

### 1. Create and activate a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train all models
This downloads MovieLens 100K, runs EDA, trains SVD/KNN/NCF and saves models.
```bash
python -m src.train_pipeline
```
*Training time: ~5-20 min depending on hardware (grid search is the bottleneck).*

### 4. Start the FastAPI server
```bash
python run_api.py
# OR
uvicorn src.api.app:app --port 8000
```

### 5. Launch the Streamlit UI
In a new terminal (with .venv active):
```bash
streamlit run app_streamlit.py
```
Open http://localhost:8501 in your browser.

### API Documentation
Interactive docs available at: http://localhost:8000/docs

## Key Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check & available models |
| POST | `/predict?model=svd` | Predict rating for user/movie pair |
| GET | `/recommend/{user_id}?model=ncf&n=10` | Top-N recommendations |

## Performance Targets (MovieLens 100K)

| Model | RMSE | MAE |
|-------|------|-----|
| KNN   | ~0.98 | ~0.78 |
| SVD   | ~0.93 | ~0.73 |
| NCF   | ~0.87 | ~0.68 |

## Docker

```bash
# Build and run both API and UI
docker-compose up --build

# API: http://localhost:8000
# UI:  http://localhost:8501
```
