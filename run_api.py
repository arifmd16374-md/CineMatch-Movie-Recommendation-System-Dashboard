"""Launch the FastAPI server."""
import uvicorn
from src.utils.config import load_config

if __name__ == "__main__":
    cfg = load_config()
    api_cfg = cfg["api"]
    uvicorn.run(
        "src.api.app:app",
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 8000),
        reload=False,
        log_level="info",
    )
