"""Centralised structured logging with file rotation."""
import logging
import os
from logging.handlers import RotatingFileHandler
import yaml


def get_logger(name: str, config_path: str = "config.yaml") -> logging.Logger:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)["logging"]

    log_dir = cfg.get("log_dir", "logs/")
    os.makedirs(log_dir, exist_ok=True)

    level = getattr(logging, cfg.get("level", "INFO"))
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Console
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # Rotating file
        fh = RotatingFileHandler(
            os.path.join(log_dir, "app.log"),
            maxBytes=cfg.get("max_bytes", 10_485_760),
            backupCount=cfg.get("backup_count", 5),
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
