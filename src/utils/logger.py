"""Centralised structured logging with file rotation."""
import logging
import os
from logging.handlers import RotatingFileHandler


def get_logger(name: str, config_path: str = "config.yaml") -> logging.Logger:
    # Safe defaults — don't crash if config.yaml missing (cloud deploy)
    log_dir  = "logs/"
    level    = logging.INFO
    max_bytes    = 10_485_760
    backup_count = 5

    try:
        import yaml
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg_all = yaml.safe_load(f) or {}
            cfg = cfg_all.get("logging", {})
            log_dir      = cfg.get("log_dir", log_dir)
            backup_count = cfg.get("backup_count", backup_count)
            max_bytes    = cfg.get("max_bytes", max_bytes)
            level        = getattr(logging, cfg.get("level", "INFO"), logging.INFO)
    except Exception:
        pass  # fall back to defaults silently

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler always
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # Rotating file handler — skip if logs dir can't be created (read-only fs)
        try:
            os.makedirs(log_dir, exist_ok=True)
            fh = RotatingFileHandler(
                os.path.join(log_dir, "app.log"),
                maxBytes=max_bytes,
                backupCount=backup_count,
            )
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception:
            pass

    return logger
