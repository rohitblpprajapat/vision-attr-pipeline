"""
Centralized structured logging for the Vision Attribute Pipeline.
Uses Python's logging module with optional JSON formatter.
"""
import logging
import logging.handlers
import os
import sys
import json
import time
import traceback
from pathlib import Path
from typing import Optional
import yaml


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "ms": int(record.msecs),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_obj["exc"] = self.formatException(record.exc_info)
        # Attach any extra fields passed via the `extra` kwarg
        for key, val in record.__dict__.items():
            if key.startswith("_") and not key.startswith("__"):
                log_obj[key[1:]] = val
        return json.dumps(log_obj, default=str)


def get_logger(name: str, cfg: Optional[dict] = None) -> logging.Logger:
    """
    Returns a configured logger for the given module name.

    Usage:
        log = get_logger(__name__, cfg=config["logging"])
        log.info("Worker started", extra={"_camera_id": "cam01"})
    """
    if cfg is None:
        cfg = {
            "level": os.environ.get("LOG_LEVEL", "INFO"),
            "format": os.environ.get("LOG_FORMAT", "text"),
            "log_dir": os.environ.get("LOG_DIR", "/tmp/vap_logs"),
            "rotate_mb": 50,
            "backup_count": 5,
            "modules": {},
        }

    logger = logging.getLogger(name)

    # Honour per-module level overrides
    module_levels = cfg.get("modules", {})
    level_str = module_levels.get(name, cfg.get("level", "INFO"))
    logger.setLevel(getattr(logging, level_str.upper(), logging.INFO))

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    fmt = cfg.get("format", "text")
    if fmt == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating file handler
    log_dir = Path(cfg.get("log_dir", "/tmp/vap_logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name.replace('.', '_')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=cfg.get("rotate_mb", 50) * 1024 * 1024,
        backupCount=cfg.get("backup_count", 5),
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Loads YAML config, resolving ${ENV_VAR:-default} substitutions.
    """
    import re

    def _resolve_env(value: str) -> str:
        pattern = re.compile(r"\$\{([^}^{]+)\}")
        matches = pattern.findall(value)
        for m in matches:
            if ":-" in m:
                var, default = m.split(":-", 1)
            else:
                var, default = m, ""
            env_val = os.environ.get(var, default)
            value = value.replace(f"${{{m}}}", env_val)
        return value

    def _resolve_dict(d):
        if isinstance(d, dict):
            return {k: _resolve_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [_resolve_dict(i) for i in d]
        elif isinstance(d, str):
            return _resolve_env(d)
        return d

    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return _resolve_dict(raw)


class PerfTimer:
    """Lightweight context-manager timer for performance logging."""

    def __init__(self, label: str, logger: Optional[logging.Logger] = None):
        self.label = label
        self.logger = logger
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        if self.logger:
            self.logger.debug(
                f"{self.label} took {self.elapsed_ms:.1f}ms",
                extra={"_perf_label": self.label, "_elapsed_ms": self.elapsed_ms},
            )
