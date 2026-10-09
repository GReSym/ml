"""Settings of the ML service.

config.example.json (in git) is the source of truth for structure and non-secret defaults.
config.json (git-ignored) is merged on top of it, so it only needs the secrets (DB and RabbitMQ credentials).
GRESYM_ML_CONFIG can point to another config.json.
"""

import json
import os
from functools import cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel

ML_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_FILE = ML_ROOT / "config.example.json"
CONFIG_FILE = ML_ROOT / "config.json"

PLACEHOLDERS = {"", "xxx", "###"}


class DatabaseSettings(BaseModel):
    host: str
    port: int
    database: str
    user: str
    password: str


class MilvusSettings(BaseModel):
    uri: str
    token: str = ""
    collection: str


class RabbitMqSettings(BaseModel):
    host: str
    port: int
    virtual_host: str
    user: str
    password: str
    exchange: str
    dead_letter_exchange: str
    queue: str
    dead_letter_queue: str
    routing_key: str
    delivery_limit: int
    prefetch: int


class ModelSettings(BaseModel):
    name: str
    path: str
    dimension: int
    max_seq_length: int
    batch_size: int
    threads: int | None = None

    @property
    def resolved_path(self) -> Path:
        path = Path(self.path)
        return path if path.is_absolute() else ML_ROOT / path


class ApiSettings(BaseModel):
    host: str
    port: int


class Settings(BaseModel):
    database: DatabaseSettings
    milvus: MilvusSettings
    rabbitmq: RabbitMqSettings
    model: ModelSettings
    api: ApiSettings


class ConfigError(Exception):
    pass


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"{path} is not valid JSON: {e}") from e


def _check_secrets(settings: Settings, config_path: Path) -> None:
    missing = [
        name
        for name, value in {
            "database.user": settings.database.user,
            "database.password": settings.database.password,
            "rabbitmq.user": settings.rabbitmq.user,
            "rabbitmq.password": settings.rabbitmq.password,
        }.items()
        if value in PLACEHOLDERS
    ]
    if missing:
        raise ConfigError(f"Fill in {', '.join(missing)} in {config_path}")


@cache
def get_settings() -> Settings:
    config_path = Path(os.environ.get("GRESYM_ML_CONFIG", CONFIG_FILE))
    if not config_path.exists():
        raise ConfigError(
            f"{config_path} not found. Create it with the secrets from {EXAMPLE_FILE.name} "
            "(only the values that differ are needed)"
        )

    data = _merge(_read_json(EXAMPLE_FILE), _read_json(config_path))
    settings = Settings.model_validate(data)
    _check_secrets(settings, config_path)
    return settings
