"""Settings of the ML service.

config.example.json (in git) is the source of truth for structure and non-secret defaults.
config.json (git-ignored) is merged on top of it, so it only needs the secrets (DB and RabbitMQ credentials).
GRESYM_ML_CONFIG can point to another config.json.
Environment variables GRESYM_ML__<SECTION>__<KEY> (e.g. GRESYM_ML__DATABASE__HOST) are merged last. They're how
the Docker containers are configured; config.json is optional when they provide the secrets.
"""

import json
import os
from functools import cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

ML_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_FILE = ML_ROOT / "config.example.json"
CONFIG_FILE = ML_ROOT / "config.json"

PLACEHOLDERS = {"", "xxx", "###"}
ENV_PREFIX = "GRESYM_ML__"


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
    text_collection: str  # BM25 index of titles and descriptions (ml.text_index)


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
    delivery_limit: int  # also used for the evaluation and recommendation results queues
    prefetch: int
    # Offline evaluation (ml.evaluation): API → evaluation_queue → worker → evaluation_results_queue → API
    evaluation_queue: str
    evaluation_dead_letter_queue: str
    evaluation_routing_key: str
    evaluation_delivery_limit: int
    evaluation_results_queue: str
    evaluation_results_dead_letter_queue: str
    evaluation_results_routing_key: str
    # Recommendations (ml.personal): API → recommendation_queue → worker → recommendation_results_queue → API
    recommendation_queue: str
    recommendation_dead_letter_queue: str
    recommendation_routing_key: str
    recommendation_delivery_limit: int
    recommendation_results_queue: str
    recommendation_results_dead_letter_queue: str
    recommendation_results_routing_key: str


class ModelSettings(BaseModel):
    name: str
    backend: Literal["onnx", "torch"] = "onnx"  # onnx: FastEmbed, CPU only, ~1.55x faster; torch: Sentence Transformers (GPU)
    path: str  # torch checkpoint
    onnx_path: str  # ONNX export (onnx/model.onnx + tokenizer files)
    dimension: int
    max_seq_length: int
    batch_size: int
    threads: int | None = None
    device: str = "cpu"  # "cpu", "cuda" (also ROCm: torch exposes AMD GPUs as cuda) or "auto". onnx: cpu only
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"  # torch only. fp32 on CPU: bf16 is ~4x slower there

    @property
    def resolved_path(self) -> Path:
        """Model directory of the selected backend."""
        path = Path(self.onnx_path if self.backend == "onnx" else self.path)
        return path if path.is_absolute() else ML_ROOT / path


class ReviewSettings(BaseModel):
    max_selected: int = Field(ge=1, le=64)  # reviews embedded per game (also the Milvus array capacity limit)
    min_chars: int  # shorter reviews ("10/10", "best game") say nothing about the game
    max_chars: int  # each review is cut to this many characters
    weight_max: float = Field(ge=0)  # α for a game with max_selected reviews


class StalenessSettings(BaseModel):
    min_overlap: float  # Jaccard of stored vs. current review selection below this → reviews changed
    min_selected: int  # ...but only if either selection has at least this many reviews
    sentiment_shift: float  # |Δ positive share| on Steam at least this → sentiment shift
    sentiment_min_total: int  # ...only for games with at least this many Steam reviews
    growth_min_new: int  # Steam review count grew by at least this many
    growth_ratio: float  # ...and by at least this share of the old count


class SteamSettings(BaseModel):
    """Only used by scripts/collect_steam_users.py, which checks the secrets itself."""
    api_key: str = ""  # Steam Web API key (GetPlayerSummaries, GetOwnedGames)
    anonymization_salt: str = ""  # HMAC key for steam_id_hash; keep it stable, or re-collected users get new hashes
    request_delay_seconds: float = 1.0  # between Steam Web API / store requests
    community_delay_seconds: float = 3.0  # between steamcommunity.com page requests (stricter rate limit)
    min_played_games: int = 20  # a user needs MORE than this many played games that are in our catalogue
    min_reviews: int = 5  # ...and at least this many reviews in the profile
    min_playtime_minutes: int = 60  # a game counts as played from this playtime on
    friends_per_user: int = 10  # friends sampled from each visited user (snowball over the friend graph)
    max_depth: int = 4  # hops from the seed profiles


class SearchSettings(BaseModel):
    """Game search (ml.search): weighted RRF over the BM25 title/text lists and, for description queries, the dense list."""
    dense: bool  # false: description queries are BM25-only and the API never loads the model
    candidates: int = Field(ge=1, le=1000)  # hits taken from each list before fusion
    rrf_k: int = Field(ge=1)
    title_weight: float = Field(ge=0)
    text_weight: float = Field(ge=0)
    dense_weight: float = Field(ge=0)


class ApiSettings(BaseModel):
    host: str
    port: int


class Settings(BaseModel):
    database: DatabaseSettings
    milvus: MilvusSettings
    rabbitmq: RabbitMqSettings
    model: ModelSettings
    reviews: ReviewSettings
    staleness: StalenessSettings
    search: SearchSettings
    api: ApiSettings
    steam: SteamSettings = SteamSettings()


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


def _env_overrides() -> dict[str, Any]:
    """GRESYM_ML__RABBITMQ__HOST=x → {"rabbitmq": {"host": "x"}}. Values stay strings, pydantic converts them."""
    result: dict[str, Any] = {}
    for name, value in os.environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        path = [part.lower() for part in name.removeprefix(ENV_PREFIX).split("__") if part]
        if not path:
            continue
        node = result
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
    return result


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
        raise ConfigError(f"Fill in {', '.join(missing)} in {config_path} (or {ENV_PREFIX}* environment variables)")


@cache
def get_settings() -> Settings:
    config_path = Path(os.environ.get("GRESYM_ML_CONFIG", CONFIG_FILE))
    env = _env_overrides()
    if not config_path.exists() and not env:
        raise ConfigError(
            f"{config_path} not found. Create it with the secrets from {EXAMPLE_FILE.name} "
            "(only the values that differ are needed)"
        )

    data = _read_json(EXAMPLE_FILE)
    if config_path.exists():
        data = _merge(data, _read_json(config_path))
    data = _merge(data, env)
    settings = Settings.model_validate(data)
    _check_secrets(settings, config_path)
    return settings
