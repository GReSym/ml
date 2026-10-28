"""HTTP API of the ML service, called synchronously by the ASP.NET backend.

Doesn't load the embedding model: recommendations only use vectors already stored in Milvus.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from ml import recommend as rec
from ml.catalogue import CatalogueCache
from ml.config import get_settings
from ml.db import GameRepository
from ml.staleness import Reason, filter_items, find_stale
from ml.vectors import GameVectorStore


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class RatingDto(CamelModel):
    game_id: int
    rating: int | None = Field(default=None, ge=rec.RATING_MIN, le=rec.RATING_MAX)
    recommend: bool | None = None


class RecommendationsRequest(CamelModel):
    ratings: list[RatingDto] = Field(max_length=5000)
    limit: int = Field(default=20, ge=1, le=100)
    # 0 = pure similarity (the most niche games), 100 = strongly favours popular games; None = the default
    mainstream: int | None = Field(default=None, ge=0, le=100)


class ScoredGameDto(CamelModel):
    game_id: int
    score: float


class RecommendationsResponse(CamelModel):
    items: list[ScoredGameDto]
    used_ratings: int  # liked games with a vector (dislikes don't influence the kNN algorithm)
    mainstream: int


class SimilarResponse(CamelModel):
    items: list[ScoredGameDto]


class StaleGameDto(CamelModel):
    game_id: int
    score: float
    reasons: list[Reason]
    vectorized_at: int | None  # unix seconds
    details: dict[str, float | int | None]


class StaleGamesResponse(CamelModel):
    collection: str
    text_version: int
    vectors: int
    games: int
    counts: dict[Reason, int]  # over all stale games, before filtering
    total: int  # matching the filter
    items: list[StaleGameDto]


class HealthResponse(CamelModel):
    status: str
    collection: str
    vectors: int


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    store = GameVectorStore(settings.milvus, settings.model)
    store.ensure_collection()
    app.state.store = store
    # Loaded on the first recommendation request, so the API starts even if MariaDB is briefly unavailable
    app.state.catalogue = CatalogueCache(settings, store)
    yield
    store.close()


app = FastAPI(title="GReSym ML", lifespan=lifespan)


def _store() -> GameVectorStore:
    return app.state.store


def _scored(items: list[tuple[int, float]]) -> list[ScoredGameDto]:
    return [ScoredGameDto(game_id=game_id, score=score) for game_id, score in items]


# Sync endpoints: FastAPI runs them in a thread pool, so the blocking Milvus client doesn't block the event loop.
@app.get("/health", response_model=HealthResponse, response_model_by_alias=True)
def health() -> HealthResponse:
    store = _store()
    return HealthResponse(status="ok", collection=store.collection, vectors=store.count())


@app.post("/recommendations", response_model=RecommendationsResponse, response_model_by_alias=True)
def recommendations(request: RecommendationsRequest) -> RecommendationsResponse:
    ratings = [rec.Rating(r.game_id, r.rating, r.recommend) for r in request.ratings]
    mainstream = rec.DEFAULT_MAINSTREAM if request.mainstream is None else request.mainstream
    result = rec.recommend(_store(), app.state.catalogue.get(), ratings, request.limit, mainstream)
    return RecommendationsResponse(items=_scored(result.items), used_ratings=result.used_ratings, mainstream=mainstream)


@app.get("/games/{game_id}/similar", response_model=SimilarResponse, response_model_by_alias=True)
def similar_games(game_id: int, limit: int = Query(default=10, ge=1, le=100)) -> SimilarResponse:
    items = rec.similar(_store(), game_id, limit)
    if items is None:
        raise HTTPException(status_code=404, detail=f"Game {game_id} has no vector")
    return SimilarResponse(items=_scored(items))


@app.get("/vectorization/stale", response_model=StaleGamesResponse, response_model_by_alias=True)
def stale_games(
    reason: list[Reason] | None = Query(default=None, description="only games with any of these reasons"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> StaleGamesResponse:
    """Games whose vector is out of date, best candidates first. Takes a few seconds (scans all games)."""
    games = GameRepository(get_settings().database)  # not thread-safe, so one per request
    try:
        report = find_stale(get_settings(), _store(), games)
    finally:
        games.close()
    items = filter_items(report.items, reason)
    return StaleGamesResponse(
        collection=report.collection,
        text_version=report.text_version,
        vectors=report.vectors,
        games=report.games,
        counts=report.counts(),
        total=len(items),
        items=[
            StaleGameDto(game_id=i.game_id, score=round(i.score, 3), reasons=i.reasons,
                         vectorized_at=i.vectorized_at, details=i.details)
            for i in items[offset : offset + limit]
        ],
    )
