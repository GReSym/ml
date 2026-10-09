"""HTTP API of the ML service, called synchronously by the ASP.NET backend.

Doesn't load the embedding model: recommendations only use vectors already stored in Milvus.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from ml import recommend as rec
from ml.config import get_settings
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


class ScoredGameDto(CamelModel):
    game_id: int
    score: float


class RecommendationsResponse(CamelModel):
    items: list[ScoredGameDto]
    used_ratings: int


class SimilarResponse(CamelModel):
    items: list[ScoredGameDto]


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
    result = rec.recommend(_store(), ratings, request.limit)
    return RecommendationsResponse(items=_scored(result.items), used_ratings=result.used_ratings)


@app.get("/games/{game_id}/similar", response_model=SimilarResponse, response_model_by_alias=True)
def similar_games(game_id: int, limit: int = Query(default=10, ge=1, le=100)) -> SimilarResponse:
    items = rec.similar(_store(), game_id, limit)
    if items is None:
        raise HTTPException(status_code=404, detail=f"Game {game_id} has no vector")
    return SimilarResponse(items=_scored(items))
