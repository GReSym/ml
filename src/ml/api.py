"""HTTP API of the ML service, called synchronously by the ASP.NET backend (game search, stale vectors; similar games and
recommendations for debugging, the backend gets recommendations from the worker).

Recommendations only use vectors already stored in Milvus. The embedding model is loaded only by the first
description search (GET /search?exact=false), so the API starts in seconds and stays small until then.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from ml import personal
from ml import recommend as rec
from ml.catalogue import CatalogueCache
from ml.config import get_settings
from ml.db import GameRepository
from ml.personal import CamelModel, RecommendationsRequest, RecommendationsResponse, ScoredGameDto, scored
from ml.search import GameSearch, SearchMode
from ml.staleness import Reason, filter_items, find_stale
from ml.text_index import GameTextStore
from ml.vectors import GameVectorStore


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


class SearchResponse(CamelModel):
    items: list[ScoredGameDto]
    mode: SearchMode


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
    texts = GameTextStore(settings.milvus)
    texts.ensure_collection()
    app.state.search = GameSearch(settings, texts, store)
    # Loaded on the first recommendation request, so the API starts even if MariaDB is briefly unavailable
    app.state.catalogue = CatalogueCache(settings, store)
    yield
    texts.close()
    store.close()


app = FastAPI(title="GReSym ML", lifespan=lifespan)


def _store() -> GameVectorStore:
    return app.state.store


# Sync endpoints: FastAPI runs them in a thread pool, so the blocking Milvus client doesn't block the event loop.
@app.get("/health", response_model=HealthResponse, response_model_by_alias=True)
def health() -> HealthResponse:
    store = _store()
    return HealthResponse(status="ok", collection=store.collection, vectors=store.count())


@app.post("/recommendations", response_model=RecommendationsResponse, response_model_by_alias=True)
def recommendations(request: RecommendationsRequest) -> RecommendationsResponse:
    """Not used by the backend any more (it queues ml.recommend for the worker); kept for debugging."""
    return personal.recommend(_store(), app.state.catalogue.get(), request)


@app.get("/games/{game_id}/similar", response_model=SimilarResponse, response_model_by_alias=True)
def similar_games(game_id: int, limit: int = Query(default=10, ge=1, le=100)) -> SimilarResponse:
    items = rec.similar(_store(), game_id, limit)
    if items is None:
        raise HTTPException(status_code=404, detail=f"Game {game_id} has no vector")
    return SimilarResponse(items=scored(items))


@app.get("/search", response_model=SearchResponse, response_model_by_alias=True)
def search(
    q: str = Query(min_length=1, max_length=500),
    exact: bool = Query(default=True, description="true: keywords only (BM25); false: also by meaning (loads the model)"),
    limit: int = Query(default=200, ge=1, le=1000),
) -> SearchResponse:
    """Games ranked by relevance to the query (weighted RRF, see ml.search)."""
    result = app.state.search.search(q.strip(), exact, limit)
    return SearchResponse(items=scored([(game_id, round(score, 6)) for game_id, score in result.items]), mode=result.mode)


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
