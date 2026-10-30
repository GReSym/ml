"""Personal recommendations as requested by the backend: the payload, the response and the call into `recommend`.

Shared by the RabbitMQ worker (ml.recommend, the path the backend uses) and the HTTP API (POST /recommendations, for debugging).
"""

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from ml import recommend as rec
from ml.catalogue import CatalogueStats
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
    # How disliked games act (the user's setting, see recommend.USER_DISLIKE_MODES); None = ignore
    dislike_mode: rec.UserDislikeMode | None = None
    # 0 = no effect; penalty weight or neighbours hidden per dislike, depending on the mode; None = the default (50)
    dislike_strength: int | None = Field(default=None, ge=0, le=100)


class ScoredGameDto(CamelModel):
    game_id: int
    score: float


class RecommendationsResponse(CamelModel):
    items: list[ScoredGameDto]
    used_ratings: int  # liked games with a vector, plus disliked ones when the dislike mode is active
    mainstream: int
    dislike_mode: rec.UserDislikeMode
    dislike_strength: int


def scored(items: list[tuple[int, float]]) -> list[ScoredGameDto]:
    return [ScoredGameDto(game_id=game_id, score=score) for game_id, score in items]


def recommend(store: GameVectorStore, stats: CatalogueStats, request: RecommendationsRequest) -> RecommendationsResponse:
    """Blocking (Milvus searches): call it from a thread."""
    ratings = [rec.Rating(r.game_id, r.rating, r.recommend) for r in request.ratings]
    mainstream = rec.DEFAULT_MAINSTREAM if request.mainstream is None else request.mainstream
    dislike_mode = request.dislike_mode or rec.DEFAULT_USER_DISLIKE_MODE
    dislike_strength = rec.DEFAULT_DISLIKE_STRENGTH if request.dislike_strength is None else request.dislike_strength
    result = rec.recommend(store, stats, ratings, request.limit, mainstream,
                           dislikes=rec.dislike_params(dislike_mode, dislike_strength))
    return RecommendationsResponse(items=scored(result.items), used_ratings=result.used_ratings, mainstream=mainstream,
                                   dislike_mode=dislike_mode, dislike_strength=dislike_strength)
