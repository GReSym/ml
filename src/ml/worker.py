"""RabbitMQ consumers: vectorizes games, computes recommendations and runs offline evaluations requested by the backend.

Messages (JSON):
- routing key game.vectorize: {"gameId": 123, "reason": "updated"}
- routing key recommendation.run: {"recommendationId": 5, "ratings": [...], "limit", "mainstream", "dislikeMode",
  "dislikeStrength"} (personal.RecommendationsRequest + the id); the result goes back to the API as
  {"recommendationId", "status": "completed"|"failed", "items"?: [{"gameId", "score"}], "usedRatings"?, "error"?, "at"}
  (recommendation.result). "failed" only once retrying is pointless; the API then falls back to popular games
- routing key evaluation.run: {"runId": 7}; progress and results go back to the API as
  {"runId", "status": "running"|"completed"|"failed", "results"?, "usersEvaluated"?, "error"?, "at"} (evaluation.result)
Topology must match the backend's declaration exactly (backend: Infrastructure/Messaging/RabbitMqTopology.cs).
"""

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime

import aio_pika
from aio_pika.abc import AbstractChannel, AbstractExchange, AbstractIncomingMessage, AbstractQueue
from pydantic import Field, ValidationError

from ml import personal
from ml.catalogue import CatalogueCache
from ml.config import RabbitMqSettings, Settings
from ml.db import GameRepository
from ml.evaluation import EvaluationError, EvaluationParams, run_evaluation
from ml.vectorize import Vectorizer
from ml.vectors import GameVectorStore

log = logging.getLogger(__name__)


async def _declare_work_queue(
    channel: AbstractChannel,
    exchange: AbstractExchange,
    dlx: AbstractExchange,
    name: str,
    dead_letter_name: str,
    routing_key: str,
    delivery_limit: int,
    dead_letter_exchange: str,
) -> AbstractQueue:
    """Quorum queue bound to the exchange, dead-lettered to its .dlq with the same routing key."""
    queue = await channel.declare_queue(
        name,
        durable=True,
        arguments={
            "x-queue-type": "quorum",
            "x-delivery-limit": delivery_limit,
            "x-dead-letter-exchange": dead_letter_exchange,
        },
    )
    await queue.bind(exchange, routing_key=routing_key)

    dead_letters = await channel.declare_queue(dead_letter_name, durable=True, arguments={"x-queue-type": "quorum"})
    await dead_letters.bind(dlx, routing_key=routing_key)
    return queue


async def declare_topology(channel: AbstractChannel, settings: RabbitMqSettings) -> AbstractQueue:
    """Declares everything both services use; returns the vectorization queue."""
    exchange = await channel.declare_exchange(settings.exchange, aio_pika.ExchangeType.DIRECT, durable=True)
    dlx = await channel.declare_exchange(settings.dead_letter_exchange, aio_pika.ExchangeType.DIRECT, durable=True)

    vectorize = await _declare_work_queue(
        channel, exchange, dlx, settings.queue, settings.dead_letter_queue, settings.routing_key,
        settings.delivery_limit, settings.dead_letter_exchange,
    )
    await _declare_work_queue(
        channel, exchange, dlx, settings.evaluation_queue, settings.evaluation_dead_letter_queue,
        settings.evaluation_routing_key, settings.evaluation_delivery_limit, settings.dead_letter_exchange,
    )
    # Consumed by the API; declared here too, so results published before the API ever started aren't lost
    await _declare_work_queue(
        channel, exchange, dlx, settings.evaluation_results_queue, settings.evaluation_results_dead_letter_queue,
        settings.evaluation_results_routing_key, settings.delivery_limit, settings.dead_letter_exchange,
    )
    await _declare_work_queue(
        channel, exchange, dlx, settings.recommendation_queue, settings.recommendation_dead_letter_queue,
        settings.recommendation_routing_key, settings.recommendation_delivery_limit, settings.dead_letter_exchange,
    )
    await _declare_work_queue(
        channel, exchange, dlx, settings.recommendation_results_queue,
        settings.recommendation_results_dead_letter_queue, settings.recommendation_results_routing_key,
        settings.delivery_limit, settings.dead_letter_exchange,
    )
    return vectorize


def _parse_game_id(message: AbstractIncomingMessage) -> int | None:
    try:
        game_id = json.loads(message.body)["gameId"]
    except (ValueError, KeyError, TypeError):
        return None
    return game_id if isinstance(game_id, int) and game_id > 0 else None


async def _handle(message: AbstractIncomingMessage, vectorizer: Vectorizer) -> None:
    game_id = _parse_game_id(message)
    if game_id is None:
        log.error("Malformed message %s, dead-lettering: %r", message.message_id, message.body[:200])
        await message.reject(requeue=False)
        return

    try:
        # Embedding is CPU-bound: run it off the event loop so the AMQP heartbeats keep going
        result = await asyncio.to_thread(vectorizer.vectorize, [game_id])
    except Exception:
        # Requeued; after x-delivery-limit redeliveries the broker dead-letters it
        log.exception("Vectorizing game %d failed (delivery count %s)", game_id, message.headers.get("x-delivery-count"))
        await message.nack(requeue=True)
        return

    if result.deleted:
        log.info("Game %d not in MariaDB, removed its vector", game_id)
    else:
        log.info("Vectorized game %d", game_id)
    await message.ack()


def _parse_run_id(message: AbstractIncomingMessage) -> int | None:
    try:
        run_id = json.loads(message.body)["runId"]
    except (ValueError, KeyError, TypeError):
        return None
    return run_id if isinstance(run_id, int) and run_id > 0 else None


def _evaluate_run(settings: Settings, run_id: int) -> dict:
    """Runs in a worker thread with its own DB connection and Milvus client (the vectorizer's aren't thread-safe)."""
    games = GameRepository(settings.database)
    store = GameVectorStore(settings.milvus, settings.model)
    try:
        run = games.get_evaluation_run(run_id)
        if run is None:
            raise EvaluationError(f"Evaluation run {run_id} not found")
        try:
            params = EvaluationParams.model_validate_json(run.parameters or "{}")
        except ValidationError as e:
            raise EvaluationError(f"Invalid parameters: {e}") from e

        def progress(done: int, total: int) -> None:
            if done % 50 == 0 or done == total:
                log.info("Evaluation run %d: %d/%d users", run_id, done, total)

        return run_evaluation(params, store, games, progress)
    finally:
        games.close()
        store.close()


async def _publish(exchange: AbstractExchange, routing_key: str, status: str, **fields: object) -> None:
    """A report for the API: {**fields, "status", "at"}."""
    body = {**fields, "status": status, "at": datetime.now(UTC).isoformat()}
    await exchange.publish(
        aio_pika.Message(
            json.dumps(body).encode(),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            message_id=str(uuid.uuid4()),
            type=routing_key,
        ),
        routing_key=routing_key,
        mandatory=True,
    )


async def _publish_result(exchange: AbstractExchange, settings: RabbitMqSettings, run_id: int, status: str,
                          **fields: object) -> None:
    await _publish(exchange, settings.evaluation_results_routing_key, status, runId=run_id, **fields)


async def _handle_evaluation(message: AbstractIncomingMessage, settings: Settings, exchange: AbstractExchange) -> None:
    rabbit = settings.rabbitmq
    run_id = _parse_run_id(message)
    if run_id is None:
        log.error("Malformed evaluation message %s, dead-lettering: %r", message.message_id, message.body[:200])
        await message.reject(requeue=False)
        return

    log.info("Evaluation run %d started", run_id)
    await _publish_result(exchange, rabbit, run_id, "running")
    try:
        # Milvus searches and SQL are blocking: off the event loop, so heartbeats and vectorization keep going
        report = await asyncio.to_thread(_evaluate_run, settings, run_id)
    except EvaluationError as e:
        # Retrying won't help: report it and drop the message
        log.error("Evaluation run %d failed: %s", run_id, e)
        await _publish_result(exchange, rabbit, run_id, "failed", error=str(e))
        await message.ack()
        return
    except Exception as e:
        # Maybe transient (Milvus/MariaDB down): requeue; after x-delivery-limit the broker dead-letters it
        log.exception("Evaluation run %d failed (delivery count %s)", run_id, message.headers.get("x-delivery-count"))
        await _publish_result(exchange, rabbit, run_id, "failed", error=f"{type(e).__name__}: {e}")
        await message.nack(requeue=True)
        return

    await _publish_result(exchange, rabbit, run_id, "completed", results=report,
                          usersEvaluated=report["users"]["evaluated"])
    log.info("Evaluation run %d completed: %d users in %.1f s", run_id, report["users"]["evaluated"],
             report["durationSeconds"])
    await message.ack()


class RecommendTask(personal.RecommendationsRequest):
    recommendation_id: int = Field(gt=0)


def _parse_recommendation_id(message: AbstractIncomingMessage) -> int | None:
    """Only the id, to report an otherwise invalid task as failed."""
    try:
        recommendation_id = json.loads(message.body)["recommendationId"]
    except (ValueError, KeyError, TypeError):
        return None
    return recommendation_id if isinstance(recommendation_id, int) and recommendation_id > 0 else None


async def _handle_recommendation(message: AbstractIncomingMessage, settings: RabbitMqSettings,
                                 exchange: AbstractExchange, store: GameVectorStore,
                                 catalogue: CatalogueCache) -> None:
    async def publish(recommendation_id: int, status: str, **fields: object) -> None:
        await _publish(exchange, settings.recommendation_results_routing_key, status,
                       recommendationId=recommendation_id, **fields)

    try:
        task = RecommendTask.model_validate_json(message.body)
    except ValidationError as e:
        recommendation_id = _parse_recommendation_id(message)
        if recommendation_id is None:
            log.error("Malformed recommendation message %s, dead-lettering: %r", message.message_id, message.body[:200])
            await message.reject(requeue=False)
            return
        # Retrying won't help: report it, so the API falls back to popular games
        log.error("Invalid recommendation task %d: %s", recommendation_id, e)
        await publish(recommendation_id, "failed", error=f"Invalid task: {e.error_count()} validation errors")
        await message.ack()
        return

    try:
        # Milvus searches are blocking: off the event loop. The first call also loads the catalogue stats (~6 s)
        result = await asyncio.to_thread(lambda: personal.recommend(store, catalogue.get(), task))
    except Exception as e:
        # x-delivery-count = earlier failed deliveries; the broker dead-letters the message past x-delivery-limit.
        # Report one attempt early, so a message never reaches the DLQ without the API hearing about it
        failures = int(message.headers.get("x-delivery-count") or 0)
        if failures < settings.recommendation_delivery_limit - 1:
            # Maybe transient (Milvus/MariaDB down): requeue silently
            log.exception("Recommendation %d failed (delivery count %d), requeueing", task.recommendation_id, failures)
            await message.nack(requeue=True)
            return
        log.exception("Recommendation %d failed for the last time", task.recommendation_id)
        await publish(task.recommendation_id, "failed", error=f"{type(e).__name__}: {e}")
        await message.ack()
        return

    await publish(task.recommendation_id, "completed", items=[i.model_dump(by_alias=True) for i in result.items],
                  usedRatings=result.used_ratings)
    log.info("Recommendation %d: %d games from %d ratings", task.recommendation_id, len(result.items),
             result.used_ratings)
    await message.ack()


async def _consume_vectorization(queue: AbstractQueue, vectorizer: Vectorizer) -> None:
    # Messages are handled one at a time: the vectorizer isn't thread-safe and the CPU is the bottleneck
    async with queue.iterator() as messages:
        async for message in messages:
            await _handle(message, vectorizer)


async def _consume_evaluations(queue: AbstractQueue, settings: Settings, exchange: AbstractExchange) -> None:
    async with queue.iterator() as messages:
        async for message in messages:
            await _handle_evaluation(message, settings, exchange)


async def _consume_recommendations(queue: AbstractQueue, settings: RabbitMqSettings, exchange: AbstractExchange,
                                   store: GameVectorStore, catalogue: CatalogueCache) -> None:
    # One at a time: a request takes ~0.3 s, and the Milvus client is shared
    async with queue.iterator() as messages:
        async for message in messages:
            await _handle_recommendation(message, settings, exchange, store, catalogue)


async def run_worker(settings: Settings) -> None:
    rabbit = settings.rabbitmq
    vectorizer = Vectorizer(settings)  # load the model before taking messages
    # Recommendations get their own Milvus client (the vectorizer's isn't thread-safe); the catalogue loads on first use
    store = GameVectorStore(settings.milvus, settings.model)
    catalogue = CatalogueCache(settings, store)
    connection = await aio_pika.connect_robust(
        host=rabbit.host,
        port=rabbit.port,
        virtualhost=rabbit.virtual_host,
        login=rabbit.user,
        password=rabbit.password,
        client_properties={"connection_name": "gresym-ml-worker"},
    )
    try:
        # One channel per consumer, so their prefetch limits are independent
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=rabbit.prefetch)
        vectorize = await declare_topology(channel, rabbit)

        evaluation_channel = await connection.channel(publisher_confirms=True)
        await evaluation_channel.set_qos(prefetch_count=1)  # one run at a time
        evaluate = await evaluation_channel.get_queue(rabbit.evaluation_queue, ensure=False)
        exchange = await evaluation_channel.get_exchange(rabbit.exchange, ensure=False)

        recommendation_channel = await connection.channel(publisher_confirms=True)
        await recommendation_channel.set_qos(prefetch_count=rabbit.prefetch)
        recommend = await recommendation_channel.get_queue(rabbit.recommendation_queue, ensure=False)
        recommendation_exchange = await recommendation_channel.get_exchange(rabbit.exchange, ensure=False)

        log.info("Waiting for messages on %s, %s and %s", rabbit.queue, rabbit.recommendation_queue,
                 rabbit.evaluation_queue)
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(_consume_vectorization(vectorize, vectorizer))
            tasks.create_task(_consume_recommendations(recommend, rabbit, recommendation_exchange, store, catalogue))
            tasks.create_task(_consume_evaluations(evaluate, settings, exchange))
    finally:
        await connection.close()
        store.close()
        vectorizer.close()
