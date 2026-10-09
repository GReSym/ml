"""RabbitMQ consumer: vectorizes games requested by the backend.

Message (JSON, routing key game.vectorize): {"gameId": 123, "reason": "updated"}
Topology must match the backend's declaration exactly (backend: Infrastructure/Messaging).
"""

import asyncio
import json
import logging

import aio_pika
from aio_pika.abc import AbstractChannel, AbstractIncomingMessage, AbstractQueue

from ml.config import RabbitMqSettings, Settings
from ml.vectorize import Vectorizer

log = logging.getLogger(__name__)


async def declare_topology(channel: AbstractChannel, settings: RabbitMqSettings) -> AbstractQueue:
    exchange = await channel.declare_exchange(settings.exchange, aio_pika.ExchangeType.DIRECT, durable=True)
    dlx = await channel.declare_exchange(settings.dead_letter_exchange, aio_pika.ExchangeType.DIRECT, durable=True)

    queue = await channel.declare_queue(
        settings.queue,
        durable=True,
        arguments={
            "x-queue-type": "quorum",
            "x-delivery-limit": settings.delivery_limit,
            "x-dead-letter-exchange": settings.dead_letter_exchange,
        },
    )
    await queue.bind(exchange, routing_key=settings.routing_key)

    dead_letters = await channel.declare_queue(
        settings.dead_letter_queue, durable=True, arguments={"x-queue-type": "quorum"}
    )
    await dead_letters.bind(dlx, routing_key=settings.routing_key)
    return queue


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


async def run_worker(settings: Settings) -> None:
    rabbit = settings.rabbitmq
    vectorizer = Vectorizer(settings)  # load the model before taking messages
    connection = await aio_pika.connect_robust(
        host=rabbit.host,
        port=rabbit.port,
        virtualhost=rabbit.virtual_host,
        login=rabbit.user,
        password=rabbit.password,
        client_properties={"connection_name": "gresym-ml-worker"},
    )
    try:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=rabbit.prefetch)
        queue = await declare_topology(channel, rabbit)
        log.info("Waiting for messages on %s", rabbit.queue)

        # Messages are handled one at a time: the vectorizer isn't thread-safe and the CPU is the bottleneck
        async with queue.iterator() as messages:
            async for message in messages:
                await _handle(message, vectorizer)
    finally:
        await connection.close()
        vectorizer.close()
