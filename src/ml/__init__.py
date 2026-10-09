import argparse
import asyncio
import logging
import sys


def _parse_ids(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(prog="ml", description="GReSym ML service")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("api", help="run the HTTP API (recommendations, similar games)")
    commands.add_parser("worker", help="consume game vectorization requests from RabbitMQ")

    vectorize = commands.add_parser("vectorize", help="vectorize games directly (bulk)")
    target = vectorize.add_mutually_exclusive_group(required=True)
    target.add_argument("--limit", type=int, help="first N games by id")
    target.add_argument("--ids", type=_parse_ids, help="comma-separated game ids")
    target.add_argument("--all", action="store_true", help="every game (hours on CPU)")
    vectorize.add_argument("--skip-existing", action="store_true",
                           help="skip games that already have a vector with the current text version")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Imports are deferred: torch/sentence-transformers take seconds to import and the API doesn't need them
    from ml.config import ConfigError, get_settings

    try:
        settings = get_settings()
    except ConfigError as e:
        sys.exit(f"Config error: {e}")

    match args.command:
        case "api":
            import uvicorn

            uvicorn.run("ml.api:app", host=settings.api.host, port=settings.api.port)
        case "worker":
            from ml.worker import run_worker

            try:
                asyncio.run(run_worker(settings))
            except KeyboardInterrupt:
                pass
        case "vectorize":
            from ml.vectorize import run_bulk

            run_bulk(settings, ids=args.ids, limit=None if args.all else args.limit, skip_existing=args.skip_existing)
