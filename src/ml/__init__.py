import argparse
import asyncio
import logging
import sys
from pathlib import Path


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

    stale = commands.add_parser("stale", help="list games whose vector is out of date (nothing is vectorized)")
    stale.add_argument("--limit", type=int, default=30, help="rows to print (default 30)")
    stale.add_argument("--reason", action="append",
                       help="only these reasons (repeatable): description, sentiment, reviews, growth, "
                            "outdated, missing, deleted")
    stale.add_argument("--ids-only", action="store_true",
                       help="print only comma-separated ids (for `ml vectorize --ids`)")

    export = commands.add_parser("export", help="back up the game vectors from Milvus into an .npz file")
    export.add_argument("--out", type=Path, help="target file (default milvus-<collection>-<date>.npz)")
    export.add_argument("--collection", help="collection to export (default milvus.collection)")

    restore = commands.add_parser("import", help="restore game vectors from an `ml export` file into Milvus")
    restore.add_argument("file", type=Path)
    restore.add_argument("--collection", help="target collection (default milvus.collection), created if missing")
    restore.add_argument("--yes", action="store_true", help="import into a collection that already has vectors")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Imports are deferred: the embedding libraries (onnxruntime, torch) take seconds to import and the API doesn't need them
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
        case "stale":
            from ml.staleness import print_report

            print_report(settings, limit=args.limit, reasons=args.reason, ids_only=args.ids_only)
        case "export" | "import":
            from ml.backup import BackupError, export_vectors, import_vectors

            try:
                if args.command == "export":
                    export_vectors(settings, args.out, args.collection)
                else:
                    import_vectors(settings, args.file, args.collection, overwrite=args.yes)
            except BackupError as e:
                sys.exit(f"Error: {e}")
