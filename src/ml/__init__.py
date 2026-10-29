import argparse
import asyncio
import logging
import sys
from pathlib import Path


def _parse_ids(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


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

    evaluate = commands.add_parser("evaluate", help="offline evaluation on the collected Steam users (writes nothing)")
    evaluate.add_argument("--k", type=_parse_ids, default=[5, 10, 20], help="cut-offs, comma-separated (default 5,10,20)")
    evaluate.add_argument("--test-size", type=int, default=3, help="hidden games per user (default 3)")
    evaluate.add_argument("--min-train-games", type=int, default=5, help="liked games left after hiding (default 5)")
    evaluate.add_argument("--min-playtime", type=int, default=60, help="minutes for an unreviewed game to count (default 60)")
    evaluate.add_argument("--test-selection", choices=["top", "random-liked"], default="top")
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--hybrid-betas", type=_parse_floats,
                          default=[0.05, 0.1, 0.2, 0.4], help="β of the hybrid strategies (default 0.05,0.1,0.2,0.4)")
    evaluate.add_argument("--center", action="store_true", help="mean-centered vectors for the vector strategies")
    evaluate.add_argument("--dislike-modes", type=lambda v: [m.strip() for m in v.split(",") if m.strip()], default=[],
                          help="dislike strategies: centroid,knn,margin,exclude,rocchio (default none)")
    evaluate.add_argument("--dislike-lambdas", type=_parse_floats, default=[],
                          help="λ of the dislike strategies (γ for rocchio), e.g. 0.1,0.25,0.5,1")
    evaluate.add_argument("--dislike-betas", type=_parse_floats, default=[0.0],
                          help="bases of the dislike strategies: 0 = knn, else knn-hybrid-β (default 0)")
    evaluate.add_argument("--hidden-dislikes", type=int, default=0,
                          help="disliked games hidden per user to check they rank low (default 0)")
    evaluate.add_argument("--tags", action="store_true", help="also evaluate tag-only strategies (tags-top, tags, tags-hybrid-β)")
    evaluate.add_argument("--niche-head", type=int, default=1000,
                          help="the niche metrics ignore hidden games among this many most popular (default 1000)")
    evaluate.add_argument("--json", type=Path, help="also write the full report (as stored by the API) to this file")

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
        case "evaluate":
            import json

            from pydantic import ValidationError

            from ml.db import GameRepository
            from ml.evaluation import EvaluationError, EvaluationParams, format_report, run_evaluation
            from ml.vectors import GameVectorStore

            try:
                params = EvaluationParams(
                    ks=args.k, test_size=args.test_size, min_train_games=args.min_train_games,
                    min_playtime_minutes=args.min_playtime, test_selection=args.test_selection, seed=args.seed,
                    hybrid_betas=args.hybrid_betas, center=args.center, dislike_modes=args.dislike_modes,
                    dislike_lambdas=args.dislike_lambdas, dislike_betas=args.dislike_betas,
                    hidden_dislikes=args.hidden_dislikes, tags=args.tags, niche_head=args.niche_head,
                )
            except ValidationError as e:
                sys.exit(f"Invalid parameters: {e}")
            games = GameRepository(settings.database)
            store = GameVectorStore(settings.milvus, settings.model)
            try:
                report = run_evaluation(params, store, games)
            except EvaluationError as e:
                sys.exit(f"Error: {e}")
            finally:
                games.close()
                store.close()
            print(format_report(report))
            if args.json:
                args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        case "export" | "import":
            from ml.backup import BackupError, export_vectors, import_vectors

            try:
                if args.command == "export":
                    export_vectors(settings, args.out, args.collection)
                else:
                    import_vectors(settings, args.file, args.collection, overwrite=args.yes)
            except BackupError as e:
                sys.exit(f"Error: {e}")
