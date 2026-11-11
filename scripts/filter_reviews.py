"""Flags low-quality reviews with the fine-tuned ALBERT classifier (BAD/GOOD). Run from ml/:

    uv run python scripts/filter_reviews.py --model <HF repo id or path to steam_reviews_classifier> [--threshold 0.2]

Every review in MariaDB is scored in pages of --batch-size (keyset pagination by id). The classifier input is built
exactly like in the labeling notebook: "Review: {content}, Playtime: {playtime}, Voted Up: {bool}, Upvotes: {n},
Votes Funny: {n}" (NULLs become "None"; the tokenizer is plain albert-base-v2 by default, like in training).
A review is flagged when P(BAD) > --threshold.

Writes to --out-dir (default data/review-filter/, git-ignored):
- bad_review_ids.txt      flagged review ids, comma-separated (for DELETE FROM reviews WHERE id IN (...))
- affected_game_ids.txt   their distinct game ids, comma-separated (for `ml vectorize --ids`)
- flagged.csv             review_id, game_id, p_bad, content: for checking the threshold

Only reads MariaDB (database section of config.json); nothing is deleted. Ctrl+C writes what was scored so far and
prints the id to continue from (--from); a continued run writes to its own --out-dir, merge the files yourself.
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import pymysql
import pymysql.cursors
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml.config import ML_ROOT, ConfigError, get_settings

log = logging.getLogger("filter_reviews")

MAX_ID = 2**31 - 1  # reviews.id is INT


def format_review(row: dict) -> str:
    # Must match format_review() of the labeling notebook (the training data), including None/False for NULLs
    return (
        f"Review: {row['content']}, "
        f"Playtime: {row['playtime_at_review']}, "
        f"Voted Up: {bool(row['recommended'])}, "
        f"Upvotes: {row['votes_up']}, "
        f"Votes Funny: {row['votes_funny']}"
    )


class Classifier:
    def __init__(self, model: str, tokenizer: str, device: str, max_length: int, batch_size: int) -> None:
        local = Path(model).expanduser()
        source = str(local.resolve()) if local.exists() else model  # a local directory, else a HF repo id
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self._model = AutoModelForSequenceClassification.from_pretrained(source, dtype=dtype).to(device).eval()
        self._bad = self._model.config.label2id["BAD"]
        self._device = device
        self._max_length = max_length
        self._batch_size = batch_size
        log.info("Model %s (BAD = label %d) on %s, tokenizer %s", source, self._bad, device, tokenizer)

    @torch.inference_mode()
    def bad_probabilities(self, texts: list[str]) -> list[float]:
        # Length-sorted batches: most reviews are short, so this cuts padding a lot
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        scores = [0.0] * len(texts)
        for start in range(0, len(order), self._batch_size):
            chunk = order[start:start + self._batch_size]
            encoded = self._tokenizer([texts[i] for i in chunk], padding=True, truncation=True,
                                      max_length=self._max_length, return_tensors="pt").to(self._device)
            logits = self._model(**encoded).logits.float()
            probs = torch.softmax(logits, dim=-1)[:, self._bad].tolist()
            for i, p in zip(chunk, probs):
                scores[i] = p
        return scores


def write_results(out_dir: Path, flagged: list[tuple[int, int | None, float, str | None]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    flagged.sort(key=lambda row: row[0])
    review_ids = [str(row[0]) for row in flagged]
    game_ids = sorted({row[1] for row in flagged if row[1] is not None})
    (out_dir / "bad_review_ids.txt").write_text(",".join(review_ids) + "\n", encoding="utf-8")
    (out_dir / "affected_game_ids.txt").write_text(",".join(map(str, game_ids)) + "\n", encoding="utf-8")
    with open(out_dir / "flagged.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["review_id", "game_id", "p_bad", "content"])
        for review_id, game_id, p_bad, content in flagged:
            writer.writerow([review_id, game_id, f"{p_bad:.4f}", content])
    log.info("%d reviews flagged, %d games affected → %s", len(review_ids), len(game_ids), out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HF repo id or local directory of the fine-tuned classifier")
    parser.add_argument("--tokenizer", default="albert-base-v2", help="tokenizer (default albert-base-v2)")
    parser.add_argument("--threshold", type=float, default=0.2, help="flag when P(BAD) > threshold (default 0.2)")
    parser.add_argument("--batch-size", type=int, default=1000, help="reviews fetched from the DB per page (default 1000)")
    parser.add_argument("--infer-batch-size", type=int, default=32, help="reviews per forward pass (default 32)")
    parser.add_argument("--max-length", type=int, default=512, help="max tokens per review (default 512)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="cpu or cuda (default: cuda if available)")
    parser.add_argument("--threads", type=int, help="torch CPU threads (default: torch's choice)")
    parser.add_argument("--from", dest="from_id", type=int, default=1, help="first review id (default 1)")
    parser.add_argument("--to", dest="to_id", type=int, default=MAX_ID, help="last review id")
    parser.add_argument("--limit", type=int, help="score at most N reviews (for a quick test)")
    parser.add_argument("--out-dir", type=Path, default=ML_ROOT / "data" / "review-filter")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        db = get_settings().database
    except ConfigError as e:
        sys.exit(f"Config error: {e}")
    if args.threads:
        torch.set_num_threads(args.threads)

    classifier = Classifier(args.model, args.tokenizer, args.device, args.max_length, args.infer_batch_size)
    connection = pymysql.connect(host=db.host, port=db.port, user=db.user, password=db.password,
                                 database=db.database, charset="utf8mb4", autocommit=True,
                                 cursorclass=pymysql.cursors.DictCursor)

    flagged: list[tuple[int, int | None, float, str | None]] = []
    scanned = 0
    last_id = args.from_id - 1
    started = time.monotonic()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS n FROM reviews WHERE id BETWEEN %s AND %s", (args.from_id, args.to_id))
            total = cursor.fetchone()["n"]
            if args.limit is not None:
                total = min(total, args.limit)
            log.info("%d reviews to score (ids %d..%d), threshold %.2f", total, args.from_id, args.to_id, args.threshold)

            while scanned < total:
                page_size = min(args.batch_size, total - scanned)
                cursor.execute(
                    "SELECT id, game_id, content, recommended, playtime_at_review, votes_up, votes_funny "
                    "FROM reviews WHERE id > %s AND id <= %s ORDER BY id LIMIT %s",
                    (last_id, args.to_id, page_size))
                rows = cursor.fetchall()
                if not rows:
                    break
                scores = classifier.bad_probabilities([format_review(row) for row in rows])
                for row, p_bad in zip(rows, scores):
                    if p_bad > args.threshold:
                        flagged.append((row["id"], row["game_id"], p_bad, row["content"]))
                scanned += len(rows)
                last_id = rows[-1]["id"]

                elapsed = time.monotonic() - started
                rate = scanned / elapsed if elapsed else 0.0
                eta = (total - scanned) / rate if rate else 0.0
                log.info("%d/%d (%.1f%%), flagged %d, %.1f reviews/s, ETA %d min, last id %d",
                         scanned, total, 100 * scanned / total, len(flagged), rate, eta // 60, last_id)
    except KeyboardInterrupt:
        log.warning("Interrupted after review id %d: continue with --from %d", last_id, last_id + 1)
    finally:
        connection.close()
        write_results(args.out_dir, flagged)

    if scanned:
        log.info("Scanned %d, flagged %d (%.2f%%) in %.0f s", scanned, len(flagged),
                 100 * len(flagged) / scanned, time.monotonic() - started)
    log.info("Next: DELETE FROM reviews WHERE id IN (<%s>); then "
             "uv run ml vectorize --ids \"$(cat %s)\"",
             args.out_dir / "bad_review_ids.txt", args.out_dir / "affected_game_ids.txt")


if __name__ == "__main__":
    main()
