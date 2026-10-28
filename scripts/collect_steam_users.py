"""Collects anonymized public Steam users for the offline evaluation (docs/evaluation.md). Run once, from ml/:

    uv run python scripts/collect_steam_users.py --seeds <profile>[,<profile>...] --count 200 [--resume]

Candidates are found by snowball sampling over the friend graph (official Web API, ISteamUser/GetFriendList):
starting from the seed profiles (SteamID64, vanity name or profile URL), a random sample of each public user's friends
is visited, breadth-first, up to --max-depth hops away. (The store's review API, used for this at first, redirects
to the store front page since 2026-10-26.) A visited user is accepted when:
- the profile is public (GetPlayerSummaries) and its game details are public (GetOwnedGames returns games),
- more than steam.min_played_games games with playtime >= steam.min_playtime_minutes are in our catalogue,
- the profile's public review pages list at least steam.min_reviews reviews (their thumbs up/down are kept).
Seed profiles are only starting points, never collected themselves.

Only the anonymized result is written: `user-N` names, an HMAC of the SteamID (deduplication only), Steam app ids,
playtime and thumbs up/down. No SteamID, persona name, profile URL, review text or dates, and no traversal state:
--resume walks the graph again from the seeds and skips users already in the file (by hash).
The backend imports the file (POST /api/admin/evaluation/steam-users/import); this script only reads MariaDB.

Needs steam.api_key and steam.anonymization_salt in config.json.
"""

import argparse
import hashlib
import hmac
import json
import logging
import random
import re
import sys
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import httpx

from ml.config import ML_ROOT, PLACEHOLDERS, ConfigError, SteamSettings, get_settings
from ml.db import GameRepository

log = logging.getLogger("collect_steam_users")

FORMAT_VERSION = 1
SUMMARIES_URL = "https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v2/"
FRIENDS_URL = "https://api.steampowered.com/ISteamUser/GetFriendList/v1/"
VANITY_URL = "https://api.steampowered.com/ISteamUser/ResolveVanityURL/v1/"
OWNED_GAMES_URL = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/"
PROFILE_REVIEWS_URL = "https://steamcommunity.com/profiles/{steam_id}/recommended/"
PUBLIC_PROFILE = 3  # communityvisibilitystate
REVIEWS_PER_PAGE = 10
MAX_RETRIES = 5
STEAM_ID = re.compile(r"^\d{17}$")
PROFILE_LINK = re.compile(r"steamcommunity\.com/(profiles|id)/([^/?#]+)")

# A user's review page lists boxes like <div class="review_box"> … href=".../recommended/620/" … icon_thumbsUp_v6.png
REVIEW_BOX = 'class="review_box"'
REVIEW_APP = re.compile(r"/recommended/(\d+)")
REVIEW_THUMB = re.compile(r"icon_thumbs(Up|Down)")


class Steam:
    """Polite HTTP client: a fixed delay before every request, backoff on 429/5xx."""

    def __init__(self, settings: SteamSettings) -> None:
        self._settings = settings
        self._client = httpx.Client(timeout=30, follow_redirects=True)
        self.requests = 0

    def get(self, url: str, params: dict | None = None, community: bool = False) -> httpx.Response | None:
        delay = self._settings.community_delay_seconds if community else self._settings.request_delay_seconds
        for attempt in range(MAX_RETRIES):
            time.sleep(delay)
            self.requests += 1
            try:
                response = self._client.get(url, params=params)
            except httpx.HTTPError as e:
                log.warning("%s: %s", url, e)
            else:
                if response.status_code == 200:
                    return response
                if response.status_code not in (429, 500, 502, 503, 504):
                    # 401/403: private friend list or profile, nothing to retry
                    log.debug("%s: HTTP %d", url, response.status_code)
                    return None
                log.warning("%s: HTTP %d, retrying", url, response.status_code)
            time.sleep(min(300, 10 * 2**attempt))
        log.error("%s: giving up after %d attempts", url, MAX_RETRIES)
        return None

    def get_json(self, url: str, params: dict | None = None) -> dict:
        response = self.get(url, params)
        if response is None:
            return {}
        try:
            return response.json()
        except ValueError:
            log.warning("%s: not JSON (HTTP %d, %s): %r", url, response.status_code, response.url, response.text[:200])
            return {}

    def close(self) -> None:
        self._client.close()


def resolve_seed(steam: Steam, api_key: str, seed: str) -> str | None:
    """SteamID64 of a seed given as an id, a vanity name or a profile URL."""
    seed = seed.strip()
    if link := PROFILE_LINK.search(seed):
        kind, value = link.groups()
        if kind == "profiles":
            return value if STEAM_ID.match(value) else None
        seed = value
    if STEAM_ID.match(seed):
        return seed
    data = steam.get_json(VANITY_URL, {"key": api_key, "vanityurl": seed}).get("response", {})
    return data.get("steamid") if data.get("success") == 1 else None


def public_steam_ids(steam: Steam, api_key: str, steam_ids: list[str]) -> set[str]:
    public: set[str] = set()
    for start in range(0, len(steam_ids), 100):
        batch = steam_ids[start : start + 100]
        data = steam.get_json(SUMMARIES_URL, {"key": api_key, "steamids": ",".join(batch)})
        for player in data.get("response", {}).get("players", []):
            if player.get("communityvisibilitystate") == PUBLIC_PROFILE:
                public.add(player["steamid"])
    return public


def friend_ids(steam: Steam, api_key: str, steam_id: str) -> list[str]:
    """Empty if the friend list is private (Steam answers 401; that's a separate setting from the profile's visibility)."""
    data = steam.get_json(FRIENDS_URL, {"key": api_key, "steamid": steam_id, "relationship": "friend"})
    if not data:
        log.debug("Friend list private or unavailable, not expanding this user")
    return [friend["steamid"] for friend in data.get("friendslist", {}).get("friends", [])]


def owned_games(steam: Steam, api_key: str, steam_id: str) -> dict[int, int] | None:
    """App id → playtime in minutes. None if the game details are private."""
    data = steam.get_json(
        OWNED_GAMES_URL,
        {"key": api_key, "steamid": steam_id, "include_played_free_games": 1, "include_appinfo": 0,
         "skip_unvetted_apps": 0},
    )
    games = data.get("response", {}).get("games")
    if not games:
        return None
    return {int(game["appid"]): int(game.get("playtime_forever", 0)) for game in games}


def profile_reviews(steam: Steam, steam_id: str, max_pages: int) -> dict[int, bool]:
    """App id → thumbs up, from the profile's public review pages."""
    result: dict[int, bool] = {}
    for page in range(1, max_pages + 1):
        response = steam.get(PROFILE_REVIEWS_URL.format(steam_id=steam_id), {"p": page}, community=True)
        if response is None:
            break
        boxes = response.text.split(REVIEW_BOX)[1:]
        for box in boxes:
            app = REVIEW_APP.search(box)
            thumb = REVIEW_THUMB.search(box)
            if app and thumb:
                result.setdefault(int(app.group(1)), thumb.group(1) == "Up")
        if len(boxes) < REVIEWS_PER_PAGE:
            break
    return result


def steam_id_hash(salt: str, steam_id: str) -> str:
    return hmac.new(salt.encode(), steam_id.encode(), hashlib.sha256).hexdigest()


def save(path: Path, dataset: dict) -> None:
    """Atomic, so an interrupted run never leaves a broken file."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(dataset, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", required=True,
                        help="comma-separated seed profiles: SteamID64, vanity name or profile URL (not stored)")
    parser.add_argument("--count", type=int, default=200, help="users to collect (default 200)")
    parser.add_argument("--out", type=Path, default=ML_ROOT / "data" / "steam-users.json")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run from --out")
    parser.add_argument("--seed", type=int, default=42, help="random seed of the friend sampling")
    parser.add_argument("--max-review-pages", type=int, default=10, help="profile review pages per user (10 per page)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    try:
        settings = get_settings()
    except ConfigError as e:
        sys.exit(f"Config error: {e}")
    steam_settings = settings.steam
    if steam_settings.api_key in PLACEHOLDERS or steam_settings.anonymization_salt in PLACEHOLDERS:
        sys.exit("Fill in steam.api_key and steam.anonymization_salt in config.json "
                 "(the salt is any long random string, e.g. `openssl rand -hex 32`; never change it afterwards)")

    if args.out.exists() and not args.resume:
        sys.exit(f"{args.out} exists: pass --resume to continue it, or choose another --out")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        dataset = json.loads(args.out.read_text(encoding="utf-8"))
    else:
        dataset = {
            "version": FORMAT_VERSION,
            "collectedAt": None,
            "criteria": {
                "sampling": "friend-graph snowball",
                "minPlayedGamesExclusive": steam_settings.min_played_games,
                "minReviews": steam_settings.min_reviews,
                "minPlaytimeMinutes": steam_settings.min_playtime_minutes,
                "friendsPerUser": steam_settings.friends_per_user,
                "maxDepth": steam_settings.max_depth,
                "seed": args.seed,
            },
            "users": [],
        }
    users: list[dict] = dataset["users"]
    known_hashes = {user["steamIdHash"] for user in users}

    games = GameRepository(settings.database)
    try:
        catalogue = games.get_steam_app_ids()
    finally:
        games.close()

    steam = Steam(steam_settings)
    api_key = steam_settings.api_key
    rng = random.Random(args.seed)
    try:
        seeds = []
        for seed in args.seeds.split(","):
            steam_id = resolve_seed(steam, api_key, seed)
            if steam_id is None:
                sys.exit(f"Can't resolve seed profile {seed!r}")
            seeds.append(steam_id)
        log.info("%d games in the catalogue, %d seed profiles; %d users collected so far",
                 len(catalogue), len(seeds), len(users))

        # BFS over public profiles; SteamIDs live only in memory
        visited: set[str] = set(seeds)
        queue: deque[tuple[str, int]] = deque((steam_id, 0) for steam_id in seeds)
        examined = 0
        while queue and len(users) < args.count:
            steam_id, depth = queue.popleft()

            if depth < steam_settings.max_depth:
                friends = [f for f in friend_ids(steam, api_key, steam_id) if f not in visited]
                sample = rng.sample(friends, min(steam_settings.friends_per_user, len(friends)))
                visited.update(sample)
                public = sorted(public_steam_ids(steam, api_key, sample))
                rng.shuffle(public)
                queue.extend((friend, depth + 1) for friend in public)

            if depth == 0:
                continue  # seeds are only starting points
            examined += 1
            id_hash = steam_id_hash(steam_settings.anonymization_salt, steam_id)
            if id_hash in known_hashes:
                continue

            library = owned_games(steam, api_key, steam_id)
            if library is None:
                log.debug("Depth %d: game details private", depth)
                continue
            played = [a for a, minutes in library.items() if minutes >= steam_settings.min_playtime_minutes]
            played_in_catalogue = sum(1 for a in played if a in catalogue)
            if played_in_catalogue <= steam_settings.min_played_games:
                log.debug("Depth %d: %d played games in the catalogue", depth, played_in_catalogue)
                continue

            thumbs = profile_reviews(steam, steam_id, args.max_review_pages)
            if len(thumbs) < steam_settings.min_reviews:
                log.debug("Depth %d: %d reviews", depth, len(thumbs))
                continue

            users.append({
                "name": f"user-{len(users) + 1}",
                "steamIdHash": id_hash,
                "gamesOwned": len(library),
                "gamesPlayed": len(played),
                "reviewsCount": len(thumbs),
                "games": [
                    {"steamAppId": a, "playtimeMinutes": library.get(a, 0), "recommended": thumbs.get(a)}
                    for a in sorted(set(library) | set(thumbs))
                ],
            })
            known_hashes.add(id_hash)
            dataset["collectedAt"] = datetime.now(UTC).isoformat()
            save(args.out, dataset)
            log.info("%s (depth %d): %d owned, %d played (%d in catalogue), %d reviews; examined %d, queue %d, %d requests",
                     users[-1]["name"], depth, len(library), len(played), played_in_catalogue, len(thumbs),
                     examined, len(queue), steam.requests)

        if len(users) < args.count:
            log.warning("The friend graph ran out after %d examined users: add seeds or raise "
                        "steam.max_depth / steam.friends_per_user", examined)
    except KeyboardInterrupt:
        log.info("Interrupted; continue with --resume")
    finally:
        steam.close()
        save(args.out, dataset)

    log.info("Done: %d users in %s", len(users), args.out)


if __name__ == "__main__":
    main()
