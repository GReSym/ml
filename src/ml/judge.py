"""LLM relevance judge for `ml judge` (docs/evaluation.md, "LLM-judged evaluation").

Leave-k-out can only reward games the user already owns. The judge supplies graded relevance for any recommended game,
the way RAG/IR evaluation replaces missing relevance labels with LLM judgements (TREC pooling + an UMBRELA-style 0–3 scale):
it sees the user's taste profile (their strongest liked games, a few disliked ones) and one candidate game, and answers
how well the candidate fits. One request per (profile, candidate) pair, so a candidate never influences another's score.

Any OpenAI-compatible chat completions API works (`judge` in the config). Judgements are cached in a JSONL file keyed by
model, PROMPT_VERSION, the profile text and the game, so reruns and new strategies only ask for pairs not judged yet.
Change PROMPT_VERSION whenever the prompt or the profile/candidate rendering changes.
"""

import asyncio
import hashlib
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from ml.config import JudgeSettings
from ml.db import GameCard
from ml.text import html_to_text

log = logging.getLogger(__name__)

PROMPT_VERSION = 1
PROFILE_LIKED = 15  # strongest liked train games shown
PROFILE_DISLIKED = 3
PROFILE_TAGS = 8
PROFILE_DESCRIPTION_CHARS = 200
CANDIDATE_TAGS = 12
CANDIDATE_DESCRIPTION_CHARS = 700
MAX_SCORE = 3
RATE_LIMIT_RETRIES = 30
# Steam store features (categories, accessibility, input, platform services): say nothing about taste, hidden from the judge
FEATURE_TAGS = frozenset({
    "Семейный доступ", "Для одного игрока", "Достижения Steam", "Steam Cloud", "Полная поддержка контроллеров", "Контроллер",
    "Частичная поддержка контроллеров", "Поддержка контроллера DualShock", "Стереозвук", "Поддержка контроллера DualSense",
    "Таблицы лидеров Steam", "#category_playable_at_your_own_pace", "Remote Play Together", "Коллекционные карточки Steam",
    "Регулируемая сложность", "Статистика", "Альтернативные цвета", "Поддержка отслеживания контроллеров", "Объёмный звук",
    "Регулируемый размер текста", "Мастерская Steam", "Настройки субтитров", "Поддержка API системы ввода Steam",
    "Скрытые субтитры", "Remote Play на телевизоре", "#category_contrast_controls", "Поддержка HDR", "Remote Play на планшете",
    "Озвученные игровые меню", "Remote Play на телефоне", "Поддержка комментариев", "Временная шкала Steam",
    "#category_playable_without_vision", "Преобразование текста в речь в чате", "Включён античит Valve",
    "Преобразование речи в текст в чате", "Предметы для SteamVR", "Steam Machine", "Внутриигровые покупки",
    "Управление только мышью", "Управление только клавиатурой", "Только сенсорное управление", "Сенсорное управление",
    "Управление мышью", "Кросс-платформенный мультиплеер", "Поддержка модификаций", "С редактором уровней", "Поддержка VR",
})

SYSTEM_PROMPT = """You are an expert game curator evaluating a game recommender system.
You get a player's taste profile (games they played a lot or reviewed positively, and games they reviewed negatively)
and ONE candidate game the player does not own. Judge how likely this player is to enjoy the candidate, based on its
genre, mechanics, setting, tone and pacing compared with the profile. Game texts are in Russian; answer in English.

Scale:
3 = strong fit: clearly matches a major part of the player's taste; they would very likely enjoy it.
2 = good fit: matches their taste in important ways; they would probably enjoy it.
1 = weak fit: only superficial overlap (a shared broad tag), unlikely to interest them.
0 = no fit: unrelated to or contrary to their taste.

Judge taste fit only, not how famous, popular or well-reviewed the game is. Do not assume the player likes everything
in a genre because of one game. Answer with a JSON object only: {"reason": "<one short sentence>", "score": <0-3>}"""


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit * 0.7 else cut).rstrip(" ,.;:-") + "…"


def _game_line(card: GameCard, tags: int, chars: int) -> str:
    year = f" ({card.release_year})" if card.release_year else ""
    shown = [tag for tag in card.tags if tag not in FEATURE_TAGS][:tags]
    tag_text = f" [{', '.join(shown)}]" if shown else ""
    description = _shorten(html_to_text(card.description), chars)
    return f"{card.title.strip()}{year}{tag_text}" + (f": {description}" if description else "")


def render_profile(liked: list[GameCard], disliked: list[GameCard]) -> str:
    lines = ["Games the player liked (strongest first):"]
    lines += [f"- {_game_line(card, PROFILE_TAGS, PROFILE_DESCRIPTION_CHARS)}" for card in liked[:PROFILE_LIKED]]
    if disliked:
        lines.append("Games the player reviewed negatively:")
        lines += [f"- {_game_line(card, PROFILE_TAGS, PROFILE_DESCRIPTION_CHARS)}" for card in disliked[:PROFILE_DISLIKED]]
    return "\n".join(lines)


def render_candidate(card: GameCard) -> str:
    return "Candidate game:\n" + _game_line(card, CANDIDATE_TAGS, CANDIDATE_DESCRIPTION_CHARS)


def pair_key(model: str, profile: str, game_id: int, repeat: int = 0) -> str:
    digest = hashlib.blake2b(f"{PROMPT_VERSION}\0{model}\0{profile}".encode(), digest_size=12).hexdigest()
    return f"{digest}:{game_id}:{repeat}"


_SCORE = re.compile(r'"score"\s*:\s*"?([0-3])')
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_answer(text: str) -> tuple[int, str] | None:
    """(score, reason) from the model's answer, tolerant of code fences and extra text; None if there's no score."""
    if match := _JSON_OBJECT.search(text or ""):
        try:
            data = json.loads(match.group(0))
            score = int(data["score"])
            if 0 <= score <= MAX_SCORE:
                return score, str(data.get("reason", ""))[:300]
        except (ValueError, KeyError, TypeError):
            pass
    if match := _SCORE.search(text or ""):
        return int(match.group(1)), ""
    return None


@dataclass(frozen=True)
class Judgement:
    score: int
    reason: str


class JudgeError(Exception):
    pass


class JudgeCache:
    """Append-only JSONL: {"key", "score", "reason"} per line. The last line of a key wins."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, Judgement] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    self._entries[row["key"]] = Judgement(int(row["score"]), row.get("reason", ""))
                except (ValueError, KeyError):
                    continue
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")

    def get(self, key: str) -> Judgement | None:
        return self._entries.get(key)

    def put(self, key: str, judgement: Judgement) -> None:
        self._entries[key] = judgement
        self._file.write(json.dumps({"key": key, "score": judgement.score, "reason": judgement.reason},
                                    ensure_ascii=False) + "\n")
        self._file.flush()

    def __len__(self) -> int:
        return len(self._entries)

    def close(self) -> None:
        self._file.close()


class Judge:
    """Async client. `judge_many` asks for every uncached pair with `concurrency` requests in flight."""

    def __init__(self, settings: JudgeSettings, cache: JudgeCache) -> None:
        if not settings.base_url or not settings.model:
            raise JudgeError("Set judge.base_url and judge.model in ml/config.json")
        if not settings.api_key:
            raise JudgeError("Set judge.api_key in ml/config.json")
        self.settings = settings
        self.cache = cache
        self.asked = 0
        self.failed = 0

    def _body(self, profile: str, candidate: str) -> dict:
        body = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{profile}\n\n{candidate}"},
            ],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        if self.settings.json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(self.settings.extra_body)
        return body

    async def _ask(self, client: httpx.AsyncClient, profile: str, candidate: str) -> Judgement | None:
        """Retries transient errors (`max_retries`) and unparseable answers (twice); rate limits (429) are waited out
        (Retry-After when given) up to RATE_LIMIT_RETRIES times without using up the other retries."""
        url = self.settings.base_url.rstrip("/") + "/chat/completions"
        body = self._body(profile, candidate)
        delay = 2.0
        errors = unparseable = limited = 0
        while True:
            try:
                response = await client.post(url, json=body)
            except httpx.HTTPError as e:
                errors += 1
                if errors > self.settings.max_retries:
                    log.warning("Judge request failed: %s", e)
                    return None
                await asyncio.sleep(delay * (0.5 + random.random()))
                delay = min(delay * 2, 60.0)
                continue
            if response.status_code == 429:
                limited += 1
                if limited > RATE_LIMIT_RETRIES:
                    log.warning("Judge request failed: still rate limited after %d waits", RATE_LIMIT_RETRIES)
                    return None
                try:
                    wait = float(response.headers.get("retry-after", ""))
                except ValueError:
                    wait = min(5.0 * limited, 60.0)
                await asyncio.sleep(wait * (0.8 + 0.4 * random.random()))
                continue
            if response.status_code >= 500:
                errors += 1
                if errors > self.settings.max_retries:
                    log.warning("Judge request failed: HTTP %d", response.status_code)
                    return None
                await asyncio.sleep(delay * (0.5 + random.random()))
                delay = min(delay * 2, 60.0)
                continue
            if response.status_code >= 400:
                raise JudgeError(f"HTTP {response.status_code}: {response.text[:500]}")
            try:
                content = response.json()["choices"][0]["message"].get("content") or ""
            except (ValueError, KeyError, IndexError):
                content = ""
            if (parsed := parse_answer(content)) is not None:
                return Judgement(*parsed)
            unparseable += 1
            if unparseable > 2:
                log.warning("Unparseable answer: %r", content[:200])
                return None

    async def judge_many(self, pairs: list[tuple[str, str, str]]) -> dict[str, Judgement]:
        """pairs = (cache key, profile text, candidate text). Returns the judgements found (cached or new) by key."""
        result: dict[str, Judgement] = {}
        todo = []
        queued: set[str] = set()
        for key, profile, candidate in pairs:
            if (cached := self.cache.get(key)) is not None:
                result[key] = cached
            elif key not in queued:
                queued.add(key)
                todo.append((key, profile, candidate))
        log.info("%d pairs: %d cached, %d to ask", len(queued) + len(result), len(result), len(todo))
        semaphore = asyncio.Semaphore(self.settings.concurrency)
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        done = 0

        async with httpx.AsyncClient(headers=headers, timeout=self.settings.timeout_seconds) as client:
            async def run(key: str, profile: str, candidate: str) -> None:
                nonlocal done
                async with semaphore:
                    judgement = await self._ask(client, profile, candidate)
                self.asked += 1
                done += 1
                if judgement is None:
                    self.failed += 1
                else:
                    self.cache.put(key, judgement)
                    result[key] = judgement
                if done % 100 == 0 or done == len(todo):
                    log.info("Judged %d/%d (%d failed)", done, len(todo), self.failed)

            await asyncio.gather(*(run(*item) for item in todo))
        return result
