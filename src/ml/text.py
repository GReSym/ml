"""Builds the text that gets embedded for a game.

Bump TEXT_VERSION whenever the template or the review cleaning/selection rules change: stored vectors
with an older version are stale and the games need to be re-vectorized.
"""

import hashlib
import re
from html import unescape
from html.parser import HTMLParser

from ml.db import GameText

# v1: title + description (HTML stripped).
# v2: same description text, plus selected reviews embedded separately and mixed in (ml.reviews).
TEXT_VERSION = 2

_BLOCK_TAGS = {"p", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol", "tr", "table"}
_SKIP_TAGS = {"script", "style"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    """Strips tags (images, links, formatting) and keeps paragraph breaks."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    text = unescape("".join(parser.parts)).replace("\xa0", " ")
    lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def build_game_text(game: GameText) -> str:
    description = html_to_text(game.description)
    return f"{game.title.strip()}\n\n{description}".strip()


def text_hash(text: str) -> str:
    """Fingerprint of an embedded text, stored next to the vector to notice description changes."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
