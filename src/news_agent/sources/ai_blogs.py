from __future__ import annotations

from typing import Any

from news_agent.sources.rss import RssSource


class AiBlogsSource(RssSource):
    name = "ai_blogs"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
