from __future__ import annotations

from typing import Any

import aiohttp

from news_agent.models import Article
from news_agent.sources.rss import RssSource


class WiredSource(RssSource):
    name = "wired"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        if not self.feeds:
            self.feeds = ["https://www.wired.com/feed/rss"]

    async def _fetch(self, session: aiohttp.ClientSession) -> list[Article]:
        articles = await super()._fetch(session)
        for a in articles:
            a.source = self.name
        return articles
