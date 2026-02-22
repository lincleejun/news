from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import aiohttp
import feedparser

from news_agent.models import Article
from news_agent.sources.base import BaseSource


class RssSource(BaseSource):
    name = "rss"
    timeout = 120

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.feeds = self.config.get("feeds", [])

    async def _fetch(self, session: aiohttp.ClientSession) -> list[Article]:
        articles = []
        for feed_url in self.feeds:
            try:
                async with session.get(feed_url) as resp:
                    content = await resp.text()
                feed = feedparser.parse(content)
                for entry in feed.entries:
                    published = None
                    if (
                        hasattr(entry, "published_parsed")
                        and entry.published_parsed
                    ):
                        published = datetime.fromtimestamp(
                            time.mktime(entry.published_parsed), tz=timezone.utc
                        )
                    articles.append(
                        Article(
                            source=self.name,
                            title=entry.title,
                            url=entry.link,
                            summary=entry.get("summary", "")[:500],
                            author=entry.get("author", ""),
                            published_at=published,
                        )
                    )
            except Exception:
                continue
        return articles
