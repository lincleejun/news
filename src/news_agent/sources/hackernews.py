from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import aiohttp

from news_agent.models import Article
from news_agent.sources.base import BaseSource


class HackerNewsSource(BaseSource):
    name = "hackernews"
    BASE_URL = "https://hacker-news.firebaseio.com/v0"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.max_items = self.config.get("max_items", 30)

    async def _fetch(self, session: aiohttp.ClientSession) -> list[Article]:
        async with session.get(f"{self.BASE_URL}/topstories.json") as resp:
            story_ids = await resp.json()
        story_ids = story_ids[: self.max_items]
        tasks = [self._fetch_item(session, sid) for sid in story_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, Article)]

    async def _fetch_item(self, session: aiohttp.ClientSession, item_id: int) -> Article:
        async with session.get(f"{self.BASE_URL}/item/{item_id}.json") as resp:
            data = await resp.json()
        url = data.get("url", "")
        if not url:
            url = f"https://news.ycombinator.com/item?id={item_id}"
        return Article(
            source=self.name,
            title=data.get("title", ""),
            url=url,
            author=data.get("by", ""),
            score=data.get("score", 0),
            comments_count=data.get("descendants", 0),
            published_at=datetime.fromtimestamp(data.get("time", 0), tz=timezone.utc),
        )
