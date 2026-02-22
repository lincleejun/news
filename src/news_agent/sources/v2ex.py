from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import aiohttp

from news_agent.models import Article
from news_agent.sources.base import BaseSource


class V2exSource(BaseSource):
    name = "v2ex"
    BASE_URL = "https://www.v2ex.com/api/v2"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.nodes = self.config.get("nodes", ["programmer", "create", "apple"])

    async def _fetch(self, session: aiohttp.ClientSession) -> list[Article]:
        articles = []
        headers = {"User-Agent": "NewsAgent/0.1"}
        for node in self.nodes:
            url = f"{self.BASE_URL}/nodes/{node}/topics?p=1"
            try:
                async with session.get(url, headers=headers) as resp:
                    topics = await resp.json()
                for t in topics:
                    articles.append(
                        Article(
                            source=self.name,
                            title=t.get("title", ""),
                            url=t.get(
                                "url",
                                f"https://www.v2ex.com/t/{t.get('id', '')}",
                            ),
                            summary=t.get("content", "")[:500],
                            author=t.get("member", {}).get("username", ""),
                            comments_count=t.get("replies", 0),
                            tags=[node],
                            published_at=datetime.fromtimestamp(
                                t.get("created", 0), tz=timezone.utc
                            ),
                        )
                    )
            except Exception:
                continue
        return articles
