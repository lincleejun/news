from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import aiohttp

from news_agent.models import Article
from news_agent.sources.base import BaseSource


class RedditSource(BaseSource):
    name = "reddit"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.subreddits = self.config.get(
            "subreddits", ["technology", "programming"]
        )
        self.client_id = self.config.get("client_id", "")
        self.client_secret = self.config.get("client_secret", "")

    async def _fetch(self, session: aiohttp.ClientSession) -> list[Article]:
        token = await self._get_token(session)
        headers = {"Authorization": f"Bearer {token}", "User-Agent": "NewsAgent/0.1"}
        articles = []
        for subreddit in self.subreddits:
            url = f"https://oauth.reddit.com/r/{subreddit}/hot?limit=25"
            async with session.get(url, headers=headers) as resp:
                data = await resp.json()
            for post in data.get("data", {}).get("children", []):
                p = post["data"]
                articles.append(
                    Article(
                        source=self.name,
                        title=p["title"],
                        url=p["url"],
                        summary=p.get("selftext", "")[:500],
                        author=p.get("author", ""),
                        score=p.get("score", 0),
                        comments_count=p.get("num_comments", 0),
                        tags=[p.get("subreddit", "")],
                        published_at=datetime.fromtimestamp(
                            p.get("created_utc", 0), tz=timezone.utc
                        ),
                    )
                )
        return articles

    async def _get_token(self, session: aiohttp.ClientSession) -> str:
        auth = aiohttp.BasicAuth(self.client_id, self.client_secret)
        async with session.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=auth,
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": "NewsAgent/0.1"},
        ) as resp:
            data = await resp.json()
            return data["access_token"]
