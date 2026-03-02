from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import aiohttp

from news_agent.models import Article
from news_agent.sources.base import BaseSource

logger = logging.getLogger(__name__)

HF_DAILY_PAPERS_URL = "https://huggingface.co/api/daily_papers"
PWC_PAPERS_URL = "https://paperswithcode.com/api/v1/papers/"


class ArxivPapersSource(BaseSource):
    name = "arxiv_papers"
    timeout = 60

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.max_papers = self.config.get("max_papers", 10)

    async def _fetch(self, session: aiohttp.ClientSession) -> list[Article]:
        hf_papers = await self._fetch_huggingface(session)
        pwc_papers = await self._fetch_paperswithcode(session)

        # Merge: HF takes priority for duplicate arxiv IDs
        papers_by_id: dict[str, Article] = {}
        for article in pwc_papers:
            arxiv_id = self._extract_arxiv_id(article.url)
            if arxiv_id:
                papers_by_id[arxiv_id] = article

        for article in hf_papers:
            arxiv_id = self._extract_arxiv_id(article.url)
            if arxiv_id:
                papers_by_id[arxiv_id] = article

        merged = list(papers_by_id.values())
        merged.sort(key=lambda a: a.score, reverse=True)
        return merged[: self.max_papers]

    async def _fetch_huggingface(self, session: aiohttp.ClientSession) -> list[Article]:
        try:
            async with session.get(
                HF_DAILY_PAPERS_URL, params={"limit": 30}
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

            articles = []
            for paper in data:
                arxiv_id = paper.get("id", "") or paper.get("paper", {}).get("id", "")
                title = paper.get("title", "") or paper.get("paper", {}).get("title", "")
                summary = paper.get("summary", "") or paper.get("paper", {}).get("summary", "")
                published = paper.get("publishedAt", "") or paper.get("paper", {}).get("publishedAt", "")

                published_at = None
                if published:
                    try:
                        published_at = datetime.fromisoformat(
                            published.replace("Z", "+00:00")
                        )
                    except (ValueError, AttributeError):
                        pass

                articles.append(
                    Article(
                        source=self.name,
                        title=title,
                        url=f"https://arxiv.org/abs/{arxiv_id}",
                        summary=summary,
                        score=paper.get("upvotes", 0),
                        comments_count=paper.get("numComments", 0),
                        published_at=published_at,
                        tags=["arxiv", "ai"],
                    )
                )
            return articles
        except Exception as e:
            logger.warning(f"[{self.name}] HuggingFace fetch failed: {e}")
            return []

    async def _fetch_paperswithcode(
        self, session: aiohttp.ClientSession
    ) -> list[Article]:
        try:
            async with session.get(
                PWC_PAPERS_URL,
                params={"ordering": "-published", "items_per_page": 30, "page": 1},
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

            articles = []
            for paper in data.get("results", []):
                arxiv_id = paper.get("arxiv_id", "")
                if not arxiv_id:
                    continue

                published_at = None
                published = paper.get("published", "")
                if published:
                    try:
                        published_at = datetime.fromisoformat(
                            published.replace("Z", "+00:00")
                        )
                    except (ValueError, AttributeError):
                        pass

                url_abs = f"https://arxiv.org/abs/{arxiv_id}"

                articles.append(
                    Article(
                        source=self.name,
                        title=paper.get("title", ""),
                        url=url_abs,
                        summary=paper.get("abstract", ""),
                        published_at=published_at,
                        tags=["arxiv", "ai"],
                    )
                )
            return articles
        except Exception as e:
            logger.warning(f"[{self.name}] PapersWithCode fetch failed: {e}")
            return []

    @staticmethod
    def _extract_arxiv_id(url: str) -> str:
        """Extract arxiv ID from a URL like https://arxiv.org/abs/2401.12345."""
        if "arxiv.org/abs/" in url:
            return url.split("arxiv.org/abs/")[-1].split("?")[0].strip("/")
        return ""
