from __future__ import annotations

from typing import Any

import aiohttp
from bs4 import BeautifulSoup

from news_agent.models import Article
from news_agent.sources.base import BaseSource


class GitHubTrendingSource(BaseSource):
    name = "github"
    URL = "https://github.com/trending?since=daily"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

    async def _fetch(self, session: aiohttp.ClientSession) -> list[Article]:
        async with session.get(self.URL) as resp:
            html = await resp.text()
        soup = BeautifulSoup(html, "html.parser")
        articles = []
        for repo in soup.select("article.Box-row"):
            name_tag = repo.select_one("h2 a")
            if not name_tag:
                continue
            repo_path = name_tag["href"].strip("/")
            name = repo_path.replace("/", " / ")
            url = f"https://github.com/{repo_path}"
            desc_tag = repo.select_one("p")
            description = desc_tag.get_text(strip=True) if desc_tag else ""
            stars = 0
            star_tag = repo.select_one('a[href$="/stargazers"]')
            if star_tag:
                star_text = star_tag.get_text(strip=True).replace(",", "")
                try:
                    stars = int(star_text)
                except ValueError:
                    pass
            language = ""
            lang_tag = repo.select_one('[itemprop="programmingLanguage"]')
            if lang_tag:
                language = lang_tag.get_text(strip=True)
            articles.append(
                Article(
                    source=self.name,
                    title=name,
                    url=url,
                    summary=description,
                    score=stars,
                    tags=[language] if language else [],
                )
            )
        return articles
