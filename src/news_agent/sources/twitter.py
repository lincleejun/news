from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from news_agent.models import Article
from news_agent.sources.base import BaseSource

logger = logging.getLogger(__name__)


class XSource(BaseSource):
    name = "x_com"
    timeout = 60

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.session_file = self.config.get("session_file", "data/x_session.json")

    async def fetch(self) -> list[Article]:
        try:
            return await self._fetch_with_playwright()
        except Exception as e:
            logger.error(f"[{self.name}] fetch failed: {e}")
            return []

    async def _fetch(self, session):
        return []

    async def _fetch_with_playwright(self) -> list[Article]:
        from playwright.async_api import async_playwright

        session_path = Path(self.session_file)
        if not session_path.exists():
            logger.warning(
                f"[{self.name}] No session file at {self.session_file}. "
                "Run `news-agent login-x` first."
            )
            return []

        storage_state = json.loads(session_path.read_text())

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(storage_state=storage_state)
            page = await context.new_page()

            await page.goto(
                "https://x.com/home", wait_until="networkidle", timeout=30000
            )
            await page.wait_for_timeout(3000)

            for _ in range(3):
                await page.evaluate("window.scrollBy(0, 1000)")
                await page.wait_for_timeout(1500)

            articles = []
            tweets = await page.query_selector_all('article[data-testid="tweet"]')

            for tweet in tweets[:30]:
                try:
                    text_el = await tweet.query_selector('[data-testid="tweetText"]')
                    text = await text_el.inner_text() if text_el else ""

                    user_el = await tweet.query_selector(
                        '[data-testid="User-Name"] a'
                    )
                    author, tweet_url = "", ""
                    if user_el:
                        author = await user_el.inner_text()
                        href = await user_el.get_attribute("href")
                        if href:
                            tweet_url = f"https://x.com{href}"

                    link_el = await tweet.query_selector('a[href*="http"]')
                    url = tweet_url
                    if link_el:
                        href = await link_el.get_attribute("href")
                        if href and "x.com" not in href:
                            url = href

                    likes = await self._get_metric(tweet, '[data-testid="like"]')
                    retweets = await self._get_metric(
                        tweet, '[data-testid="retweet"]'
                    )
                    replies = await self._get_metric(tweet, '[data-testid="reply"]')

                    if text:
                        title = text[:100] + ("..." if len(text) > 100 else "")
                        articles.append(
                            Article(
                                source=self.name,
                                title=title,
                                url=url or "https://x.com",
                                summary=text[:500],
                                author=author,
                                score=likes + retweets,
                                comments_count=replies,
                            )
                        )
                except Exception:
                    continue

            await browser.close()
            return articles

    async def _get_metric(self, tweet, selector: str) -> int:
        el = await tweet.query_selector(selector)
        if el:
            text = await el.inner_text()
            text = text.strip().replace(",", "")
            multiplier = 1
            if text.endswith("K"):
                multiplier = 1000
                text = text[:-1]
            elif text.endswith("M"):
                multiplier = 1_000_000
                text = text[:-1]
            try:
                return int(float(text) * multiplier)
            except (ValueError, TypeError):
                return 0
        return 0

    async def save_session(self) -> None:
        from playwright.async_api import async_playwright

        session_path = Path(self.session_file)
        session_path.parent.mkdir(parents=True, exist_ok=True)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()

            await page.goto("https://x.com/login")
            print("Please log in to X.com in the browser window.")
            print(
                "Press Enter here after you've logged in and see your timeline..."
            )
            input()

            storage = await context.storage_state()
            session_path.write_text(json.dumps(storage))
            print(f"Session saved to {self.session_file}")
            await browser.close()
