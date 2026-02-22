# News Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a news aggregation agent that fetches from 7 tech/lifestyle source types, filters with DeepSeek LLM via LiteLLM, and pushes curated content via Email and Telegram. Runs on GitHub Actions schedule.

**Architecture:** Async concurrent fetching with asyncio + aiohttp, SQLite for persistence/dedup, DeepSeek via LiteLLM (OpenAI-compatible) for intelligent filtering. Runs as GitHub Actions scheduled workflow twice daily.

**Tech Stack:** Python 3.11+, uv, aiohttp, aiosqlite, litellm, playwright, feedparser, beautifulsoup4, jinja2, python-telegram-bot

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/news_agent/__init__.py`
- Create: `src/news_agent/models.py`
- Create: `config.example.yaml`
- Create: `.gitignore`
- Create: `tests/__init__.py`

**Step 1: Initialize git repo**

```bash
cd /Users/outman/workspace/personal/News
git init
```

**Step 2: Create pyproject.toml**

```toml
[project]
name = "news-agent"
version = "0.1.0"
description = "A news aggregation agent with LLM-based filtering"
requires-python = ">=3.11"
dependencies = [
    "aiohttp>=3.9",
    "aiosqlite>=0.20",
    "litellm>=1.0",
    "beautifulsoup4>=4.12",
    "feedparser>=6.0",
    "jinja2>=3.1",
    "playwright>=1.40",
    "pyyaml>=6.0",
    "python-telegram-bot>=21.0",
    "aiosmtplib>=3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
    "aioresponses>=0.7",
]

[project.scripts]
news-agent = "news_agent.main:cli_main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Step 3: Create .gitignore**

```
__pycache__/
*.pyc
.venv/
data/
config.yaml
.env
*.egg-info/
dist/
```

**Step 4: Create data model in `src/news_agent/models.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Article:
    source: str
    title: str
    url: str
    summary: str = ""
    author: str = ""
    published_at: datetime | None = None
    fetched_at: datetime = field(default_factory=datetime.now)
    score: int = 0
    comments_count: int = 0
    tags: list[str] = field(default_factory=list)
    llm_score: float = 0.0
    llm_reason: str = ""
    is_recommended: bool = False
    is_hot: bool = False
    sent: bool = False

    @property
    def id(self) -> str:
        """Unique identifier based on source and URL."""
        import hashlib
        return hashlib.sha256(f"{self.source}:{self.url}".encode()).hexdigest()[:16]
```

**Step 5: Create `src/news_agent/__init__.py`**

```python
"""News Agent - A news aggregation agent with LLM-based filtering."""
```

**Step 6: Create `config.example.yaml`**

```yaml
sources:
  hackernews:
    enabled: true
    max_items: 30
  reddit:
    enabled: true
    subreddits: ["technology", "programming", "MachineLearning"]
    client_id: ""
    client_secret: ""
  v2ex:
    enabled: true
    nodes: ["programmer", "create", "apple"]
  x_com:
    enabled: false
    session_file: "data/x_session.json"
  github:
    enabled: true
  wired:
    enabled: true
  rss:
    enabled: true
    feeds:
      - "https://simonwillison.net/atom/everything/"
      - "https://www.jeffgeerling.com/blog.xml"
      - "https://krebsonsecurity.com/feed/"
      - "https://daringfireball.net/feeds/main"
      - "https://pluralistic.net/feed/"
      - "https://lcamtuf.substack.com/feed"
      - "https://mitchellh.com/feed.xml"
      - "https://dynomight.net/feed.xml"
      - "https://xeiaso.net/blog.rss"
      - "https://devblogs.microsoft.com/oldnewthing/feed"
      - "https://www.righto.com/feeds/posts/default"
      - "https://lucumr.pocoo.org/feed.atom"
      - "https://garymarcus.substack.com/feed"
      - "https://rachelbythebay.com/w/atom.xml"
      - "https://overreacted.io/rss.xml"
      - "https://matklad.github.io/feed.xml"
      - "https://eli.thegreenplace.net/feeds/all.atom.xml"
      - "https://fabiensanglard.net/rss.xml"
      - "https://www.troyhunt.com/rss/"
      - "https://blog.miguelgrinberg.com/feed"
      - "https://computer.rip/rss.xml"
      - "https://www.tedunangst.com/flak/rss"
      - "https://feeds.arstechnica.com/arstechnica/index"
      - "https://www.theverge.com/rss/index.xml"

llm:
  model: "deepseek/deepseek-chat"
  api_base: "https://api.deepseek.com"

notifier:
  email:
    enabled: true
    smtp_host: "smtp.gmail.com"
    smtp_port: 587
    username: ""
    password: ""
    to: ""
  telegram:
    enabled: true
    bot_token: ""
    chat_id: ""

interests:
  - "AI/机器学习"
  - "编程/开发工具"
  - "硬件/数码"
  - "科技新闻"
  - "徒步/户外"
  - "摄影"
  - "国家地理/孤独星球"
```

**Step 7: Create tests/__init__.py (empty)**

**Step 8: Install dependencies**

```bash
cd /Users/outman/workspace/personal/News
uv sync
```

**Step 9: Write test for Article model**

Create `tests/test_models.py`:
```python
from news_agent.models import Article


def test_article_id_deterministic():
    a = Article(source="hackernews", title="Test", url="https://example.com")
    b = Article(source="hackernews", title="Test", url="https://example.com")
    assert a.id == b.id


def test_article_id_unique_across_sources():
    a = Article(source="hackernews", title="Test", url="https://example.com")
    b = Article(source="reddit", title="Test", url="https://example.com")
    assert a.id != b.id
```

**Step 10: Run tests**

```bash
uv run pytest tests/test_models.py -v
```
Expected: PASS

**Step 11: Commit**

```bash
git add .
git commit -m "feat: project scaffolding with data model and config"
```

---

### Task 2: Storage Layer (SQLite)

**Files:**
- Create: `src/news_agent/storage.py`
- Create: `tests/test_storage.py`

**Step 1: Write failing tests for storage**

Create `tests/test_storage.py`:
```python
import pytest
from news_agent.storage import Storage
from news_agent.models import Article


@pytest.fixture
async def storage(tmp_path):
    db_path = tmp_path / "test.db"
    s = Storage(str(db_path))
    await s.initialize()
    yield s
    await s.close()


async def test_save_and_exists(storage):
    article = Article(source="hackernews", title="Test", url="https://example.com")
    await storage.save_article(article)
    assert await storage.article_exists(article.id)


async def test_not_exists(storage):
    assert not await storage.article_exists("nonexistent")


async def test_get_unsent_articles(storage):
    a1 = Article(source="hn", title="A1", url="https://a1.com", llm_score=8.0, is_recommended=True)
    a2 = Article(source="hn", title="A2", url="https://a2.com", llm_score=5.0, is_recommended=False)
    await storage.save_article(a1)
    await storage.save_article(a2)
    unsent = await storage.get_unsent_recommended()
    assert len(unsent) == 1
    assert unsent[0].title == "A1"


async def test_mark_as_sent(storage):
    a = Article(source="hn", title="A1", url="https://a1.com", is_recommended=True)
    await storage.save_article(a)
    await storage.mark_sent([a.id])
    unsent = await storage.get_unsent_recommended()
    assert len(unsent) == 0


async def test_get_previous_score(storage):
    a = Article(source="hn", title="A1", url="https://a1.com", score=10)
    await storage.save_article(a)
    prev = await storage.get_previous_score(a.url)
    assert prev == 10
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_storage.py -v
```
Expected: FAIL (module not found)

**Step 3: Implement storage.py**

```python
from __future__ import annotations

import json
from datetime import datetime

import aiosqlite

from news_agent.models import Article


class Storage:
    def __init__(self, db_path: str = "data/news.db"):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                url TEXT NOT NULL,
                summary TEXT DEFAULT '',
                author TEXT DEFAULT '',
                published_at TEXT,
                fetched_at TEXT NOT NULL,
                score INTEGER DEFAULT 0,
                comments_count INTEGER DEFAULT 0,
                tags TEXT DEFAULT '[]',
                llm_score REAL DEFAULT 0.0,
                llm_reason TEXT DEFAULT '',
                is_recommended INTEGER DEFAULT 0,
                is_hot INTEGER DEFAULT 0,
                sent INTEGER DEFAULT 0
            )
        """)
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_url ON articles(url)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_sent ON articles(sent, is_recommended)"
        )
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def save_article(self, article: Article) -> None:
        await self._db.execute(
            """INSERT OR REPLACE INTO articles
            (id, source, title, url, summary, author, published_at, fetched_at,
             score, comments_count, tags, llm_score, llm_reason,
             is_recommended, is_hot, sent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                article.id,
                article.source,
                article.title,
                article.url,
                article.summary,
                article.author,
                article.published_at.isoformat() if article.published_at else None,
                article.fetched_at.isoformat(),
                article.score,
                article.comments_count,
                json.dumps(article.tags),
                article.llm_score,
                article.llm_reason,
                int(article.is_recommended),
                int(article.is_hot),
                int(article.sent),
            ),
        )
        await self._db.commit()

    async def save_articles(self, articles: list[Article]) -> None:
        for article in articles:
            await self.save_article(article)

    async def article_exists(self, article_id: str) -> bool:
        cursor = await self._db.execute(
            "SELECT 1 FROM articles WHERE id = ?", (article_id,)
        )
        return await cursor.fetchone() is not None

    async def get_unsent_recommended(self) -> list[Article]:
        cursor = await self._db.execute(
            """SELECT * FROM articles
            WHERE is_recommended = 1 AND sent = 0
            ORDER BY llm_score DESC, score DESC""",
        )
        rows = await cursor.fetchall()
        return [self._row_to_article(row) for row in rows]

    async def mark_sent(self, article_ids: list[str]) -> None:
        placeholders = ",".join("?" for _ in article_ids)
        await self._db.execute(
            f"UPDATE articles SET sent = 1 WHERE id IN ({placeholders})",
            article_ids,
        )
        await self._db.commit()

    async def get_previous_score(self, url: str) -> int | None:
        cursor = await self._db.execute(
            "SELECT score FROM articles WHERE url = ? ORDER BY fetched_at DESC LIMIT 1",
            (url,),
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    def _row_to_article(self, row) -> Article:
        return Article(
            source=row[1],
            title=row[2],
            url=row[3],
            summary=row[4],
            author=row[5],
            published_at=datetime.fromisoformat(row[6]) if row[6] else None,
            fetched_at=datetime.fromisoformat(row[7]),
            score=row[8],
            comments_count=row[9],
            tags=json.loads(row[10]),
            llm_score=row[11],
            llm_reason=row[12],
            is_recommended=bool(row[13]),
            is_hot=bool(row[14]),
            sent=bool(row[15]),
        )
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_storage.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/news_agent/storage.py tests/test_storage.py
git commit -m "feat: add SQLite storage layer with dedup and history"
```

---

### Task 3: Base Source & Hacker News Source

**Files:**
- Create: `src/news_agent/sources/__init__.py`
- Create: `src/news_agent/sources/base.py`
- Create: `src/news_agent/sources/hackernews.py`
- Create: `tests/test_sources/__init__.py`
- Create: `tests/test_sources/test_hackernews.py`

**Step 1: Write base source class**

`src/news_agent/sources/__init__.py`: empty file

`src/news_agent/sources/base.py`:
```python
from __future__ import annotations

import abc
import asyncio
import logging
from typing import Any

import aiohttp

from news_agent.models import Article

logger = logging.getLogger(__name__)


class BaseSource(abc.ABC):
    name: str = "base"
    timeout: int = 30

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    async def fetch(self) -> list[Article]:
        try:
            async with aiohttp.ClientSession() as session:
                return await asyncio.wait_for(
                    self._fetch(session), timeout=self.timeout
                )
        except asyncio.TimeoutError:
            logger.warning(f"[{self.name}] fetch timed out after {self.timeout}s")
            return []
        except Exception as e:
            logger.error(f"[{self.name}] fetch failed: {e}")
            return []

    @abc.abstractmethod
    async def _fetch(self, session: aiohttp.ClientSession) -> list[Article]:
        ...
```

**Step 2: Write failing test for Hacker News source**

`tests/test_sources/__init__.py`: empty file

`tests/test_sources/test_hackernews.py`:
```python
from aioresponses import aioresponses

from news_agent.sources.hackernews import HackerNewsSource


async def test_hackernews_fetch():
    source = HackerNewsSource({"max_items": 2})

    with aioresponses() as mocked:
        mocked.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json",
            payload=[101, 102, 103],
        )
        mocked.get(
            "https://hacker-news.firebaseio.com/v0/item/101.json",
            payload={
                "id": 101,
                "title": "Show HN: Cool Project",
                "url": "https://cool.com",
                "by": "user1",
                "score": 150,
                "descendants": 42,
                "time": 1708500000,
                "type": "story",
            },
        )
        mocked.get(
            "https://hacker-news.firebaseio.com/v0/item/102.json",
            payload={
                "id": 102,
                "title": "Ask HN: Best editor?",
                "url": "",
                "by": "user2",
                "score": 80,
                "descendants": 100,
                "time": 1708500100,
                "type": "story",
            },
        )

        articles = await source.fetch()

    assert len(articles) == 2
    assert articles[0].source == "hackernews"
    assert articles[0].title == "Show HN: Cool Project"
    assert articles[0].score == 150
```

**Step 3: Run test to verify it fails**

```bash
uv run pytest tests/test_sources/test_hackernews.py -v
```
Expected: FAIL

**Step 4: Implement Hacker News source**

`src/news_agent/sources/hackernews.py`:
```python
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

        articles = []
        for r in results:
            if isinstance(r, Article):
                articles.append(r)
        return articles

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
            published_at=datetime.fromtimestamp(
                data.get("time", 0), tz=timezone.utc
            ),
        )
```

**Step 5: Run tests**

```bash
uv run pytest tests/test_sources/test_hackernews.py -v
```
Expected: PASS

**Step 6: Commit**

```bash
git add src/news_agent/sources/ tests/test_sources/
git commit -m "feat: add base source class and Hacker News source"
```

---

### Task 4: Reddit Source

**Files:**
- Create: `src/news_agent/sources/reddit.py`
- Create: `tests/test_sources/test_reddit.py`

**Step 1: Write failing test**

`tests/test_sources/test_reddit.py`:
```python
from aioresponses import aioresponses

from news_agent.sources.reddit import RedditSource


async def test_reddit_fetch():
    source = RedditSource({
        "subreddits": ["technology"],
        "client_id": "test_id",
        "client_secret": "test_secret",
    })

    with aioresponses() as mocked:
        mocked.post(
            "https://www.reddit.com/api/v1/access_token",
            payload={"access_token": "test_token", "token_type": "bearer"},
        )
        mocked.get(
            "https://oauth.reddit.com/r/technology/hot?limit=25",
            payload={
                "data": {
                    "children": [
                        {
                            "data": {
                                "id": "abc123",
                                "title": "New AI Breakthrough",
                                "url": "https://ai-news.com/article",
                                "author": "techuser",
                                "score": 500,
                                "num_comments": 200,
                                "created_utc": 1708500000,
                                "selftext": "This is a summary",
                                "subreddit": "technology",
                            }
                        }
                    ]
                }
            },
        )

        articles = await source.fetch()

    assert len(articles) == 1
    assert articles[0].source == "reddit"
    assert articles[0].title == "New AI Breakthrough"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_sources/test_reddit.py -v
```

**Step 3: Implement Reddit source**

`src/news_agent/sources/reddit.py`:
```python
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
        self.subreddits = self.config.get("subreddits", ["technology", "programming"])
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
                articles.append(Article(
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
                ))
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
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_sources/test_reddit.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/news_agent/sources/reddit.py tests/test_sources/test_reddit.py
git commit -m "feat: add Reddit source with OAuth"
```

---

### Task 5: V2EX Source

**Files:**
- Create: `src/news_agent/sources/v2ex.py`
- Create: `tests/test_sources/test_v2ex.py`

**Step 1: Write failing test**

`tests/test_sources/test_v2ex.py`:
```python
from aioresponses import aioresponses

from news_agent.sources.v2ex import V2exSource


async def test_v2ex_fetch():
    source = V2exSource({"nodes": ["programmer"]})

    with aioresponses() as mocked:
        mocked.get(
            "https://www.v2ex.com/api/v2/nodes/programmer/topics?p=1",
            payload=[
                {
                    "id": 12345,
                    "title": "Python最佳实践",
                    "url": "https://www.v2ex.com/t/12345",
                    "content": "分享一些经验",
                    "member": {"username": "dev123"},
                    "created": 1708500000,
                    "replies": 30,
                }
            ],
        )

        articles = await source.fetch()

    assert len(articles) == 1
    assert articles[0].source == "v2ex"
    assert articles[0].title == "Python最佳实践"
```

**Step 2: Run test to verify fail**

```bash
uv run pytest tests/test_sources/test_v2ex.py -v
```

**Step 3: Implement V2EX source**

`src/news_agent/sources/v2ex.py`:
```python
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
                    articles.append(Article(
                        source=self.name,
                        title=t.get("title", ""),
                        url=t.get("url", f"https://www.v2ex.com/t/{t.get('id', '')}"),
                        summary=t.get("content", "")[:500],
                        author=t.get("member", {}).get("username", ""),
                        comments_count=t.get("replies", 0),
                        tags=[node],
                        published_at=datetime.fromtimestamp(
                            t.get("created", 0), tz=timezone.utc
                        ),
                    ))
            except Exception:
                continue
        return articles
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_sources/test_v2ex.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/news_agent/sources/v2ex.py tests/test_sources/test_v2ex.py
git commit -m "feat: add V2EX source"
```

---

### Task 6: RSS, Wired & GitHub Trending Sources

**Files:**
- Create: `src/news_agent/sources/rss.py`
- Create: `src/news_agent/sources/wired.py`
- Create: `src/news_agent/sources/github_trending.py`
- Create: `tests/test_sources/test_rss.py`
- Create: `tests/test_sources/test_github.py`

**Step 1: Write RSS source test**

`tests/test_sources/test_rss.py`:
```python
from unittest.mock import patch, MagicMock

from news_agent.sources.rss import RssSource


async def test_rss_fetch():
    source = RssSource({"feeds": ["https://example.com/feed.xml"]})

    mock_feed = MagicMock()
    mock_feed.entries = [
        MagicMock(
            title="RSS Article",
            link="https://example.com/article1",
            get=lambda k, d="": {"summary": "A summary", "author": "Author"}.get(k, d),
            published_parsed=(2026, 2, 21, 8, 0, 0, 0, 0, 0),
        )
    ]

    with patch("news_agent.sources.rss.feedparser.parse", return_value=mock_feed):
        articles = await source.fetch()

    assert len(articles) == 1
    assert articles[0].source == "rss"
    assert articles[0].title == "RSS Article"
```

**Step 2: Run test to verify fail**

```bash
uv run pytest tests/test_sources/test_rss.py -v
```

**Step 3: Implement RSS source**

`src/news_agent/sources/rss.py`:
```python
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
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        published = datetime.fromtimestamp(
                            time.mktime(entry.published_parsed), tz=timezone.utc
                        )
                    articles.append(Article(
                        source=self.name,
                        title=entry.title,
                        url=entry.link,
                        summary=entry.get("summary", "")[:500],
                        author=entry.get("author", ""),
                        published_at=published,
                    ))
            except Exception:
                continue
        return articles
```

**Step 4: Implement Wired source (thin wrapper over RSS)**

`src/news_agent/sources/wired.py`:
```python
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
```

**Step 5: Write GitHub trending test**

`tests/test_sources/test_github.py`:
```python
from aioresponses import aioresponses

from news_agent.sources.github_trending import GitHubTrendingSource


async def test_github_trending_fetch():
    source = GitHubTrendingSource()

    html = """
    <article class="Box-row">
        <h2 class="h3 lh-condensed">
            <a href="/user/repo" data-view-component="true">
                <span>user</span> / <span>repo</span>
            </a>
        </h2>
        <p class="col-9 color-fg-muted my-1 pr-4">A cool project description</p>
        <div class="f6 color-fg-muted mt-2">
            <a class="Link d-inline-block mr-3" href="/user/repo/stargazers">
                1,234
            </a>
        </div>
    </article>
    """

    with aioresponses() as mocked:
        mocked.get("https://github.com/trending?since=daily", body=html)
        articles = await source.fetch()

    assert len(articles) >= 1
    assert articles[0].source == "github"
    assert "repo" in articles[0].title
```

**Step 6: Implement GitHub trending source**

`src/news_agent/sources/github_trending.py`:
```python
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

            articles.append(Article(
                source=self.name,
                title=name,
                url=url,
                summary=description,
                score=stars,
                tags=[language] if language else [],
            ))

        return articles
```

**Step 7: Run all source tests**

```bash
uv run pytest tests/test_sources/ -v
```
Expected: PASS

**Step 8: Commit**

```bash
git add src/news_agent/sources/rss.py src/news_agent/sources/wired.py src/news_agent/sources/github_trending.py tests/test_sources/
git commit -m "feat: add RSS, Wired, and GitHub Trending sources"
```

---

### Task 7: X.com Source (Playwright)

**Files:**
- Create: `src/news_agent/sources/twitter.py`

**Step 1: Install Playwright browsers**

```bash
uv run playwright install chromium
```

**Step 2: Implement X.com source**

`src/news_agent/sources/twitter.py`:
```python
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
        """Override base fetch to use Playwright instead of aiohttp."""
        try:
            return await self._fetch_with_playwright()
        except Exception as e:
            logger.error(f"[{self.name}] fetch failed: {e}")
            return []

    async def _fetch(self, session):
        """Not used - overridden by fetch()."""
        return []

    async def _fetch_with_playwright(self) -> list[Article]:
        from playwright.async_api import async_playwright

        session_path = Path(self.session_file)
        if not session_path.exists():
            logger.warning(
                f"[{self.name}] No session file found at {self.session_file}. "
                "Run `news-agent login-x` to set up X.com session."
            )
            return []

        storage_state = json.loads(session_path.read_text())

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(storage_state=storage_state)
            page = await context.new_page()

            await page.goto("https://x.com/home", wait_until="networkidle", timeout=30000)
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

                    user_el = await tweet.query_selector('[data-testid="User-Name"] a')
                    author = ""
                    tweet_url = ""
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
                    retweets = await self._get_metric(tweet, '[data-testid="retweet"]')
                    replies = await self._get_metric(tweet, '[data-testid="reply"]')

                    if text:
                        title = text[:100] + ("..." if len(text) > 100 else "")
                        articles.append(Article(
                            source=self.name,
                            title=title,
                            url=url or "https://x.com",
                            summary=text[:500],
                            author=author,
                            score=likes + retweets,
                            comments_count=replies,
                        ))
                except Exception:
                    continue

            await browser.close()
            return articles

    async def _get_metric(self, tweet, selector: str) -> int:
        el = await tweet.query_selector(selector)
        if el:
            text = await el.inner_text()
            text = text.strip().replace(",", "").replace("K", "000").replace("M", "000000")
            try:
                return int(float(text))
            except (ValueError, TypeError):
                return 0
        return 0

    async def save_session(self) -> None:
        """Interactive login to save session cookies."""
        from playwright.async_api import async_playwright

        session_path = Path(self.session_file)
        session_path.parent.mkdir(parents=True, exist_ok=True)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()

            await page.goto("https://x.com/login")
            print("Please log in to X.com in the browser window.")
            print("Press Enter here after you've logged in and see your timeline...")
            input()

            storage = await context.storage_state()
            session_path.write_text(json.dumps(storage))
            print(f"Session saved to {self.session_file}")

            await browser.close()
```

**Step 3: Commit** (no automated test for Playwright browser interactions)

```bash
git add src/news_agent/sources/twitter.py
git commit -m "feat: add X.com source via Playwright browser automation"
```

---

### Task 8: LLM Filter (DeepSeek via LiteLLM)

**Files:**
- Create: `src/news_agent/filter.py`
- Create: `tests/test_filter.py`

**Step 1: Write failing test**

`tests/test_filter.py`:
```python
import json
from unittest.mock import AsyncMock, patch, MagicMock

from news_agent.filter import LLMFilter
from news_agent.models import Article


async def test_filter_scores_articles():
    articles = [
        Article(source="hn", title="New AI Model Released", url="https://ai.com", summary="Big breakthrough"),
        Article(source="hn", title="Cat Video Goes Viral", url="https://cats.com", summary="Funny cat"),
    ]

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps([
        {"index": 0, "score": 9.0, "reason": "Major AI development", "is_hot": True},
        {"index": 1, "score": 3.0, "reason": "Not tech related", "is_hot": False},
    ])))]

    with patch("news_agent.filter.litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
        f = LLMFilter(config={
            "model": "deepseek/deepseek-chat",
            "interests": ["AI/机器学习"],
        })
        result = await f.filter_articles(articles)

    assert result[0].llm_score == 9.0
    assert result[0].is_recommended is True
    assert result[0].is_hot is True
    assert result[1].llm_score == 3.0
    assert result[1].is_recommended is False
```

**Step 2: Run test to verify fail**

```bash
uv run pytest tests/test_filter.py -v
```

**Step 3: Implement LLM filter using LiteLLM**

`src/news_agent/filter.py`:
```python
from __future__ import annotations

import json
import logging
from typing import Any

import litellm

from news_agent.models import Article

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个科技资讯筛选助手。你的任务是对一批文章进行评分和筛选。

用户关注的领域：
{interests}

评分标准 (0-10):
- 9-10: 重大突破、行业变革、必读内容
- 7-8: 有价值的信息、值得了解
- 5-6: 一般资讯、可看可不看
- 3-4: 相关性较低
- 0-2: 无关或低质量内容

请对每篇文章输出 JSON 数组，每个元素包含:
- index: 文章序号 (从0开始)
- score: 评分 (0-10, 浮点数)
- reason: 推荐理由 (一句话, 中文)
- is_hot: 是否高热度/高潜力 (布尔值)

只输出 JSON 数组，不要其他内容。"""


class LLMFilter:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.model = self.config.get("model", "deepseek/deepseek-chat")
        self.interests = self.config.get("interests", [])
        self.recommend_threshold = self.config.get("recommend_threshold", 7.0)
        self.api_base = self.config.get("api_base", None)

    async def filter_articles(
        self, articles: list[Article], batch_size: int = 25
    ) -> list[Article]:
        all_results = []
        for i in range(0, len(articles), batch_size):
            batch = articles[i : i + batch_size]
            scored = await self._score_batch(batch)
            all_results.extend(scored)
        return all_results

    async def _score_batch(self, articles: list[Article]) -> list[Article]:
        articles_text = "\n\n".join(
            f"[{i}] 来源: {a.source} | 标题: {a.title} | 摘要: {a.summary} | "
            f"热度: {a.score} | 评论数: {a.comments_count}"
            for i, a in enumerate(articles)
        )

        system = SYSTEM_PROMPT.format(interests="\n".join(f"- {i}" for i in self.interests))

        try:
            kwargs = {
                "model": self.model,
                "max_tokens": 4096,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": articles_text},
                ],
            }
            if self.api_base:
                kwargs["api_base"] = self.api_base

            response = await litellm.acompletion(**kwargs)

            content = response.choices[0].message.content
            scores = json.loads(content)

            for item in scores:
                idx = item["index"]
                if 0 <= idx < len(articles):
                    articles[idx].llm_score = item["score"]
                    articles[idx].llm_reason = item["reason"]
                    articles[idx].is_hot = item.get("is_hot", False)
                    articles[idx].is_recommended = item["score"] >= self.recommend_threshold

        except Exception as e:
            logger.error(f"LLM filtering failed: {e}")

        return articles
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_filter.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/news_agent/filter.py tests/test_filter.py
git commit -m "feat: add LLM-based article filtering with DeepSeek via LiteLLM"
```

---

### Task 9: Email Notifier

**Files:**
- Create: `src/news_agent/notifier/__init__.py`
- Create: `src/news_agent/notifier/email.py`
- Create: `src/news_agent/templates/email.html`
- Create: `tests/test_notifier/__init__.py`
- Create: `tests/test_notifier/test_email.py`

**Step 1: Write failing test**

`tests/test_notifier/__init__.py`: empty file

`tests/test_notifier/test_email.py`:
```python
from unittest.mock import patch, MagicMock, AsyncMock

from news_agent.notifier.email import EmailNotifier
from news_agent.models import Article


async def test_email_send():
    articles = [
        Article(
            source="hackernews",
            title="AI News",
            url="https://ai.com",
            summary="Big news",
            llm_score=9.0,
            llm_reason="重大AI进展",
            is_hot=True,
            is_recommended=True,
        ),
    ]

    notifier = EmailNotifier({
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "username": "test@gmail.com",
        "password": "pass",
        "to": "me@gmail.com",
    })

    with patch("news_agent.notifier.email.aiosmtplib", new_callable=MagicMock) as mock_smtp:
        mock_smtp.send = AsyncMock()
        await notifier.send(articles)
        mock_smtp.send.assert_called_once()
```

**Step 2: Run test to verify fail**

```bash
uv run pytest tests/test_notifier/test_email.py -v
```

**Step 3: Create email HTML template**

Create dir: `src/news_agent/templates/`

`src/news_agent/templates/email.html`:
```html
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 680px; margin: 0 auto; padding: 20px; color: #333; }
h1 { color: #1a1a1a; border-bottom: 2px solid #0066cc; padding-bottom: 10px; }
.article { margin: 16px 0; padding: 12px; border-left: 3px solid #ddd; }
.article.hot { border-left-color: #ff4444; background: #fff5f5; }
.source { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 12px; background: #e8e8e8; color: #666; }
.title a { color: #0066cc; text-decoration: none; font-weight: 600; }
.title a:hover { text-decoration: underline; }
.reason { color: #666; font-size: 14px; margin-top: 4px; }
.score { color: #999; font-size: 12px; }
.hot-badge { color: #ff4444; font-weight: bold; }
.footer { margin-top: 30px; padding-top: 10px; border-top: 1px solid #eee; color: #999; font-size: 12px; }
</style>
</head>
<body>
<h1>{{ title }}</h1>
<p>{{ date }} | 共 {{ articles|length }} 条精选</p>
{% for article in articles %}
<div class="article {{ 'hot' if article.is_hot else '' }}">
    {% if article.is_hot %}<span class="hot-badge">🔥 热门</span>{% endif %}
    <span class="source">{{ article.source }}</span>
    <div class="title"><a href="{{ article.url }}">{{ article.title }}</a></div>
    <div class="reason">{{ article.llm_reason }}</div>
    <div class="score">评分: {{ article.llm_score }} | 热度: {{ article.score }}</div>
</div>
{% endfor %}
<div class="footer">由 News Agent 自动生成</div>
</body>
</html>
```

**Step 4: Implement email notifier**

`src/news_agent/notifier/__init__.py`:
```python
"""Notification modules."""
```

`src/news_agent/notifier/email.py`:
```python
from __future__ import annotations

import logging
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

import aiosmtplib
from jinja2 import Template

from news_agent.models import Article

logger = logging.getLogger(__name__)

TEMPLATE_PATH = Path(__file__).parent.parent / "templates" / "email.html"


class EmailNotifier:
    def __init__(self, config: dict[str, Any]):
        self.smtp_host = config["smtp_host"]
        self.smtp_port = config.get("smtp_port", 587)
        self.username = config["username"]
        self.password = config["password"]
        self.to_addr = config["to"]

    async def send(self, articles: list[Article]) -> None:
        now = datetime.now()
        period = "早间" if now.hour < 14 else "晚间"
        subject = f"[科技日报] {now.strftime('%Y-%m-%d')} {period}精选"

        template = Template(TEMPLATE_PATH.read_text())
        sorted_articles = sorted(articles, key=lambda a: (a.is_hot, a.llm_score), reverse=True)

        html = template.render(
            title=subject,
            date=now.strftime("%Y-%m-%d %H:%M"),
            articles=sorted_articles,
        )

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.username
        msg["To"] = self.to_addr
        msg.attach(MIMEText(html, "html"))

        await aiosmtplib.send(
            msg,
            hostname=self.smtp_host,
            port=self.smtp_port,
            username=self.username,
            password=self.password,
            start_tls=True,
        )
        logger.info(f"Email sent to {self.to_addr}: {subject}")
```

**Step 5: Run tests**

```bash
uv run pytest tests/test_notifier/test_email.py -v
```
Expected: PASS

**Step 6: Commit**

```bash
git add src/news_agent/notifier/ src/news_agent/templates/ tests/test_notifier/
git commit -m "feat: add email notifier with HTML template"
```

---

### Task 10: Telegram Notifier

**Files:**
- Create: `src/news_agent/notifier/telegram.py`
- Create: `tests/test_notifier/test_telegram.py`

**Step 1: Write failing test**

`tests/test_notifier/test_telegram.py`:
```python
from unittest.mock import AsyncMock, patch, MagicMock

from news_agent.notifier.telegram import TelegramNotifier
from news_agent.models import Article


async def test_telegram_send():
    articles = [
        Article(
            source="hackernews",
            title="AI News",
            url="https://ai.com",
            llm_score=9.0,
            llm_reason="重大AI进展",
            is_hot=True,
            is_recommended=True,
        ),
    ]

    notifier = TelegramNotifier({
        "bot_token": "123:ABC",
        "chat_id": "456",
    })

    with patch("news_agent.notifier.telegram.Bot") as MockBot:
        mock_bot = MockBot.return_value
        mock_bot.send_message = AsyncMock()
        await notifier.send(articles)
        mock_bot.send_message.assert_called_once()
```

**Step 2: Run test to verify fail**

```bash
uv run pytest tests/test_notifier/test_telegram.py -v
```

**Step 3: Implement Telegram notifier**

`src/news_agent/notifier/telegram.py`:
```python
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from telegram import Bot

from news_agent.models import Article

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, config: dict[str, Any]):
        self.bot_token = config["bot_token"]
        self.chat_id = config["chat_id"]

    async def send(self, articles: list[Article]) -> None:
        bot = Bot(token=self.bot_token)

        now = datetime.now()
        period = "早间" if now.hour < 14 else "晚间"

        sorted_articles = sorted(
            articles, key=lambda a: (a.is_hot, a.llm_score), reverse=True
        )

        lines = [f"📰 *科技日报 {now.strftime('%Y-%m-%d')} {period}精选*\n"]

        for i, a in enumerate(sorted_articles, 1):
            hot = "🔥 " if a.is_hot else ""
            source = f"`{a.source}`"
            title = a.title.replace("[", "\\[").replace("]", "\\]")
            lines.append(
                f"{i}\\. {hot}{source} [{title}]({a.url})\n"
                f"   _{a.llm_reason}_"
            )

        lines.append(f"\n_共 {len(articles)} 条精选_")
        text = "\n\n".join(lines)

        await bot.send_message(
            chat_id=self.chat_id,
            text=text,
            parse_mode="MarkdownV2",
            disable_web_page_preview=True,
        )
        logger.info(f"Telegram message sent to chat {self.chat_id}")
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_notifier/test_telegram.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/news_agent/notifier/telegram.py tests/test_notifier/test_telegram.py
git commit -m "feat: add Telegram notifier"
```

---

### Task 11: Main Orchestrator

**Files:**
- Create: `src/news_agent/main.py`
- Create: `tests/test_main.py`

**Step 1: Write test for main orchestrator**

`tests/test_main.py`:
```python
from unittest.mock import AsyncMock, patch, MagicMock

from news_agent.main import run_agent
from news_agent.models import Article


async def test_run_agent_orchestrates_pipeline():
    mock_article = Article(
        source="hn",
        title="Test",
        url="https://test.com",
        llm_score=8.0,
        is_recommended=True,
    )

    mock_source = AsyncMock()
    mock_source.fetch = AsyncMock(return_value=[mock_article])
    mock_source.name = "test"

    mock_storage = AsyncMock()
    mock_storage.article_exists = AsyncMock(return_value=False)
    mock_storage.get_previous_score = AsyncMock(return_value=None)

    mock_filter = AsyncMock()
    mock_filter.filter_articles = AsyncMock(return_value=[mock_article])

    mock_email = AsyncMock()
    mock_telegram = AsyncMock()

    with patch("news_agent.main.load_config") as mock_cfg, \
         patch("news_agent.main.create_sources", return_value=[mock_source]), \
         patch("news_agent.main.Storage", return_value=mock_storage), \
         patch("news_agent.main.LLMFilter", return_value=mock_filter), \
         patch("news_agent.main.EmailNotifier", return_value=mock_email), \
         patch("news_agent.main.TelegramNotifier", return_value=mock_telegram):

        mock_cfg.return_value = {
            "sources": {},
            "llm": {"model": "deepseek/deepseek-chat"},
            "notifier": {
                "email": {"enabled": True, "smtp_host": "h", "smtp_port": 587, "username": "u", "password": "p", "to": "t"},
                "telegram": {"enabled": True, "bot_token": "t", "chat_id": "c"},
            },
            "interests": ["AI"],
        }

        await run_agent()

    mock_filter.filter_articles.assert_called_once()
    mock_email.send.assert_called_once()
    mock_telegram.send.assert_called_once()
    mock_storage.mark_sent.assert_called_once()
```

**Step 2: Run test to verify fail**

```bash
uv run pytest tests/test_main.py -v
```

**Step 3: Implement main.py**

`src/news_agent/main.py`:
```python
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from news_agent.filter import LLMFilter
from news_agent.models import Article
from news_agent.notifier.email import EmailNotifier
from news_agent.notifier.telegram import TelegramNotifier
from news_agent.storage import Storage

logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        logger.error(f"Config file not found: {path}")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_sources(config: dict[str, Any]) -> list:
    from news_agent.sources.hackernews import HackerNewsSource
    from news_agent.sources.reddit import RedditSource
    from news_agent.sources.v2ex import V2exSource
    from news_agent.sources.github_trending import GitHubTrendingSource
    from news_agent.sources.rss import RssSource
    from news_agent.sources.wired import WiredSource
    from news_agent.sources.twitter import XSource

    source_map = {
        "hackernews": HackerNewsSource,
        "reddit": RedditSource,
        "v2ex": V2exSource,
        "github": GitHubTrendingSource,
        "rss": RssSource,
        "wired": WiredSource,
        "x_com": XSource,
    }

    sources = []
    source_configs = config.get("sources", {})
    for name, cls in source_map.items():
        src_cfg = source_configs.get(name, {})
        if src_cfg.get("enabled", True):
            sources.append(cls(src_cfg))
    return sources


async def run_agent(config_path: str = "config.yaml") -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(config_path)

    storage = Storage()
    await storage.initialize()

    try:
        sources = create_sources(config)
        logger.info(f"Fetching from {len(sources)} sources...")

        fetch_tasks = [source.fetch() for source in sources]
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        all_articles: list[Article] = []
        for i, result in enumerate(results):
            if isinstance(result, list):
                logger.info(f"  [{sources[i].name}] fetched {len(result)} articles")
                all_articles.extend(result)
            else:
                logger.error(f"  [{sources[i].name}] failed: {result}")

        logger.info(f"Total fetched: {len(all_articles)} articles")

        # Dedup
        unique_articles = []
        seen_urls: set[str] = set()
        for article in all_articles:
            if article.url in seen_urls:
                continue
            seen_urls.add(article.url)

            exists = await storage.article_exists(article.id)
            if exists:
                prev_score = await storage.get_previous_score(article.url)
                if prev_score and article.score >= prev_score * 3:
                    unique_articles.append(article)
                continue
            unique_articles.append(article)

        logger.info(f"After dedup: {len(unique_articles)} articles")

        if not unique_articles:
            logger.info("No new articles to process.")
            return

        # LLM Filter
        llm_filter = LLMFilter(config={
            **config.get("llm", {}),
            "interests": config.get("interests", []),
        })
        scored_articles = await llm_filter.filter_articles(unique_articles)

        await storage.save_articles(scored_articles)

        recommended = [a for a in scored_articles if a.is_recommended]
        recommended.sort(key=lambda a: (a.is_hot, a.llm_score), reverse=True)
        recommended = recommended[:10]

        logger.info(f"Recommended: {len(recommended)} articles")

        if not recommended:
            logger.info("No articles passed the quality threshold.")
            return

        # Push notifications
        notifier_config = config.get("notifier", {})
        push_tasks = []

        email_cfg = notifier_config.get("email", {})
        if email_cfg.get("enabled"):
            email = EmailNotifier(email_cfg)
            push_tasks.append(email.send(recommended))

        tg_cfg = notifier_config.get("telegram", {})
        if tg_cfg.get("enabled"):
            telegram = TelegramNotifier(tg_cfg)
            push_tasks.append(telegram.send(recommended))

        if push_tasks:
            await asyncio.gather(*push_tasks)
            await storage.mark_sent([a.id for a in recommended])
            logger.info("Notifications sent successfully!")

    finally:
        await storage.close()


def cli_main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "login-x":
        from news_agent.sources.twitter import XSource
        source = XSource()
        asyncio.run(source.save_session())
    else:
        config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
        asyncio.run(run_agent(config_path))


if __name__ == "__main__":
    cli_main()
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_main.py -v
```
Expected: PASS

**Step 5: Run all tests**

```bash
uv run pytest -v
```
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/news_agent/main.py tests/test_main.py
git commit -m "feat: add main orchestrator with full pipeline"
```

---

### Task 12: GitHub Actions Workflow

**Files:**
- Create: `.github/workflows/news-agent.yml`
- Create: `.github/workflows/tests.yml`

**Step 1: Create the scheduled workflow**

`.github/workflows/news-agent.yml`:
```yaml
name: News Agent

on:
  schedule:
    # Run at 00:00 and 12:00 UTC (08:00 and 20:00 CST)
    - cron: '0 0 * * *'
    - cron: '0 12 * * *'
  workflow_dispatch:  # Allow manual trigger

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Set up Python
        run: uv python install 3.12

      - name: Install dependencies
        run: uv sync

      - name: Restore SQLite database
        uses: actions/cache@v4
        with:
          path: data/news.db
          key: news-db-${{ github.run_id }}
          restore-keys: |
            news-db-

      - name: Create config from secrets
        run: |
          cat > config.yaml << 'HEREDOC'
          sources:
            hackernews:
              enabled: true
              max_items: 30
            reddit:
              enabled: ${{ secrets.REDDIT_CLIENT_ID != '' }}
              subreddits: ["technology", "programming", "MachineLearning"]
              client_id: "${{ secrets.REDDIT_CLIENT_ID }}"
              client_secret: "${{ secrets.REDDIT_CLIENT_SECRET }}"
            v2ex:
              enabled: true
              nodes: ["programmer", "create", "apple"]
            x_com:
              enabled: false
            github:
              enabled: true
            wired:
              enabled: true
            rss:
              enabled: true
              feeds:
                - "https://simonwillison.net/atom/everything/"
                - "https://www.jeffgeerling.com/blog.xml"
                - "https://krebsonsecurity.com/feed/"
                - "https://daringfireball.net/feeds/main"
                - "https://pluralistic.net/feed/"
                - "https://lcamtuf.substack.com/feed"
                - "https://mitchellh.com/feed.xml"
                - "https://dynomight.net/feed.xml"
                - "https://xeiaso.net/blog.rss"
                - "https://devblogs.microsoft.com/oldnewthing/feed"
                - "https://www.righto.com/feeds/posts/default"
                - "https://lucumr.pocoo.org/feed.atom"
                - "https://garymarcus.substack.com/feed"
                - "https://rachelbythebay.com/w/atom.xml"
                - "https://overreacted.io/rss.xml"
                - "https://matklad.github.io/feed.xml"
                - "https://eli.thegreenplace.net/feeds/all.atom.xml"
                - "https://fabiensanglard.net/rss.xml"
                - "https://www.troyhunt.com/rss/"
                - "https://blog.miguelgrinberg.com/feed"
                - "https://computer.rip/rss.xml"
                - "https://www.tedunangst.com/flak/rss"
                - "https://feeds.arstechnica.com/arstechnica/index"
                - "https://www.theverge.com/rss/index.xml"
          llm:
            model: "deepseek/deepseek-chat"
            api_base: "https://api.deepseek.com"
          notifier:
            email:
              enabled: ${{ secrets.EMAIL_USERNAME != '' }}
              smtp_host: "${{ secrets.SMTP_HOST }}"
              smtp_port: 587
              username: "${{ secrets.EMAIL_USERNAME }}"
              password: "${{ secrets.EMAIL_PASSWORD }}"
              to: "${{ secrets.EMAIL_TO }}"
            telegram:
              enabled: ${{ secrets.TELEGRAM_BOT_TOKEN != '' }}
              bot_token: "${{ secrets.TELEGRAM_BOT_TOKEN }}"
              chat_id: "${{ secrets.TELEGRAM_CHAT_ID }}"
          interests:
            - "AI/机器学习"
            - "编程/开发工具"
            - "硬件/数码"
            - "科技新闻"
            - "徒步/户外"
            - "摄影"
            - "国家地理/孤独星球"
          HEREDOC

      - name: Create data directory
        run: mkdir -p data

      - name: Run News Agent
        env:
          DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
        run: uv run news-agent

      - name: Save SQLite database
        if: always()
        uses: actions/cache/save@v4
        with:
          path: data/news.db
          key: news-db-${{ github.run_id }}
```

**Step 2: Create CI test workflow**

`.github/workflows/tests.yml`:
```yaml
name: Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Set up Python
        run: uv python install 3.12

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Run tests
        run: uv run pytest -v --tb=short
```

**Step 3: Commit**

```bash
git add .github/
git commit -m "feat: add GitHub Actions workflows for scheduled runs and CI"
```

---

## Post-Implementation Setup Checklist

After all tasks complete, set up these GitHub repository secrets:

**Required:**
- `DEEPSEEK_API_KEY` - DeepSeek API key

**For Email notifications:**
- `SMTP_HOST` - e.g. `smtp.gmail.com`
- `EMAIL_USERNAME` - sender email
- `EMAIL_PASSWORD` - app password
- `EMAIL_TO` - recipient email

**For Telegram notifications:**
- `TELEGRAM_BOT_TOKEN` - from @BotFather
- `TELEGRAM_CHAT_ID` - your chat ID

**For Reddit (optional):**
- `REDDIT_CLIENT_ID` - Reddit app client ID
- `REDDIT_CLIENT_SECRET` - Reddit app client secret

**Local testing:**
1. `cp config.example.yaml config.yaml` and fill in credentials
2. `uv sync`
3. `uv run pytest -v`
4. `uv run news-agent` to test manually

**Note:** X.com source is disabled in GitHub Actions (requires browser session). For local use, run `uv run news-agent login-x` to set up.
