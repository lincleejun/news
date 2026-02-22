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
