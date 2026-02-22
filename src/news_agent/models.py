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
