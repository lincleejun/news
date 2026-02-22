# News Agent Design

## Overview

A local news aggregation agent that runs on a cron schedule, fetching content from multiple tech and lifestyle sources, using LLM-based intelligent filtering, and pushing curated recommendations via Email and Telegram.

## Architecture: Async Concurrent Fetching (Plan B)

```
cron → main.py → [asyncio concurrent fetch all sources] → [dedup] → [LLM filter] → [push notifications]
```

- Python 3.11+, managed with uv
- Local cron job (twice daily: 08:00, 20:00)
- SQLite for persistence, dedup, and history tracking
- Claude Haiku for intelligent content scoring and filtering

## Project Structure

```
News/
├── pyproject.toml
├── config.yaml
├── src/
│   └── news_agent/
│       ├── __init__.py
│       ├── main.py
│       ├── sources/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── hackernews.py
│       │   ├── reddit.py
│       │   ├── v2ex.py
│       │   ├── wired.py
│       │   ├── github_trending.py
│       │   ├── rss.py
│       │   └── twitter.py       # X.com via Playwright
│       ├── filter.py
│       ├── storage.py
│       ├── notifier/
│       │   ├── __init__.py
│       │   ├── email.py
│       │   └── telegram.py
│       └── models.py
└── data/
    └── news.db
```

## Data Sources

| Source | Method | Notes |
|--------|--------|-------|
| Hacker News | Official API | Free, no key needed |
| Reddit | OAuth API | Needs client_id/secret |
| V2EX | Public API + HTML | Partial API coverage |
| X.com | Playwright browser | Login session, Timeline + Trending |
| GitHub | Trending page scraping | Parse trending page |
| Wired | RSS feed | Standard RSS |
| RSS (generic) | feedparser | Configurable feed URLs |

## Data Model

```python
@dataclass
class Article:
    id: str
    source: str
    title: str
    url: str
    summary: str
    author: str
    published_at: datetime
    fetched_at: datetime
    score: int
    comments_count: int
    tags: list[str]
    llm_score: float
    llm_reason: str
    is_recommended: bool
    is_hot: bool
    sent: bool
```

## Core Flow

1. **Fetch** (asyncio concurrent): Each source fetches independently with 30s timeout
2. **Dedup**: URL-based dedup + SQLite history check; exception for articles with 3x score growth
3. **LLM Filter**: Batch 20-30 articles per call to Claude Haiku, score 0-10, threshold >= 7 for recommend, >= 9 for hot
4. **Push**: Top 10 articles formatted as HTML email + Telegram Markdown, sent concurrently

## LLM Filtering

- Provider: Anthropic Claude (claude-haiku-4-5)
- Batch processing to minimize API calls
- User interest profile embedded in system prompt
- Structured JSON output with score, reason, and hot flag

## User Interests

- AI/Machine Learning
- Programming/Dev Tools
- Hardware/Digital Products
- General Tech News
- Hiking/Outdoors
- Photography
- National Geographic / Lonely Planet

## Push Notifications

- **Email**: HTML formatted, subject `[Tech Daily] YYYY-MM-DD Morning/Evening Selection`
- **Telegram**: Markdown messages with source tags, titles, and links
- Hot items displayed at top with special markers

## Dependencies

- aiohttp, beautifulsoup4, feedparser, playwright
- anthropic, aiosqlite, pyyaml, jinja2, python-telegram-bot
