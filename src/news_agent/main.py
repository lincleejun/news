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
        "hackernews": HackerNewsSource, "reddit": RedditSource, "v2ex": V2exSource,
        "github": GitHubTrendingSource, "rss": RssSource, "wired": WiredSource, "x_com": XSource,
    }
    sources = []
    source_configs = config.get("sources", {})
    for name, cls in source_map.items():
        src_cfg = source_configs.get(name, {})
        if src_cfg.get("enabled", True):
            sources.append(cls(src_cfg))
    return sources

async def run_agent(config_path: str = "config.yaml") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
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
        llm_filter = LLMFilter(config={**config.get("llm", {}), "interests": config.get("interests", [])})
        scored_articles = await llm_filter.filter_articles(unique_articles)
        await storage.save_articles(scored_articles)
        recommended = [a for a in scored_articles if a.is_recommended]
        recommended.sort(key=lambda a: (a.is_hot, a.llm_score), reverse=True)
        recommended = recommended[:10]
        logger.info(f"Recommended: {len(recommended)} articles")
        if not recommended:
            logger.info("No articles passed the quality threshold.")
            return
        notifier_config = config.get("notifier", {})
        push_tasks = []
        email_cfg = notifier_config.get("email", {})
        if email_cfg.get("enabled"):
            push_tasks.append(EmailNotifier(email_cfg).send(recommended))
        tg_cfg = notifier_config.get("telegram", {})
        if tg_cfg.get("enabled"):
            push_tasks.append(TelegramNotifier(tg_cfg).send(recommended))
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
