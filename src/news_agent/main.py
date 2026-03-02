from __future__ import annotations
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any
import yaml
from news_agent.filter import LLMFilter
from news_agent.models import Article
from news_agent.notifier.email import EmailNotifier
from news_agent.notifier.file import FileNotifier
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
    from news_agent.sources.ai_blogs import AiBlogsSource
    from news_agent.sources.arxiv_papers import ArxivPapersSource
    source_map = {
        "hackernews": HackerNewsSource, "reddit": RedditSource, "v2ex": V2exSource,
        "github": GitHubTrendingSource, "rss": RssSource, "wired": WiredSource, "x_com": XSource,
        "ai_blogs": AiBlogsSource, "arxiv_papers": ArxivPapersSource,
    }
    sources = []
    source_configs = config.get("sources", {})
    for name, cls in source_map.items():
        src_cfg = source_configs.get(name, {})
        if src_cfg.get("enabled", True):
            sources.append(cls(src_cfg))
    return sources

def _set_api_keys(config: dict[str, Any]) -> None:
    """Set LLM API keys from config if not already in environment."""
    key_map = {"gemini": "GEMINI_API_KEY", "deepseek": "DEEPSEEK_API_KEY", "github_models": "OPENAI_API_KEY"}
    for section, env_var in key_map.items():
        key = config.get(section, {}).get("key")
        if key and not os.environ.get(env_var):
            os.environ[env_var] = key

async def run_agent(config_path: str = "config.yaml") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    config = load_config(config_path)
    _set_api_keys(config)
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
        # Separate special sources from regular articles
        blog_articles = [a for a in unique_articles if a.source == "ai_blogs"]
        paper_articles = [a for a in unique_articles if a.source == "arxiv_papers"]
        other_articles = [a for a in unique_articles if a.source not in ("ai_blogs", "arxiv_papers")]

        # Auto-recommend AI blogs
        for a in blog_articles:
            a.is_recommended = True
            a.llm_score = 8.5
            a.llm_reason = "AI公司官方博客"
        logger.info(f"AI blogs auto-recommended: {len(blog_articles)} articles")

        # Auto-recommend papers
        for a in paper_articles:
            a.is_recommended = True
            a.llm_score = 8.5
            a.llm_reason = "热门AI论文"
        logger.info(f"Arxiv papers auto-recommended: {len(paper_articles)} articles")

        # LLM filter other articles
        scored_articles: list[Article] = list(blog_articles) + list(paper_articles)
        if other_articles:
            llm_filter = LLMFilter(config={**config.get("llm", {}), "interests": config.get("interests", [])})
            scored_articles.extend(await llm_filter.filter_articles(other_articles))

        await storage.save_articles(scored_articles)

        # Split recommended news and papers for notification
        recommended_news = [a for a in scored_articles if a.is_recommended and a.source != "arxiv_papers"]
        recommended_news.sort(key=lambda a: (a.is_hot, a.llm_score), reverse=True)
        recommended_news = recommended_news[:10]

        recommended_papers = [a for a in scored_articles if a.is_recommended and a.source == "arxiv_papers"]
        recommended_papers.sort(key=lambda a: a.score, reverse=True)
        recommended_papers = recommended_papers[:10]

        logger.info(f"Recommended: {len(recommended_news)} articles, {len(recommended_papers)} papers")

        if not recommended_news and not recommended_papers:
            logger.info("No articles passed the quality threshold.")
            return

        notifier_config = config.get("notifier", {})
        push_tasks = []
        email_cfg = notifier_config.get("email", {})
        if email_cfg.get("enabled"):
            push_tasks.append(EmailNotifier(email_cfg).send(recommended_news, recommended_papers))
        tg_cfg = notifier_config.get("telegram", {})
        if tg_cfg.get("enabled"):
            push_tasks.append(TelegramNotifier(tg_cfg).send(recommended_news, recommended_papers))
        file_cfg = notifier_config.get("file", {})
        if file_cfg.get("enabled"):
            push_tasks.append(FileNotifier(file_cfg).send(recommended_news, recommended_papers))
        if push_tasks:
            results = await asyncio.gather(*push_tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.error(f"Notification failed: {r}")
            all_sent = [a.id for a in recommended_news] + [a.id for a in recommended_papers]
            await storage.mark_sent(all_sent)
            logger.info("Notifications sent (check logs for errors).")
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
