from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Template

from news_agent.models import Article

logger = logging.getLogger(__name__)
TEMPLATE_PATH = Path(__file__).parent.parent / "templates" / "email.html"


class FileNotifier:
    def __init__(self, config: dict[str, Any]):
        self.output_dir = config.get("output_dir", "output")

    async def send(self, articles: list[Article], papers: list[Article] | None = None) -> None:
        now = datetime.now()
        period = "早间" if now.hour < 14 else "晚间"
        subject = f"[科技日报] {now.strftime('%Y-%m-%d')} {period}精选"
        template = Template(TEMPLATE_PATH.read_text())
        sorted_articles = sorted(
            articles, key=lambda a: (a.is_hot, a.llm_score), reverse=True
        )
        html = template.render(
            title=subject,
            date=now.strftime("%Y-%m-%d %H:%M"),
            articles=sorted_articles,
            papers=papers or [],
        )
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f"news-{now.strftime('%Y%m%d-%H%M%S')}.html"
        filepath = output_path / filename
        filepath.write_text(html, encoding="utf-8")
        logger.info(f"News saved to {filepath}")
