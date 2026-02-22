from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from telegram import Bot

from news_agent.models import Article

logger = logging.getLogger(__name__)


def _escape_md(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    return re.sub(r'([_*\[\]()~`>#+\-=|{}.!\\])', r'\\\1', text)


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
        date_str = _escape_md(now.strftime("%Y-%m-%d"))
        lines = [f"📰 *科技日报 {date_str} {period}精选*\n"]
        for i, a in enumerate(sorted_articles, 1):
            hot = "🔥 " if a.is_hot else ""
            source = f"`{_escape_md(a.source)}`"
            title = _escape_md(a.title)
            reason = _escape_md(a.llm_reason)
            lines.append(
                f"{i}\\. {hot}{source} [{title}]({a.url})\n   _{reason}_"
            )
        count = _escape_md(str(len(articles)))
        lines.append(f"\n_共 {count} 条精选_")
        text = "\n\n".join(lines)
        await bot.send_message(
            chat_id=self.chat_id,
            text=text,
            parse_mode="MarkdownV2",
            disable_web_page_preview=True,
        )
        logger.info(f"Telegram message sent to chat {self.chat_id}")
