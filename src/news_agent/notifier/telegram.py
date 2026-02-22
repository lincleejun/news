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
        sorted_articles = sorted(articles, key=lambda a: (a.is_hot, a.llm_score), reverse=True)
        lines = [f"📰 *科技日报 {now.strftime('%Y-%m-%d')} {period}精选*\n"]
        for i, a in enumerate(sorted_articles, 1):
            hot = "🔥 " if a.is_hot else ""
            source = f"`{a.source}`"
            title = a.title.replace("[", "\\[").replace("]", "\\]")
            lines.append(f"{i}\\. {hot}{source} [{title}]({a.url})\n   _{a.llm_reason}_")
        lines.append(f"\n_共 {len(articles)} 条精选_")
        text = "\n\n".join(lines)
        await bot.send_message(chat_id=self.chat_id, text=text, parse_mode="MarkdownV2", disable_web_page_preview=True)
        logger.info(f"Telegram message sent to chat {self.chat_id}")
