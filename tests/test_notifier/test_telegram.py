from unittest.mock import AsyncMock, patch, MagicMock
from news_agent.notifier.telegram import TelegramNotifier
from news_agent.models import Article

async def test_telegram_send():
    articles = [Article(source="hackernews", title="AI News", url="https://ai.com", llm_score=9.0, llm_reason="重大AI进展", is_hot=True, is_recommended=True)]
    notifier = TelegramNotifier({"bot_token": "123:ABC", "chat_id": "456"})
    with patch("news_agent.notifier.telegram.Bot") as MockBot:
        mock_bot = MockBot.return_value
        mock_bot.send_message = AsyncMock()
        await notifier.send(articles)
        mock_bot.send_message.assert_called_once()
