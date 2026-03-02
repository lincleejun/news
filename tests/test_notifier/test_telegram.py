from unittest.mock import AsyncMock, patch

from news_agent.models import Article
from news_agent.notifier.telegram import TelegramNotifier


async def test_telegram_send():
    articles = [
        Article(
            source="hackernews", title="AI News", url="https://ai.com",
            llm_score=9.0, llm_reason="重大AI进展", is_hot=True, is_recommended=True,
        )
    ]
    notifier = TelegramNotifier({"bot_token": "123:ABC", "chat_id": "456"})
    with patch("news_agent.notifier.telegram.Bot") as MockBot:
        mock_bot = MockBot.return_value
        mock_bot.send_message = AsyncMock()
        await notifier.send(articles)
        mock_bot.send_message.assert_called_once()


async def test_telegram_send_with_papers():
    articles = [
        Article(
            source="hackernews", title="AI News", url="https://ai.com",
            llm_score=9.0, llm_reason="重大AI进展", is_hot=True, is_recommended=True,
        )
    ]
    papers = [
        Article(
            source="arxiv_papers", title="Attention Is All You Need v2",
            url="https://arxiv.org/abs/2503.00001", summary="A new transformer architecture.",
            llm_score=8.5, llm_reason="热门AI论文", is_recommended=True, score=120,
        )
    ]
    notifier = TelegramNotifier({"bot_token": "123:ABC", "chat_id": "456"})
    with patch("news_agent.notifier.telegram.Bot") as MockBot:
        mock_bot = MockBot.return_value
        mock_bot.send_message = AsyncMock()
        await notifier.send(articles, papers)
        call_args = mock_bot.send_message.call_args
        text = call_args.kwargs.get("text", "")
        assert "热门论文" in text
        assert "Attention Is All You Need v2" in text
