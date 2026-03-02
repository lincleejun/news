from unittest.mock import AsyncMock, patch

from news_agent.main import run_agent
from news_agent.models import Article


async def test_run_agent_orchestrates_pipeline():
    mock_article = Article(
        source="hn", title="Test", url="https://test.com",
        llm_score=8.0, is_recommended=True,
    )
    mock_source = AsyncMock()
    mock_source.fetch = AsyncMock(return_value=[mock_article])
    mock_source.name = "test"

    mock_storage = AsyncMock()
    mock_storage.article_exists = AsyncMock(return_value=False)
    mock_storage.get_previous_score = AsyncMock(return_value=None)

    mock_filter = AsyncMock()
    mock_filter.filter_articles = AsyncMock(return_value=[mock_article])

    mock_email = AsyncMock()
    mock_telegram = AsyncMock()

    with patch("news_agent.main.load_config") as mock_cfg, \
         patch("news_agent.main.create_sources", return_value=[mock_source]), \
         patch("news_agent.main.Storage", return_value=mock_storage), \
         patch("news_agent.main.LLMFilter", return_value=mock_filter), \
         patch("news_agent.main.EmailNotifier", return_value=mock_email), \
         patch("news_agent.main.TelegramNotifier", return_value=mock_telegram):
        mock_cfg.return_value = {
            "sources": {},
            "llm": {"model": "deepseek/deepseek-chat"},
            "notifier": {
                "email": {"enabled": True, "smtp_host": "h", "smtp_port": 587, "username": "u", "password": "p", "to": "t"},
                "telegram": {"enabled": True, "bot_token": "t", "chat_id": "c"},
            },
            "interests": ["AI"],
        }
        await run_agent()

    mock_filter.filter_articles.assert_called_once()
    mock_email.send.assert_called_once()
    mock_telegram.send.assert_called_once()
    # Verify send was called with two arguments (articles, papers)
    assert len(mock_email.send.call_args[0]) == 2
    mock_storage.mark_sent.assert_called_once()
