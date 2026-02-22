from unittest.mock import patch, MagicMock, AsyncMock
from news_agent.notifier.email import EmailNotifier
from news_agent.models import Article

async def test_email_send():
    articles = [Article(source="hackernews", title="AI News", url="https://ai.com", summary="Big news", llm_score=9.0, llm_reason="重大AI进展", is_hot=True, is_recommended=True)]
    notifier = EmailNotifier({"smtp_host": "smtp.gmail.com", "smtp_port": 587, "username": "test@gmail.com", "password": "pass", "to": "me@gmail.com"})
    with patch("news_agent.notifier.email.aiosmtplib", new_callable=MagicMock) as mock_smtp:
        mock_smtp.send = AsyncMock()
        await notifier.send(articles)
        mock_smtp.send.assert_called_once()
