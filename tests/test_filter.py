import json
from unittest.mock import AsyncMock, MagicMock, patch

from news_agent.filter import LLMFilter
from news_agent.models import Article


async def test_filter_scores_articles():
    articles = [
        Article(
            source="hn",
            title="New AI Model Released",
            url="https://ai.com",
            summary="Big breakthrough",
        ),
        Article(
            source="hn",
            title="Cat Video Goes Viral",
            url="https://cats.com",
            summary="Funny cat",
        ),
    ]

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps(
                    [
                        {
                            "index": 0,
                            "score": 9.0,
                            "reason": "Major AI development",
                            "is_hot": True,
                        },
                        {
                            "index": 1,
                            "score": 3.0,
                            "reason": "Not tech related",
                            "is_hot": False,
                        },
                    ]
                )
            )
        )
    ]

    with patch(
        "news_agent.filter.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        f = LLMFilter(
            config={
                "model": "deepseek/deepseek-chat",
                "interests": ["AI/机器学习"],
            }
        )
        result = await f.filter_articles(articles)

    assert result[0].llm_score == 9.0
    assert result[0].is_recommended is True
    assert result[0].is_hot is True
    assert result[1].llm_score == 3.0
    assert result[1].is_recommended is False
