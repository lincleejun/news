from unittest.mock import MagicMock, patch

from news_agent.sources.rss import RssSource


async def test_rss_fetch():
    source = RssSource({"feeds": ["https://example.com/feed.xml"]})
    mock_feed = MagicMock()
    mock_feed.entries = [
        MagicMock(
            title="RSS Article",
            link="https://example.com/article1",
            get=lambda k, d="": {"summary": "A summary", "author": "Author"}.get(
                k, d
            ),
            published_parsed=(2026, 2, 21, 8, 0, 0, 0, 0, 0),
        )
    ]
    with patch("news_agent.sources.rss.feedparser.parse", return_value=mock_feed):
        articles = await source.fetch()
    assert len(articles) == 1
    assert articles[0].source == "rss"
    assert articles[0].title == "RSS Article"
