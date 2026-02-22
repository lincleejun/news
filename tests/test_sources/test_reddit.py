from aioresponses import aioresponses

from news_agent.sources.reddit import RedditSource


async def test_reddit_fetch():
    source = RedditSource(
        {
            "subreddits": ["technology"],
            "client_id": "test_id",
            "client_secret": "test_secret",
        }
    )
    with aioresponses() as mocked:
        mocked.post(
            "https://www.reddit.com/api/v1/access_token",
            payload={
                "access_token": "test_token",
                "token_type": "bearer",
            },
        )
        mocked.get(
            "https://oauth.reddit.com/r/technology/hot?limit=25",
            payload={
                "data": {
                    "children": [
                        {
                            "data": {
                                "id": "abc123",
                                "title": "New AI Breakthrough",
                                "url": "https://ai-news.com/article",
                                "author": "techuser",
                                "score": 500,
                                "num_comments": 200,
                                "created_utc": 1708500000,
                                "selftext": "This is a summary",
                                "subreddit": "technology",
                            }
                        }
                    ]
                }
            },
        )
        articles = await source.fetch()
    assert len(articles) == 1
    assert articles[0].source == "reddit"
    assert articles[0].title == "New AI Breakthrough"
