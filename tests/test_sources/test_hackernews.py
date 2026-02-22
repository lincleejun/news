from aioresponses import aioresponses

from news_agent.sources.hackernews import HackerNewsSource


async def test_hackernews_fetch():
    source = HackerNewsSource({"max_items": 2})
    with aioresponses() as mocked:
        mocked.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json",
            payload=[101, 102, 103],
        )
        mocked.get(
            "https://hacker-news.firebaseio.com/v0/item/101.json",
            payload={
                "id": 101,
                "title": "Show HN: Cool Project",
                "url": "https://cool.com",
                "by": "user1",
                "score": 150,
                "descendants": 42,
                "time": 1708500000,
                "type": "story",
            },
        )
        mocked.get(
            "https://hacker-news.firebaseio.com/v0/item/102.json",
            payload={
                "id": 102,
                "title": "Ask HN: Best editor?",
                "url": "",
                "by": "user2",
                "score": 80,
                "descendants": 100,
                "time": 1708500100,
                "type": "story",
            },
        )
        articles = await source.fetch()
    assert len(articles) == 2
    assert articles[0].source == "hackernews"
    assert articles[0].title == "Show HN: Cool Project"
    assert articles[0].score == 150
