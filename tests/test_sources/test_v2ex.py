from aioresponses import aioresponses

from news_agent.sources.v2ex import V2exSource


async def test_v2ex_fetch():
    source = V2exSource({"nodes": ["programmer"]})
    with aioresponses() as mocked:
        mocked.get(
            "https://www.v2ex.com/api/v2/nodes/programmer/topics?p=1",
            payload=[
                {
                    "id": 12345,
                    "title": "Python最佳实践",
                    "url": "https://www.v2ex.com/t/12345",
                    "content": "分享一些经验",
                    "member": {"username": "dev123"},
                    "created": 1708500000,
                    "replies": 30,
                }
            ],
        )
        articles = await source.fetch()
    assert len(articles) == 1
    assert articles[0].source == "v2ex"
    assert articles[0].title == "Python最佳实践"
