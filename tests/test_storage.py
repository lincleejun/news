import pytest
from news_agent.storage import Storage
from news_agent.models import Article


@pytest.fixture
async def storage(tmp_path):
    db_path = tmp_path / "test.db"
    s = Storage(str(db_path))
    await s.initialize()
    yield s
    await s.close()


async def test_save_and_exists(storage):
    article = Article(source="hackernews", title="Test", url="https://example.com")
    await storage.save_article(article)
    assert await storage.article_exists(article.id)


async def test_not_exists(storage):
    assert not await storage.article_exists("nonexistent")


async def test_get_unsent_articles(storage):
    a1 = Article(source="hn", title="A1", url="https://a1.com", llm_score=8.0, is_recommended=True)
    a2 = Article(source="hn", title="A2", url="https://a2.com", llm_score=5.0, is_recommended=False)
    await storage.save_article(a1)
    await storage.save_article(a2)
    unsent = await storage.get_unsent_recommended()
    assert len(unsent) == 1
    assert unsent[0].title == "A1"


async def test_mark_as_sent(storage):
    a = Article(source="hn", title="A1", url="https://a1.com", is_recommended=True)
    await storage.save_article(a)
    await storage.mark_sent([a.id])
    unsent = await storage.get_unsent_recommended()
    assert len(unsent) == 0


async def test_get_previous_score(storage):
    a = Article(source="hn", title="A1", url="https://a1.com", score=10)
    await storage.save_article(a)
    prev = await storage.get_previous_score(a.url)
    assert prev == 10
