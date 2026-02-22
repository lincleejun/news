from news_agent.models import Article


def test_article_id_deterministic():
    a = Article(source="hackernews", title="Test", url="https://example.com")
    b = Article(source="hackernews", title="Test", url="https://example.com")
    assert a.id == b.id


def test_article_id_unique_across_sources():
    a = Article(source="hackernews", title="Test", url="https://example.com")
    b = Article(source="reddit", title="Test", url="https://example.com")
    assert a.id != b.id
