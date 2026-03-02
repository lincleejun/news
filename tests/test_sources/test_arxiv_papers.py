import re

from aioresponses import aioresponses

from news_agent.sources.arxiv_papers import ArxivPapersSource

HF_PATTERN = re.compile(r"^https://huggingface\.co/api/daily_papers")
PWC_PATTERN = re.compile(r"^https://paperswithcode\.com/api/v1/papers/")


def _hf_paper(arxiv_id: str, title: str, upvotes: int = 0, num_comments: int = 0):
    return {
        "id": arxiv_id,
        "title": title,
        "summary": f"Summary of {title}",
        "upvotes": upvotes,
        "numComments": num_comments,
        "publishedAt": "2025-01-15T00:00:00Z",
    }


def _pwc_paper(arxiv_id: str, title: str):
    return {
        "id": 999,
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": f"Abstract of {title}",
        "url_abs": f"https://arxiv.org/abs/{arxiv_id}",
        "published": "2025-01-15",
    }


async def test_merge_and_dedup():
    """Papers from both sources are merged; HF takes priority on duplicate arxiv IDs."""
    source = ArxivPapersSource({"max_papers": 10})
    with aioresponses() as mocked:
        mocked.get(
            HF_PATTERN,
            payload=[
                _hf_paper("2401.00001", "HF Paper A", upvotes=10),
                _hf_paper("2401.00002", "HF Paper B (dup)", upvotes=5),
            ],
        )
        mocked.get(
            PWC_PATTERN,
            payload={
                "count": 2,
                "results": [
                    _pwc_paper("2401.00002", "PWC Paper B (dup)"),
                    _pwc_paper("2401.00003", "PWC Paper C"),
                ],
            },
        )
        articles = await source.fetch()

    assert len(articles) == 3
    titles = {a.title for a in articles}
    # HF version wins for the duplicate 2401.00002
    assert "HF Paper B (dup)" in titles
    assert "PWC Paper B (dup)" not in titles
    # Unique papers from each source are present
    assert "HF Paper A" in titles
    assert "PWC Paper C" in titles


async def test_hf_failure_returns_pwc_only():
    """When HuggingFace API fails, papers from PapersWithCode are still returned."""
    source = ArxivPapersSource({"max_papers": 10})
    with aioresponses() as mocked:
        mocked.get(HF_PATTERN, status=500)
        mocked.get(
            PWC_PATTERN,
            payload={
                "count": 1,
                "results": [_pwc_paper("2401.00010", "PWC Only")],
            },
        )
        articles = await source.fetch()

    assert len(articles) == 1
    assert articles[0].title == "PWC Only"


async def test_pwc_failure_returns_hf_only():
    """When PapersWithCode API fails, papers from HuggingFace are still returned."""
    source = ArxivPapersSource({"max_papers": 10})
    with aioresponses() as mocked:
        mocked.get(
            HF_PATTERN,
            payload=[_hf_paper("2401.00020", "HF Only", upvotes=7)],
        )
        mocked.get(PWC_PATTERN, status=500)
        articles = await source.fetch()

    assert len(articles) == 1
    assert articles[0].title == "HF Only"


async def test_max_papers_limit():
    """Results are limited to max_papers."""
    source = ArxivPapersSource({"max_papers": 2})
    with aioresponses() as mocked:
        mocked.get(
            HF_PATTERN,
            payload=[
                _hf_paper("2401.00001", "Paper 1", upvotes=30),
                _hf_paper("2401.00002", "Paper 2", upvotes=20),
                _hf_paper("2401.00003", "Paper 3", upvotes=10),
            ],
        )
        mocked.get(
            PWC_PATTERN,
            payload={"count": 0, "results": []},
        )
        articles = await source.fetch()

    assert len(articles) == 2
    # Sorted by score descending
    assert articles[0].score >= articles[1].score


async def test_sorted_by_score():
    """Articles are sorted by score in descending order."""
    source = ArxivPapersSource({"max_papers": 10})
    with aioresponses() as mocked:
        mocked.get(
            HF_PATTERN,
            payload=[
                _hf_paper("2401.00001", "Low", upvotes=1),
                _hf_paper("2401.00002", "High", upvotes=100),
                _hf_paper("2401.00003", "Mid", upvotes=50),
            ],
        )
        mocked.get(PWC_PATTERN, payload={"count": 0, "results": []})
        articles = await source.fetch()

    assert articles[0].title == "High"
    assert articles[1].title == "Mid"
    assert articles[2].title == "Low"


async def test_source_metadata():
    """Articles have correct source name and tags."""
    source = ArxivPapersSource()
    with aioresponses() as mocked:
        mocked.get(
            HF_PATTERN,
            payload=[_hf_paper("2401.00001", "Test", upvotes=5)],
        )
        mocked.get(PWC_PATTERN, payload={"count": 0, "results": []})
        articles = await source.fetch()

    assert len(articles) == 1
    assert articles[0].source == "arxiv_papers"
    assert "arxiv" in articles[0].tags
    assert articles[0].url == "https://arxiv.org/abs/2401.00001"
