from aioresponses import aioresponses

from news_agent.sources.github_trending import GitHubTrendingSource


async def test_github_trending_fetch():
    source = GitHubTrendingSource()
    html = (
        '<article class="Box-row">'
        '<h2 class="h3 lh-condensed">'
        '<a href="/user/repo">user / repo</a>'
        "</h2>"
        '<p class="col-9 color-fg-muted my-1 pr-4">A cool project</p>'
        '<div class="f6 color-fg-muted mt-2">'
        '<a class="Link d-inline-block mr-3" href="/user/repo/stargazers">'
        "1,234</a>"
        "</div>"
        "</article>"
    )
    with aioresponses() as mocked:
        mocked.get("https://github.com/trending?since=daily", body=html)
        articles = await source.fetch()
    assert len(articles) >= 1
    assert articles[0].source == "github"
    assert "repo" in articles[0].title
