from __future__ import annotations

import json
import logging
from typing import Any

import litellm

from news_agent.models import Article

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个科技资讯筛选助手。你的任务是对一批文章进行评分和筛选。

用户关注的领域：
{interests}

评分标准 (0-10):
- 9-10: 重大突破、行业变革、必读内容
- 7-8: 有价值的信息、值得了解
- 5-6: 一般资讯、可看可不看
- 3-4: 相关性较低
- 0-2: 无关或低质量内容

请对每篇文章输出 JSON 数组，每个元素包含:
- index: 文章序号 (从0开始)
- score: 评分 (0-10, 浮点数)
- reason: 推荐理由 (一句话, 中文)
- is_hot: 是否高热度/高潜力 (布尔值)

只输出 JSON 数组，不要其他内容。"""


class LLMFilter:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.model = self.config.get("model", "deepseek/deepseek-chat")
        self.interests = self.config.get("interests", [])
        self.recommend_threshold = self.config.get("recommend_threshold", 7.0)
        self.api_base = self.config.get("api_base", None)

    async def filter_articles(
        self, articles: list[Article], batch_size: int = 25
    ) -> list[Article]:
        all_results = []
        for i in range(0, len(articles), batch_size):
            batch = articles[i : i + batch_size]
            scored = await self._score_batch(batch)
            all_results.extend(scored)
        return all_results

    async def _score_batch(self, articles: list[Article]) -> list[Article]:
        articles_text = "\n\n".join(
            f"[{i}] 来源: {a.source} | 标题: {a.title} | "
            f"摘要: {a.summary} | 热度: {a.score} | 评论数: {a.comments_count}"
            for i, a in enumerate(articles)
        )

        system = SYSTEM_PROMPT.format(
            interests="\n".join(f"- {i}" for i in self.interests)
        )

        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": 4096,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": articles_text},
                ],
            }
            if self.api_base:
                kwargs["api_base"] = self.api_base

            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                content = content.rsplit("```", 1)[0]
            scores = json.loads(content)

            for item in scores:
                idx = item["index"]
                if 0 <= idx < len(articles):
                    articles[idx].llm_score = item["score"]
                    articles[idx].llm_reason = item["reason"]
                    articles[idx].is_hot = item.get("is_hot", False)
                    articles[idx].is_recommended = (
                        item["score"] >= self.recommend_threshold
                    )
        except Exception as e:
            logger.error(f"LLM filtering failed: {e}")

        return articles
