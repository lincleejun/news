from __future__ import annotations

import abc
import asyncio
import logging
from typing import Any

import aiohttp

from news_agent.models import Article

logger = logging.getLogger(__name__)


class BaseSource(abc.ABC):
    name: str = "base"
    timeout: int = 30

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    async def fetch(self) -> list[Article]:
        try:
            async with aiohttp.ClientSession() as session:
                return await asyncio.wait_for(
                    self._fetch(session), timeout=self.timeout
                )
        except asyncio.TimeoutError:
            logger.warning(f"[{self.name}] fetch timed out after {self.timeout}s")
            return []
        except Exception as e:
            logger.error(f"[{self.name}] fetch failed: {e}")
            return []

    @abc.abstractmethod
    async def _fetch(self, session: aiohttp.ClientSession) -> list[Article]:
        ...
