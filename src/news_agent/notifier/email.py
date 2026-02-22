from __future__ import annotations
import logging
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any
import aiosmtplib
from jinja2 import Template
from news_agent.models import Article

logger = logging.getLogger(__name__)
TEMPLATE_PATH = Path(__file__).parent.parent / "templates" / "email.html"

class EmailNotifier:
    def __init__(self, config: dict[str, Any]):
        self.smtp_host = config["smtp_host"]
        self.smtp_port = config.get("smtp_port", 587)
        self.username = config["username"]
        self.password = config["password"]
        self.to_addr = config["to"]

    async def send(self, articles: list[Article]) -> None:
        now = datetime.now()
        period = "早间" if now.hour < 14 else "晚间"
        subject = f"[科技日报] {now.strftime('%Y-%m-%d')} {period}精选"
        template = Template(TEMPLATE_PATH.read_text())
        sorted_articles = sorted(articles, key=lambda a: (a.is_hot, a.llm_score), reverse=True)
        html = template.render(title=subject, date=now.strftime("%Y-%m-%d %H:%M"), articles=sorted_articles)
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.username
        msg["To"] = self.to_addr
        msg.attach(MIMEText(html, "html"))
        await aiosmtplib.send(msg, hostname=self.smtp_host, port=self.smtp_port, username=self.username, password=self.password, start_tls=True)
        logger.info(f"Email sent to {self.to_addr}: {subject}")
