import asyncio
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from app.utils.logging import get_logger
from app.utils.retry import with_retry
from app.utils.url_utils import is_ignorable_url, is_same_domain, normalize_url

logger = get_logger(__name__)


def extract_links(html: str, base_url: str, base_domain: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    found = []
    for tag in soup.find_all("a", href=True):
        absolute = urljoin(base_url, tag["href"])
        if is_ignorable_url(absolute):
            continue
        if not is_same_domain(absolute, base_domain):
            continue
        found.append(normalize_url(absolute))
    return list(dict.fromkeys(found))


@with_retry(max_attempts=3)
async def _fetch_html(client: httpx.AsyncClient, url: str) -> str | None:
    resp = await client.get(url, timeout=15.0, follow_redirects=True)
    if resp.status_code != 200 or "text/html" not in resp.headers.get("content-type", ""):
        return None
    return resp.text


async def recursive_crawl(
    client: httpx.AsyncClient,
    base_url: str,
    max_pages: int,
    max_depth: int,
    delay_seconds: float,
) -> list[str]:
    from app.utils.url_utils import get_domain

    base_domain = get_domain(base_url)
    start = normalize_url(base_url)
    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(start, 0)]
    discovered: list[str] = []

    while queue and len(discovered) < max_pages:
        url, depth = queue.pop(0)
        if url in visited or depth > max_depth:
            continue
        visited.add(url)

        html = await _fetch_html(client, url)
        await asyncio.sleep(delay_seconds)
        if html is None:
            continue

        discovered.append(url)
        if depth < max_depth:
            for link in extract_links(html, url, base_domain):
                if link not in visited:
                    queue.append((link, depth + 1))

    return discovered[:max_pages]
