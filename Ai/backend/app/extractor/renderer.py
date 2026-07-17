from contextlib import asynccontextmanager

from playwright.async_api import Browser, async_playwright

from app.utils.logging import get_logger
from app.utils.retry import with_retry

logger = get_logger(__name__)


@asynccontextmanager
async def launch_browser():
    # One Chromium process for the whole crawl, not one per page — launching a fresh
    # browser per page (as this used to do) multiplies memory/CPU overhead by the page
    # count and was tipping memory-constrained deployments over their limit. A page/tab
    # within an already-running browser is far cheaper than a whole new browser process.
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            await browser.close()


@with_retry(max_attempts=2)
async def render_page(browser: Browser, url: str, timeout_ms: int = 20000) -> str | None:
    page = await browser.new_page()
    try:
        await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
        return await page.content()
    except Exception:
        logger.exception("Failed to render %s", url)
        return None
    finally:
        await page.close()


async def extract_page_content(browser: Browser, url: str, min_length: int = 200) -> tuple[str, str] | None:
    from app.extractor.content_extractor import extract_content_from_html

    html = await render_page(browser, url)
    if html is None:
        return None
    return extract_content_from_html(html, url, min_length=min_length)
