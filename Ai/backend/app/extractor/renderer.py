from playwright.async_api import async_playwright

from app.utils.logging import get_logger
from app.utils.retry import with_retry

logger = get_logger(__name__)


@with_retry(max_attempts=2)
async def render_page(url: str, timeout_ms: int = 20000) -> str | None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            html = await page.content()
            return html
        except Exception:
            logger.exception("Failed to render %s", url)
            return None
        finally:
            await browser.close()


async def extract_page_content(url: str, min_length: int = 200) -> tuple[str, str] | None:
    from app.extractor.content_extractor import extract_content_from_html

    html = await render_page(url)
    if html is None:
        return None
    return extract_content_from_html(html, url, min_length=min_length)
