import httpx

from app.crawler.recursive import recursive_crawl
from app.crawler.sitemap import fetch_sitemap_urls
from app.utils.logging import get_logger

logger = get_logger(__name__)


async def discover_urls(
    base_url: str,
    max_pages: int,
    max_depth: int,
    delay_seconds: float,
) -> tuple[list[str], str | None]:
    async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0 (compatible; ChatbotCrawler/1.0)"}) as client:
        sitemap_urls = await fetch_sitemap_urls(client, base_url)
        # A sitemap alone is often incomplete (pages added since it was last generated, or
        # intentionally unlisted but still linked), and a link-following crawl alone misses
        # pages that exist only in the sitemap with no inbound internal link. Always run
        # both and merge so neither gap causes a page to be silently skipped.
        recursive_urls = await recursive_crawl(client, base_url, max_pages, max_depth, delay_seconds)

        merged = list(dict.fromkeys([*sitemap_urls, *recursive_urls]))[:max_pages]
        sitemap_used = base_url.rstrip("/") + "/sitemap.xml" if sitemap_urls else None
        logger.info(
            "Discovered %d URLs for %s (%d from sitemap, %d from recursive crawl)",
            len(merged), base_url, len(sitemap_urls), len(recursive_urls),
        )
        return merged, sitemap_used
