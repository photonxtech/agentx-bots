from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException

import httpx

from app.utils.logging import get_logger
from app.utils.retry import with_retry
from app.utils.url_utils import is_ignorable_url, is_same_domain, normalize_url

logger = get_logger(__name__)
_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
# NOTE: sitemap XML comes from arbitrary admin-submitted third-party sites, so it is parsed
# with defusedxml (not the stdlib xml.etree) to prevent XXE / billion-laughs attacks.


def is_sitemap_index(xml_text: str) -> bool:
    try:
        root = ElementTree.fromstring(xml_text)
    except (ElementTree.ParseError, DefusedXmlException):
        return False
    tag = root.tag.rsplit("}", 1)[-1]
    return tag == "sitemapindex"


def parse_sitemap_xml(xml_text: str, base_domain: str) -> list[str]:
    try:
        root = ElementTree.fromstring(xml_text)
    except (ElementTree.ParseError, DefusedXmlException):
        logger.warning("Failed to parse sitemap XML")
        return []

    locs = [el.text.strip() for el in root.findall(".//sm:url/sm:loc", _NS) if el.text]
    urls = []
    for loc in locs:
        if is_ignorable_url(loc):
            continue
        if not is_same_domain(loc, base_domain):
            continue
        urls.append(normalize_url(loc))
    return list(dict.fromkeys(urls))


def parse_sitemap_index_locs(xml_text: str) -> list[str]:
    try:
        root = ElementTree.fromstring(xml_text)
    except (ElementTree.ParseError, DefusedXmlException):
        return []
    return [el.text.strip() for el in root.findall(".//sm:sitemap/sm:loc", _NS) if el.text]


@with_retry(max_attempts=3)
async def _fetch_text(client: httpx.AsyncClient, url: str) -> str | None:
    resp = await client.get(url, timeout=15.0, follow_redirects=True)
    if resp.status_code != 200:
        return None
    return resp.text


async def fetch_sitemap_urls(client: httpx.AsyncClient, base_url: str) -> list[str] | None:
    from app.utils.url_utils import get_domain

    sitemap_url = base_url.rstrip("/") + "/sitemap.xml"
    text = await _fetch_text(client, sitemap_url)
    if text is None:
        return None

    base_domain = get_domain(base_url)
    if is_sitemap_index(text):
        all_urls: list[str] = []
        for sub_sitemap in parse_sitemap_index_locs(text):
            sub_text = await _fetch_text(client, sub_sitemap)
            if sub_text:
                all_urls.extend(parse_sitemap_xml(sub_text, base_domain))
        return list(dict.fromkeys(all_urls)) or None

    urls = parse_sitemap_xml(text, base_domain)
    return urls or None
