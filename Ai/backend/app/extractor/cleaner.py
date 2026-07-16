from bs4 import BeautifulSoup

REMOVE_TAGS = ["nav", "footer", "header", "script", "style", "noscript", "iframe", "form", "aside"]
REMOVE_SELECTOR_KEYWORDS = [
    "cookie", "consent", "banner", "advert", "ads-", "ad-slot", "social", "share-buttons",
    "sidebar", "popup", "modal", "newsletter", "breadcrumb",
]


def _matches_noise_keyword(tag) -> bool:
    haystack = " ".join([
        " ".join(tag.get("class", [])) if tag.get("class") else "",
        tag.get("id", "") or "",
    ]).lower()
    return any(keyword in haystack for keyword in REMOVE_SELECTOR_KEYWORDS)


def clean_html(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "html.parser")

    for tag_name in REMOVE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # decompose() on a parent also tears down its (already-collected) descendant Tag
    # objects, so skip anything that was decomposed as a side effect of an earlier
    # iteration in this same loop.
    for tag in soup.find_all(True):
        if tag.attrs is None:
            continue
        if _matches_noise_keyword(tag):
            tag.decompose()

    return soup


def extract_with_bs4(html: str) -> str:
    soup = clean_html(html)
    body = soup.find("body") or soup
    text = body.get_text("\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)
