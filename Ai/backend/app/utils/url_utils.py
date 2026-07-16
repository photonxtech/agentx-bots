from urllib.parse import urldefrag, urlparse

IGNORED_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".bmp",
    ".mp4", ".webm", ".mov", ".avi",
    ".pdf", ".zip", ".rar", ".tar", ".gz", ".7z",
    ".css", ".js", ".mjs",
    ".woff", ".woff2", ".ttf", ".eot",
)
IGNORED_SCHEMES = ("mailto", "tel", "javascript")


def is_ignorable_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme in IGNORED_SCHEMES:
        return True
    if parsed.scheme not in ("http", "https", ""):
        return True
    path = parsed.path.lower()
    return path.endswith(IGNORED_EXTENSIONS)


def is_same_domain(url: str, base_domain: str) -> bool:
    return urlparse(url).netloc.lower() == base_domain.lower()


def normalize_url(url: str) -> str:
    url, _ = urldefrag(url)
    if url.endswith("/") and urlparse(url).path != "/":
        url = url[:-1]
    return url


def get_domain(url: str) -> str:
    return urlparse(url).netloc.lower()
