"""
Web page reader for Ark-Chat.
Fetches URLs, extracts clean text, and prepares content for model context injection.
"""

import re
import urllib.request
import urllib.error
from html.parser import HTMLParser


# Tags whose content we want to skip entirely
_SKIP_TAGS = frozenset([
    "script", "style", "noscript", "svg", "path", "meta", "link",
    "head", "iframe", "object", "embed", "applet", "nav", "footer",
    "header",
])


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML-to-text converter that preserves paragraph structure."""

    def __init__(self):
        super().__init__()
        self._pieces: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        if tag in ("br", "p", "div", "h1", "h2", "h3", "h4", "li", "tr"):
            self._pieces.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        if tag in ("p", "div", "h1", "h2", "h3", "h4", "li", "tr", "td"):
            self._pieces.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._pieces.append(data)

    def get_text(self) -> str:
        raw = "".join(self._pieces)
        # Collapse whitespace while preserving paragraph breaks
        lines = []
        for line in raw.splitlines():
            cleaned = " ".join(line.split())
            if cleaned:
                lines.append(cleaned)
        return "\n".join(lines)


# Regex to find URLs in user text
_URL_PATTERN = re.compile(
    r"https?://[^\s<>\"')\]]+",
    re.IGNORECASE,
)


def extract_urls(text: str) -> list[str]:
    """Find all HTTP/HTTPS URLs in the given text."""
    return _URL_PATTERN.findall(text)


def fetch_page_text(url: str, max_chars: int = 8000, timeout: int = 10) -> str:
    """
    Fetch a web page and return cleaned plain text.

    Args:
        url: The URL to fetch.
        max_chars: Maximum characters to return (truncated with notice).
        timeout: HTTP request timeout in seconds.

    Returns:
        Cleaned text content of the page, or an error message.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; ArkChat/1.0)",
                "Accept": "text/html,application/xhtml+xml,text/plain",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            charset = "utf-8"
            if "charset=" in content_type:
                charset = content_type.split("charset=")[-1].split(";")[0].strip()

            raw_bytes = resp.read(500_000)  # Read max 500KB
            html = raw_bytes.decode(charset, errors="replace")

        # Check if it's plain text
        if "text/plain" in content_type:
            text = html
        else:
            parser = _HTMLTextExtractor()
            parser.feed(html)
            text = parser.get_text()

        # Clean and truncate
        text = text.strip()
        if not text:
            return f"[No readable text content found at {url}]"

        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[... content truncated at {max_chars} characters]"

        return text

    except urllib.error.HTTPError as e:
        return f"[Error fetching {url}: HTTP {e.code} {e.reason}]"
    except urllib.error.URLError as e:
        return f"[Error fetching {url}: {e.reason}]"
    except Exception as e:
        return f"[Error fetching {url}: {e}]"


def build_web_context(user_message: str, max_total_chars: int = 6000) -> tuple[str, str]:
    """
    Detect URLs in a user message, fetch their content, and build a context block.

    Args:
        user_message: The raw user input.
        max_total_chars: Maximum total characters for all web content combined.

    Returns:
        Tuple of (context_block, cleaned_user_message).
        context_block is empty string if no URLs found.
    """
    urls = extract_urls(user_message)
    if not urls:
        return "", user_message

    # Remove URLs from the user message for cleaner prompting
    cleaned_msg = user_message
    for url in urls:
        cleaned_msg = cleaned_msg.replace(url, "").strip()

    # Remove leftover phrases like "read this:" or "check this link"
    cleaned_msg = re.sub(
        r"\b(read|check|look at|open|fetch|visit|see)\s+(this|the)?\s*(link|url|page|website|site)?\s*:?\s*",
        "",
        cleaned_msg,
        flags=re.IGNORECASE,
    ).strip()

    if not cleaned_msg:
        cleaned_msg = "Summarize and explain the content from the provided web page."

    # Fetch each URL
    per_url_limit = max_total_chars // len(urls)
    context_parts = []
    for url in urls:
        print(f"🌐 Fetching: {url}")
        text = fetch_page_text(url, max_chars=per_url_limit)
        context_parts.append(f"--- Web content from {url} ---\n{text}\n--- End of web content ---")

    context_block = "\n\n".join(context_parts)
    return context_block, cleaned_msg
