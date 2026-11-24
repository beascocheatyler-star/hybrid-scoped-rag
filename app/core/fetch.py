"""Fetch remote sources: web pages (with an SSRF guard) and YouTube transcripts.

SSRF guard. A URL source makes the server issue an HTTP request on the user's
behalf, so the destination is checked before every hop:

* scheme must be http or https, port must be 80/443 or above 1023;
* the host is resolved and EVERY address must be public: loopback, private
  (RFC 1918 / ULA), link-local (incl. 169.254.169.254 cloud metadata),
  multicast, reserved and unspecified addresses are refused;
* redirects are followed manually (max 5) and each target is re-checked;
* the body is capped at URL_MAX_BYTES (default 10 MB).

Known limitation: the address is resolved here and again by the HTTP client,
so a DNS-rebinding attacker controlling a hostname's TTL could race the two
lookups. Pinning the connection to the checked IP would close that and is
noted in the README. Set URL_FETCH_ALLOW_PRIVATE=true only for trusted
self-hosted setups that need intranet pages.
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from app.core.extract import Extracted, ExtractionError, extract_doc, html_to_doc, segments_to_doc

MAX_REDIRECTS = 5
USER_AGENT = "hybrid-scoped-rag/2 (+self-hosted notebook)"


class FetchError(ExtractionError):
    pass


def _max_bytes() -> int:
    try:
        return int(os.getenv("URL_MAX_BYTES", str(10 * 1024 * 1024)))
    except ValueError:
        return 10 * 1024 * 1024


def _allow_private() -> bool:
    return os.getenv("URL_FETCH_ALLOW_PRIVATE", "false").lower() in {"1", "true", "yes"}


def _is_public(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return not (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_multicast or addr.is_reserved or addr.is_unspecified)


# Injectable for tests: hostname -> list of IP strings.
_resolver: Callable[[str], List[str]] = lambda host: sorted(
    {ai[4][0] for ai in socket.getaddrinfo(host, None)})


def check_url(url: str) -> urllib.parse.SplitResult:
    """Raise FetchError unless ``url`` is a permitted public http(s) URL."""
    try:
        parts = urllib.parse.urlsplit(url.strip())
    except ValueError as exc:
        raise FetchError("Invalid URL: {}".format(exc))
    if parts.scheme not in ("http", "https"):
        raise FetchError("Only http and https URLs are allowed.")
    host = parts.hostname
    if not host:
        raise FetchError("The URL has no host.")
    try:
        port = parts.port
    except ValueError:
        raise FetchError("Invalid port in URL.")
    if port is not None and port not in (80, 443) and port < 1024:
        raise FetchError("Port {} is not allowed.".format(port))
    if _allow_private():
        return parts
    try:
        ips = [host] if _looks_like_ip(host) else _resolver(host)
    except (socket.gaierror, UnicodeError) as exc:
        raise FetchError("Could not resolve host {!r}: {}".format(host, exc))
    if not ips:
        raise FetchError("Host {!r} did not resolve.".format(host))
    for ip in ips:
        if not _is_public(ip):
            raise FetchError(
                "Refusing to fetch {!r}: it resolves to a non-public address ({}).".format(host, ip))
    return parts


def _looks_like_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_opener = urllib.request.build_opener(_NoRedirect)


@dataclass
class Fetched:
    url: str           # final URL after redirects
    content_type: str
    data: bytes


def fetch_bytes(url: str) -> Fetched:
    limit = _max_bytes()
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        check_url(current)
        req = urllib.request.Request(current, headers={"User-Agent": USER_AGENT,
                                                       "Accept": "text/html,application/xhtml+xml,text/plain,application/pdf;q=0.9,*/*;q=0.5"})
        try:
            resp = _opener.open(req, timeout=20)
        except urllib.error.HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308) and exc.headers.get("Location"):
                current = urllib.parse.urljoin(current, exc.headers["Location"])
                continue
            raise FetchError("The server answered HTTP {}.".format(exc.code))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise FetchError("Could not fetch the URL: {}".format(exc))
        with resp:
            declared = resp.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > limit:
                raise FetchError("The page is larger than the {} MB limit.".format(limit // 1048576))
            data = resp.read(limit + 1)
            if len(data) > limit:
                raise FetchError("The page is larger than the {} MB limit.".format(limit // 1048576))
            return Fetched(resp.geturl() or current,
                           (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower(), data)
    raise FetchError("Too many redirects.")


def _canonical(html: str, base: str) -> str:
    m = re.search(r'<link[^>]+rel=["\']canonical["\'][^>]*>', html, re.I)
    if m:
        h = re.search(r'href=["\']([^"\']+)', m.group(0))
        if h:
            return urllib.parse.urljoin(base, h.group(1))
    return base


def page_to_doc(fetched: Fetched) -> Tuple[Extracted, str]:
    """Turn fetched bytes into (document, canonical_url)."""
    ct = fetched.content_type
    if ct == "application/pdf" or fetched.url.lower().endswith(".pdf"):
        return extract_doc("page.pdf", fetched.data), fetched.url
    if ct.startswith("text/plain"):
        return extract_doc("page.txt", fetched.data), fetched.url
    html = fetched.data.decode("utf-8", errors="replace")
    doc = None
    try:  # trafilatura gives much better main-content extraction if present
        import trafilatura

        main = trafilatura.extract(html, include_comments=False, include_tables=True,
                                   output_format="markdown")
        if main and main.strip():
            doc = extract_doc("page.md", main.encode("utf-8"))
            doc.title = html_to_doc(html).title
    except ImportError:
        pass
    if doc is None:
        doc = html_to_doc(html)
    if not doc.text.strip():
        raise FetchError("No readable text found on the page.")
    return doc, _canonical(html, fetched.url)


# --- YouTube -------------------------------------------------------------------

_YT_ID = re.compile(r"(?:v=|/shorts/|youtu\.be/|/embed/|/live/)([A-Za-z0-9_-]{11})")


def youtube_id(url: str) -> str:
    m = _YT_ID.search(url)
    if not m:
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", url.strip()):
            return url.strip()
        raise FetchError("Could not find a YouTube video id in that URL.")
    return m.group(1)


# Injectable for tests: video_id -> [(t_start_ms, t_end_ms, text)].
_transcript_fetcher: Optional[Callable[[str], List[Tuple[int, int, str]]]] = None


def set_transcript_fetcher(fn) -> None:
    global _transcript_fetcher
    _transcript_fetcher = fn


def _default_transcript(video_id: str) -> List[Tuple[int, int, str]]:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:
        raise FetchError("YouTube sources require 'youtube-transcript-api' "
                         "(pip install youtube-transcript-api).") from exc
    try:
        api = YouTubeTranscriptApi()
        items = api.fetch(video_id)
        rows = [(it.start, it.duration, it.text) for it in items]
    except AttributeError:  # pre-1.0 API
        rows = [(d["start"], d["duration"], d["text"])
                for d in YouTubeTranscriptApi.get_transcript(video_id)]
    except Exception as exc:
        raise FetchError("No transcript is available for this video: {}".format(exc)) from exc
    return [(int(s * 1000), int((s + d) * 1000), t) for s, d, t in rows]


def youtube_doc(url: str) -> Extracted:
    vid = youtube_id(url)
    segs = (_transcript_fetcher or _default_transcript)(vid)
    # Merge short caption lines into ~30 s segments so chunks read naturally
    # but every segment still carries a real timestamp range.
    merged: List[Tuple[int, int, str]] = []
    for t0, t1, text in segs:
        text = " ".join(text.replace("\n", " ").split())
        if not text:
            continue
        if merged and t1 - merged[-1][0] <= 30000:
            a, _, s = merged[-1]
            merged[-1] = (a, t1, s + " " + text)
        else:
            merged.append((t0, t1, text))
    doc = segments_to_doc(merged)
    if not doc.text.strip():
        raise FetchError("The transcript for this video is empty.")
    doc.title = "YouTube video {}".format(vid)
    return doc
