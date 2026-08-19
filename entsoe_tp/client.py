"""
HTTP client for the Transparency Platform RESTful API.

Endpoint and limits, from the platform's own documentation:

* ``https://web-api.tp.entsoe.eu/api`` -- HTTPS only.
* The security token goes in the ``securityToken`` query parameter. A missing or
  invalid token returns HTTP 401.
* A single request may cover at most one year.
* At most 100 ``TimeSeries`` come back per response; further pages are fetched
  with ``offset=100``, ``200`` and so on.
* 400 requests per minute per token. Exceeding it gets the *token* (not the IP)
  banned for roughly ten minutes, so the client throttles rather than relying on
  retries to sort it out.
"""

import hashlib
import os
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from .parser import TransparencyError

BASE_URL = "https://web-api.tp.entsoe.eu/api"

TIME_FORMAT = "%Y%m%d%H%M"

# The platform pages TimeSeries in blocks of this size.
PAGE_SIZE = 100

# 400 req/min is the documented ceiling; ~6 req/s leaves headroom for the
# platform's own accounting and keeps well clear of a ten-minute token ban.
MIN_REQUEST_INTERVAL = 1 / 6

MAX_RETRIES = 5


class TokenError(TransparencyError):
    """Raised when no security token is available, or the platform rejects it."""


def load_token(env_path=None):
    """Return the ENTSO-E security token.

    Looks at ``ENTSOE_API_TOKEN`` first, then falls back to a ``.env`` file in
    the project root. The file is parsed by hand rather than pulling in
    ``python-dotenv`` for a dozen lines of work.
    """
    token = os.environ.get("ENTSOE_API_TOKEN")
    if token:
        return token.strip()

    if env_path is None:
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")

    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                if key.strip() == "ENTSOE_API_TOKEN":
                    return value.strip().strip("'\"")

    raise TokenError(
        "No ENTSO-E security token found. Set the ENTSOE_API_TOKEN environment "
        "variable or add ENTSOE_API_TOKEN=... to a .env file in the project root. "
        "See entsoe_tp/README.md for how to obtain one."
    )


def _redact(text):
    """Remove the security token from anything that might be logged or raised."""
    if not text:
        return text
    out = []
    for part in str(text).replace("&", " &").split():
        if "securityToken=" in part:
            head, _, _ = part.partition("securityToken=")
            out.append(head + "securityToken=<redacted>")
        else:
            out.append(part)
    return " ".join(out).replace(" &", "&")


def _count_timeseries(xml_text):
    """Number of TimeSeries elements in a response, used to detect a full page."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return 0
    return sum(1 for child in root if child.tag.rpartition("}")[2] == "TimeSeries")


def _month_chunks(start, end):
    """Split ``[start, end)`` into month-long UTC windows.

    Months sit comfortably inside both the one-year request limit and the
    100-TimeSeries response limit (day-ahead prices arrive as roughly one
    TimeSeries per day), and give useful progress granularity on a long download.
    """
    chunks = []
    cursor = start
    while cursor < end:
        nxt = min(cursor + pd.DateOffset(months=1), end)
        chunks.append((cursor, nxt))
        cursor = nxt
    return chunks


class TransparencyClient:
    """Fetches raw XML documents from the Transparency Platform."""

    def __init__(self, token=None, session=None, cache_dir=None, timeout=(10, 120)):
        self.token = token or load_token()
        self.session = session or requests.Session()
        self.cache_dir = cache_dir
        self.timeout = timeout
        self._last_request = 0.0

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    # -- caching ---------------------------------------------------------

    def _cache_path(self, params):
        # The token is deliberately excluded from the key so the cache stays
        # valid across token rotations -- and so it never lands in a filename.
        material = "&".join(f"{k}={v}" for k, v in sorted(params.items())
                            if k != "securityToken")
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
        return os.path.join(self.cache_dir, f"{digest}.xml")

    # -- HTTP ------------------------------------------------------------

    def _throttle(self):
        elapsed = time.monotonic() - self._last_request
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request = time.monotonic()

    def _get(self, params):
        """One GET, with cache, throttle and retry. Returns the response body."""
        if self.cache_dir:
            path = self._cache_path(params)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as handle:
                    return handle.read()

        request_params = dict(params, securityToken=self.token)

        for attempt in range(MAX_RETRIES):
            self._throttle()
            try:
                response = self.session.get(
                    BASE_URL, params=request_params, timeout=self.timeout
                )
            except requests.RequestException as exc:
                if attempt == MAX_RETRIES - 1:
                    raise TransparencyError(f"Request failed: {_redact(exc)}") from None
                time.sleep(2 ** attempt)
                continue

            if response.status_code == 401:
                raise TokenError(
                    "The platform rejected the security token (HTTP 401). Check "
                    "ENTSOE_API_TOKEN, and that the account is approved for API "
                    "access and not suspended."
                )

            if response.status_code == 429 or response.status_code >= 500:
                if attempt == MAX_RETRIES - 1:
                    raise TransparencyError(
                        f"Giving up after {MAX_RETRIES} attempts "
                        f"(last status {response.status_code})"
                    )
                # 429 means the token is close to (or already under) a ban that
                # lasts about ten minutes, so back off hard rather than politely.
                backoff = min(300, 30 * 2 ** attempt) if response.status_code == 429 \
                    else 2 ** attempt
                time.sleep(backoff)
                continue

            if response.status_code != 200:
                raise TransparencyError(
                    f"Unexpected status {response.status_code}: "
                    f"{_redact(response.text[:400])}"
                )

            body = response.text
            if self.cache_dir:
                with open(self._cache_path(params), "w", encoding="utf-8") as handle:
                    handle.write(body)
            return body

        raise TransparencyError("Exhausted retries without a response")

    # -- public API ------------------------------------------------------

    def fetch(self, params, start, end, progress=None):
        """Fetch every document covering ``[start, end)``.

        ``start`` and ``end`` are tz-aware UTC timestamps. ``params`` holds the
        document-specific query parameters (``documentType``, domains, and so on);
        the time window, pagination and token are added here.

        Returns a list of raw XML strings.
        """
        documents = []
        chunks = _month_chunks(start, end)

        for number, (chunk_start, chunk_end) in enumerate(chunks, 1):
            window = {
                "periodStart": chunk_start.strftime(TIME_FORMAT),
                "periodEnd": chunk_end.strftime(TIME_FORMAT),
            }

            offset = 0
            while True:
                query = dict(params, **window)
                if offset:
                    query["offset"] = str(offset)

                body = self._get(query)
                documents.append(body)

                # A full page implies there may be more behind it.
                if _count_timeseries(body) < PAGE_SIZE:
                    break
                offset += PAGE_SIZE

            if progress:
                progress(number, len(chunks), chunk_start, chunk_end)

        return documents
