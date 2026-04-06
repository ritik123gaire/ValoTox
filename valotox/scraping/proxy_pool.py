"""
ProxyScrape proxy list + rotation for HTTP(S) crawlers.

Fetches proxies from ProxyScrape (Account API with ``api-token``, legacy v2
``auth``, or public free list) and exposes a small rotator for ``requests``.
"""

from __future__ import annotations

import itertools
import threading
from typing import Iterator

import requests
from loguru import logger

from valotox.config import settings

# Public free list (no key) — v4
PUBLIC_PROXY_LIST_URL = "https://api.proxyscrape.com/v4/free-proxy-list/get"

# Legacy v2 (deprecated but still used with ``auth`` query param)
LEGACY_V2_URL = "https://api.proxyscrape.com/v2/"


def _parse_ip_port_lines(text: str) -> list[str]:
    lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line and not line.startswith("{"):
            lines.append(line)
    return lines


def fetch_proxyscrape_proxies(
    limit: int = 500,
    protocol: str = "http",
) -> list[str]:
    """Return ``host:port`` strings from ProxyScrape.

    Resolution order:

    1. ``PROXYSCRAPE_SUBACCOUNT_ID`` + ``PROXYSCRAPE_API_KEY`` → Account API
       datacenter shared proxy list (``api-token`` header).
    2. ``PROXYSCRAPE_API_KEY`` only → legacy v2 ``auth`` query (premium lists).
    3. No key → public v4 free proxy list (may be lower quality).
    """
    api_key = (settings.proxyscrape_api_key or "").strip()
    sub_id = (settings.proxyscrape_subaccount_id or "").strip()
    dc_path = (settings.proxyscrape_datacenter_path or "datacenter_shared").strip()

    if api_key and sub_id:
        url = f"https://api.proxyscrape.com/v4/account/{sub_id}/{dc_path}/proxy-list"
        try:
            resp = requests.get(
                url,
                headers={"api-token": api_key},
                params={"protocol": protocol, "type": "getproxies"},
                timeout=45,
            )
            resp.raise_for_status()
            proxies = _parse_ip_port_lines(resp.text)
            logger.info(f"ProxyScrape Account API: fetched {len(proxies)} proxies")
            return proxies[:limit]
        except Exception as exc:
            logger.warning(f"ProxyScrape Account API failed: {exc}; trying fallbacks")

    if api_key:
        try:
            resp = requests.get(
                LEGACY_V2_URL,
                params={
                    "request": "getproxies",
                    "protocol": protocol,
                    "timeout": 10000,
                    "country": "all",
                    "ssl": "all",
                    "anonymity": "all",
                    "auth": api_key,
                },
                timeout=45,
            )
            resp.raise_for_status()
            proxies = _parse_ip_port_lines(resp.text)
            logger.info(f"ProxyScrape legacy v2 (auth): fetched {len(proxies)} proxies")
            return proxies[:limit]
        except Exception as exc:
            logger.warning(f"ProxyScrape legacy v2 failed: {exc}; trying public free list")

    try:
        resp = requests.get(
            PUBLIC_PROXY_LIST_URL,
            params={
                "request": "getproxies",
                "protocol": protocol,
                "timeout": 10000,
                "country": "all",
                "limit": min(limit, 2000),
            },
            timeout=45,
        )
        resp.raise_for_status()
        proxies = _parse_ip_port_lines(resp.text)
        logger.info(f"ProxyScrape public free list: fetched {len(proxies)} proxies")
        return proxies[:limit]
    except Exception as exc:
        logger.error(f"ProxyScrape public proxy list failed: {exc}")
        return []


def proxy_url_for_requests(host_port: str) -> dict[str, str]:
    """Build ``requests`` proxies dict for an ``ip:port`` HTTP proxy."""
    return {"http": f"http://{host_port}", "https": f"http://{host_port}"}


class ProxyRotator:
    """Thread-safe round-robin over a proxy list with optional refresh."""

    def __init__(self, proxies: list[str]) -> None:
        self._lock = threading.Lock()
        self._cycle: Iterator[str] | None = None
        self._proxies = [p for p in proxies if p and ":" in p]
        if self._proxies:
            self._cycle = itertools.cycle(self._proxies)

    @property
    def has_proxies(self) -> bool:
        return bool(self._proxies)

    def next_proxy(self) -> str | None:
        if not self._cycle:
            return None
        with self._lock:
            return next(self._cycle)

    def apply_to_session(self, session: requests.Session) -> None:
        """Set ``session.proxies`` to the next proxy in rotation."""
        p = self.next_proxy()
        if p:
            session.proxies.update(proxy_url_for_requests(p))

    def refresh(self, proxies: list[str]) -> None:
        with self._lock:
            self._proxies = [p for p in proxies if p and ":" in p]
            self._cycle = itertools.cycle(self._proxies) if self._proxies else None


def session_get(
    session: requests.Session,
    rotator: ProxyRotator | None,
    url: str,
    **kwargs,
) -> requests.Response:
    """``session.get`` with optional proxy rotation."""
    if rotator and rotator.has_proxies:
        rotator.apply_to_session(session)
    else:
        session.proxies = {}
    return session.get(url, **kwargs)


def get_rotator() -> ProxyRotator | None:
    """Build a rotator when ``use_proxyscrape_proxies`` is enabled in settings."""
    if not settings.use_proxyscrape_proxies:
        return None

    proxies = fetch_proxyscrape_proxies(limit=settings.proxyscrape_proxy_limit)
    if not proxies:
        logger.warning("No ProxyScrape proxies available; crawling without proxy")
        return None
    return ProxyRotator(proxies)
