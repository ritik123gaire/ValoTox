"""
VLR.gg forum scraper for competitive Valorant community toxicity.

Scrapes forum threads and comments from vlr.gg using BeautifulSoup.
This captures pro-scene drama, ranked toxicity, and competitive slang
that differs from casual subreddit discourse.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger
from tqdm import tqdm

from valotox.config import RAW_DIR

# ── Constants ────────────────────────────────────────────────────────────────
BASE_URL = "https://www.vlr.gg"
FORUM_URL = f"{BASE_URL}/forum"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36 ValoToxResearch/1.0"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

REQUEST_DELAY = 1.5  # seconds between requests (be polite)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _get_soup(url: str) -> BeautifulSoup:
    """Fetch a page and return parsed soup."""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def _clean_text(raw: str) -> str:
    """Strip excessive whitespace and normalise."""
    text = re.sub(r"\s+", " ", raw).strip()
    return text


# ── Thread discovery ─────────────────────────────────────────────────────────

def get_forum_thread_urls(
    pages: int = 20,
    forum_section: str | None = None,
) -> list[dict]:
    """Collect thread URLs from VLR.gg forum listing pages.

    Parameters
    ----------
    pages : int
        Number of forum listing pages to scrape.
    forum_section : str | None
        Optional section filter (e.g., "general", "off-topic").

    Returns
    -------
    list[dict]
        Each dict has ``url``, ``title``, ``replies``.
    """
    threads: list[dict] = []
    seen_urls: set[str] = set()

    for page_num in tqdm(range(1, pages + 1), desc="VLR forum pages"):
        url = f"{FORUM_URL}/?page={page_num}"
        if forum_section:
            url += f"&section={forum_section}"

        try:
            soup = _get_soup(url)
        except Exception as exc:
            logger.warning(f"Failed to fetch {url}: {exc}")
            continue

        # vlr.gg forum thread rows
        thread_items = soup.select(".forum-item")
        if not thread_items:
            # fallback selector
            thread_items = soup.select("a.wf-module-item")

        for item in thread_items:
            link_tag = item.select_one("a") if not item.name == "a" else item
            if link_tag is None:
                continue
            href = link_tag.get("href", "")
            if not href or href in seen_urls:
                continue

            thread_url = href if href.startswith("http") else BASE_URL + href
            seen_urls.add(href)

            title_el = item.select_one(".forum-item-title, .text-of")
            title = _clean_text(title_el.get_text()) if title_el else ""

            reply_el = item.select_one(".forum-item-replies, .post-count")
            replies = 0
            if reply_el:
                nums = re.findall(r"\d+", reply_el.get_text())
                replies = int(nums[0]) if nums else 0

            threads.append(
                {"url": thread_url, "title": title, "replies": replies}
            )

        time.sleep(REQUEST_DELAY)

    logger.info(f"Discovered {len(threads)} VLR.gg forum threads")
    return threads


# ── Thread comment scraping ──────────────────────────────────────────────────

def scrape_thread_comments(thread_url: str, thread_title: str = "") -> list[dict]:
    """Scrape all comments from a single VLR.gg forum thread.

    Returns
    -------
    list[dict]
        Rows with ``text``, ``author``, ``thread_title``, ``thread_url``, ``source``.
    """
    rows: list[dict] = []
    try:
        soup = _get_soup(thread_url)
    except Exception as exc:
        logger.warning(f"Failed to fetch thread {thread_url}: {exc}")
        return rows

    # Post bodies on vlr.gg
    post_bodies = soup.select(".post-body, .forum-post-body")
    if not post_bodies:
        post_bodies = soup.select("[class*='post'] [class*='body']")

    for post in post_bodies:
        text = _clean_text(post.get_text())
        if len(text.split()) < 4:
            continue

        # Try to get author
        author = ""
        author_el = post.find_previous(class_=re.compile(r"author|username|post-header"))
        if author_el:
            a_tag = author_el.select_one("a")
            author = _clean_text(a_tag.get_text()) if a_tag else ""

        rows.append(
            {
                "text": text,
                "author": author,
                "thread_title": thread_title,
                "thread_url": thread_url,
                "source": "vlr_gg",
            }
        )

    return rows


# ── Full pipeline ────────────────────────────────────────────────────────────

def scrape_vlr(
    pages: int = 20,
    max_threads: int | None = None,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """End-to-end VLR.gg forum scraping.

    Parameters
    ----------
    pages : int
        Forum listing pages to scan.
    max_threads : int | None
        Cap on threads to scrape (``None`` = all discovered).
    output_path : Path | str | None
        CSV output path. Defaults to ``data/raw/vlr_raw.csv``.

    Returns
    -------
    pd.DataFrame
    """
    output_path = Path(output_path) if output_path else RAW_DIR / "vlr_raw.csv"

    threads = get_forum_thread_urls(pages=pages)
    if max_threads:
        # prioritise threads with most replies (more likely to have toxicity)
        threads = sorted(threads, key=lambda t: t["replies"], reverse=True)[:max_threads]

    all_rows: list[dict] = []
    for thread in tqdm(threads, desc="Scraping VLR threads"):
        comments = scrape_thread_comments(thread["url"], thread["title"])
        all_rows.extend(comments)
        time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(all_rows)
    logger.info(f"Scraped {len(df):,} comments from {len(threads)} VLR.gg threads")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved → {output_path}")
    return df


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape VLR.gg forums")
    parser.add_argument("--pages", type=int, default=20)
    parser.add_argument("--max-threads", type=int, default=None)
    args = parser.parse_args()

    scrape_vlr(pages=args.pages, max_threads=args.max_threads)
