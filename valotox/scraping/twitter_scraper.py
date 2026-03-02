"""
Twitter / X scraper for Valorant toxicity data.

Uses snscrape (preferred) for historical data, with a fallback to the
official Twitter API v2 via requests.  Targets tweets that discuss or
screenshot in-game toxic chat.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from valotox.config import RAW_DIR, settings

# ── Search queries designed to capture Valorant toxicity discourse ────────
SEARCH_QUERIES: list[str] = [
    "valorant toxic",
    "valorant toxicity",
    "valorant chat ban",
    "valorant harassment",
    "valorant misogyny",
    "valorant reported",
    "valorant gender",
    "valorant kys",
    "valorant slur",
    "valorant muted",
    "valorant ban wave",
    "valorant comms ban",
    "#ValorantToxic",
    "#ValorantChat",
]

MAX_TWEETS_PER_QUERY = 2_000


# ── snscrape-based approach ──────────────────────────────────────────────────

def _scrape_with_snscrape(
    queries: list[str],
    max_per_query: int = MAX_TWEETS_PER_QUERY,
    since: str = "2024-01-01",
) -> list[dict]:
    """Use snscrape to pull historical tweets without API keys."""
    try:
        import snscrape.modules.twitter as sntwitter
    except ImportError:
        logger.warning("snscrape not installed – falling back to API method")
        return []

    rows: list[dict] = []
    seen_ids: set[str] = set()

    for query in tqdm(queries, desc="snscrape queries"):
        full_query = f"{query} since:{since} lang:en"
        try:
            scraper = sntwitter.TwitterSearchScraper(full_query)
            for i, tweet in enumerate(scraper.get_items()):
                if i >= max_per_query:
                    break
                if tweet.id in seen_ids:
                    continue
                seen_ids.add(tweet.id)

                text = tweet.rawContent if hasattr(tweet, "rawContent") else tweet.content
                text = re.sub(r"https?://\S+", "", text).strip()

                if len(text.split()) < 4:
                    continue

                rows.append(
                    {
                        "id": str(tweet.id),
                        "text": text,
                        "username": tweet.user.username,
                        "likes": tweet.likeCount,
                        "retweets": tweet.retweetCount,
                        "created_at": tweet.date.isoformat(),
                        "query": query,
                        "source": "twitter",
                    }
                )
        except Exception as exc:
            logger.warning(f"snscrape error for '{query}': {exc}")

        time.sleep(1.0)

    return rows


# ── Twitter API v2 fallback ──────────────────────────────────────────────────

def _scrape_with_api(
    queries: list[str],
    max_per_query: int = 100,  # API free tier is limited
) -> list[dict]:
    """Use Twitter API v2 recent-search endpoint (requires bearer token)."""
    import requests as req

    bearer = settings.twitter_bearer_token
    if not bearer:
        logger.error("No TWITTER_BEARER_TOKEN set – cannot use API fallback")
        return []

    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {bearer}"}

    rows: list[dict] = []
    seen_ids: set[str] = set()

    for query in tqdm(queries, desc="Twitter API queries"):
        params = {
            "query": f"{query} lang:en -is:retweet",
            "max_results": min(max_per_query, 100),
            "tweet.fields": "created_at,public_metrics,author_id",
        }

        try:
            resp = req.get(url, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning(f"Twitter API error for '{query}': {exc}")
            continue

        for tweet in data.get("data", []):
            tid = tweet["id"]
            if tid in seen_ids:
                continue
            seen_ids.add(tid)

            text = re.sub(r"https?://\S+", "", tweet["text"]).strip()
            if len(text.split()) < 4:
                continue

            metrics = tweet.get("public_metrics", {})
            rows.append(
                {
                    "id": tid,
                    "text": text,
                    "username": tweet.get("author_id", ""),
                    "likes": metrics.get("like_count", 0),
                    "retweets": metrics.get("retweet_count", 0),
                    "created_at": tweet.get("created_at", ""),
                    "query": query,
                    "source": "twitter",
                }
            )

        time.sleep(1.0)

    return rows


# ── Public API ───────────────────────────────────────────────────────────────

def scrape_twitter(
    queries: list[str] | None = None,
    max_per_query: int = MAX_TWEETS_PER_QUERY,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Scrape Valorant-related tweets.

    Attempts snscrape first; falls back to the official API.

    Parameters
    ----------
    queries : list[str] | None
        Custom search queries. Defaults to ``SEARCH_QUERIES``.
    max_per_query : int
        Maximum tweets per query.
    output_path : Path | str | None
        CSV destination. Defaults to ``data/raw/twitter_raw.csv``.

    Returns
    -------
    pd.DataFrame
    """
    queries = queries or SEARCH_QUERIES
    output_path = Path(output_path) if output_path else RAW_DIR / "twitter_raw.csv"

    rows = _scrape_with_snscrape(queries, max_per_query=max_per_query)
    if not rows:
        logger.info("snscrape returned no results – trying Twitter API v2")
        rows = _scrape_with_api(queries, max_per_query=min(max_per_query, 100))

    df = pd.DataFrame(rows)
    logger.info(f"Collected {len(df):,} tweets")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved → {output_path}")
    return df


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape Valorant tweets")
    parser.add_argument("--max-per-query", type=int, default=2000)
    args = parser.parse_args()

    scrape_twitter(max_per_query=args.max_per_query)
