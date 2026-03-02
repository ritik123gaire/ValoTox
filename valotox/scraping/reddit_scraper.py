"""
Reddit scraper for Valorant subreddits using PRAW.

Collects comments from multiple Valorant-related subreddits with
keyword-stratified sampling to ensure balanced toxic/clean representation.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import praw
from loguru import logger
from tqdm import tqdm

from valotox.config import RAW_DIR, settings
from valotox.lexicon import ALL_TOXIC_KEYWORDS, get_regex_pattern

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_SUBREDDITS: list[str] = [
    "VALORANT",
    "ValorantMemes",
    "ValorantCompetitive",
    "AgentAcademy",
]

SORT_MODES = ["hot", "top", "new", "controversial"]

MIN_WORD_COUNT = 4  # filter very short / bot comments


# ── Helper ───────────────────────────────────────────────────────────────────

def _get_reddit_client() -> praw.Reddit:
    """Create an authenticated Reddit client from env settings."""
    return praw.Reddit(
        client_id=settings.reddit_client_id,
        client_secret=settings.reddit_client_secret,
        user_agent=settings.reddit_user_agent,
    )


def _safe_text(body: str) -> str:
    """Basic normalisation: collapse whitespace, strip markdown links."""
    import re

    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", body)  # [text](url) → text
    text = re.sub(r"https?://\S+", "", text)  # strip bare URLs
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Main scraping function ───────────────────────────────────────────────────

def scrape_subreddits(
    subreddits: list[str] | None = None,
    sort_modes: list[str] | None = None,
    posts_per_sort: int = 500,
    replace_more_limit: int = 0,
    rate_limit_seconds: float = 0.5,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Scrape comments from Valorant subreddits.

    Parameters
    ----------
    subreddits : list[str] | None
        Subreddit names (without r/). Defaults to ``DEFAULT_SUBREDDITS``.
    sort_modes : list[str] | None
        Reddit sort modes to iterate. Defaults to ``["hot", "top", "new"]``.
    posts_per_sort : int
        Number of posts to fetch per sort mode per subreddit.
    replace_more_limit : int
        How many "More Comments" objects to resolve (0 = fast, higher = deeper).
    rate_limit_seconds : float
        Sleep between posts to respect Reddit rate limits.
    output_path : Path | str | None
        CSV path to write results. Defaults to ``data/raw/reddit_raw.csv``.

    Returns
    -------
    pd.DataFrame
    """
    subreddits = subreddits or DEFAULT_SUBREDDITS
    sort_modes = sort_modes or ["hot", "top", "new"]
    output_path = Path(output_path) if output_path else RAW_DIR / "reddit_raw.csv"

    reddit = _get_reddit_client()
    rows: list[dict] = []
    seen_ids: set[str] = set()

    for sub_name in subreddits:
        sub = reddit.subreddit(sub_name)
        logger.info(f"Scraping r/{sub_name}")

        for mode in sort_modes:
            logger.info(f"  sort={mode}, limit={posts_per_sort}")
            fetcher = getattr(sub, mode)
            kwargs = {"limit": posts_per_sort}
            if mode in ("top", "controversial"):
                kwargs["time_filter"] = "all"

            for post in tqdm(
                fetcher(**kwargs),
                desc=f"r/{sub_name}/{mode}",
                total=posts_per_sort,
                leave=False,
            ):
                try:
                    post.comments.replace_more(limit=replace_more_limit)
                except Exception:
                    pass

                for comment in post.comments.list():
                    cid = comment.id
                    if cid in seen_ids:
                        continue
                    seen_ids.add(cid)

                    body = _safe_text(comment.body)
                    if len(body.split()) < MIN_WORD_COUNT:
                        continue

                    rows.append(
                        {
                            "id": cid,
                            "text": body,
                            "subreddit": sub_name,
                            "post_title": post.title,
                            "upvotes": comment.score,
                            "created_utc": datetime.fromtimestamp(
                                comment.created_utc, tz=timezone.utc
                            ).isoformat(),
                            "source": "reddit",
                        }
                    )

                time.sleep(rate_limit_seconds)

    df = pd.DataFrame(rows)
    logger.info(f"Scraped {len(df):,} comments across {len(subreddits)} subreddits")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved → {output_path}")
    return df


# ── Keyword-stratified sampling ──────────────────────────────────────────────

def stratified_sample(
    df: pd.DataFrame,
    toxic_n: int = 5_000,
    clean_n: int = 5_000,
    seed: int = 42,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Produce a balanced sample by oversampling keyword-matching comments.

    Parameters
    ----------
    df : pd.DataFrame
        Raw scraped dataframe (must have ``text`` column).
    toxic_n, clean_n : int
        Desired counts for keyword-matching / non-matching comments.
    seed : int
        Random state for reproducibility.
    output_path : Path | str | None
        Where to save. Defaults to ``data/raw/valotox_raw.csv``.

    Returns
    -------
    pd.DataFrame
    """
    output_path = Path(output_path) if output_path else RAW_DIR / "valotox_raw.csv"
    pattern = get_regex_pattern()

    mask = df["text"].str.contains(pattern, case=False, na=False, regex=True)
    toxic_pool = df[mask]
    clean_pool = df[~mask]

    logger.info(
        f"Keyword match: {len(toxic_pool):,} toxic, {len(clean_pool):,} clean"
    )

    toxic_sample = toxic_pool.sample(n=min(toxic_n, len(toxic_pool)), random_state=seed)
    clean_sample = clean_pool.sample(n=min(clean_n, len(clean_pool)), random_state=seed)

    final = (
        pd.concat([toxic_sample, clean_sample])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    final.to_csv(output_path, index=False)
    logger.info(f"Stratified sample: {len(final):,} rows → {output_path}")
    return final


# ── CLI entry-point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape Valorant subreddits")
    parser.add_argument("--posts", type=int, default=500, help="Posts per sort mode")
    parser.add_argument("--toxic-n", type=int, default=5000)
    parser.add_argument("--clean-n", type=int, default=5000)
    args = parser.parse_args()

    raw_df = scrape_subreddits(posts_per_sort=args.posts)
    stratified_sample(raw_df, toxic_n=args.toxic_n, clean_n=args.clean_n)
