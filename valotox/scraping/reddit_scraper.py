"""
Reddit scraper for Valorant subreddits.

Primary path uses BeautifulSoup on old.reddit.com HTML pages so scraping can
work without API credentials. Falls back to PRAW when credentials are present.
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import praw
import requests
import zstandard as zstd
from bs4 import BeautifulSoup
from loguru import logger
from tqdm import tqdm

from valotox.config import RAW_DIR, settings
from valotox.scraping.proxy_pool import get_rotator, session_get
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
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36 ValoToxResearch/1.0"
    )
}


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


def _epoch_to_iso(value: object) -> str:
    """Convert unix timestamp or string to ISO-8601 UTC string."""
    if value is None:
        return datetime.now(tz=timezone.utc).isoformat()

    try:
        ts = float(value)
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except Exception:
        txt = str(value).strip()
        return txt or datetime.now(tz=timezone.utc).isoformat()


def stream_pushshift_dumps(
    input_paths: list[str] | None = None,
    input_glob: str | None = None,
    subreddits: list[str] | None = None,
    output_path: Path | str | None = None,
    keyword_only: bool = False,
    include_comments: bool = True,
    include_submissions: bool = True,
    min_words: int = MIN_WORD_COUNT,
    max_rows: int | None = None,
    dedupe: bool = True,
    log_every: int = 100_000,
) -> Path:
    """Stream Pushshift/Academic Torrents .zst dumps into reddit_raw.csv schema.

    Notes
    -----
    - Reads NDJSON line-by-line from zstd files without loading full files.
    - Supports both comments (RC_*.zst) and submissions (RS_*.zst).
    - Writes CSV columns compatible with the existing data pipeline.
    """
    if not include_comments and not include_submissions:
        raise ValueError("At least one of include_comments/include_submissions must be True")

    resolved_files: list[Path] = []
    if input_paths:
        resolved_files.extend(Path(p) for p in input_paths)
    if input_glob:
        resolved_files.extend(sorted(Path().glob(input_glob)))

    files = [p for p in resolved_files if p.exists() and p.suffix.lower() == ".zst"]
    if not files:
        raise FileNotFoundError("No valid .zst files found for streaming input")

    output_csv = Path(output_path) if output_path else RAW_DIR / "reddit_raw.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    allowed_subs = {s.lower() for s in (subreddits or DEFAULT_SUBREDDITS)}
    pattern = get_regex_pattern()
    keyword_re = re.compile(pattern, flags=re.IGNORECASE)

    written = 0
    seen_ids: set[str] = set()
    by_kind = {"comment": 0, "submission": 0}

    logger.info(f"Streaming {len(files)} .zst file(s) into {output_csv}")
    with output_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "id",
                "text",
                "subreddit",
                "post_title",
                "upvotes",
                "created_utc",
                "source",
            ],
        )
        writer.writeheader()

        for file_path in files:
            name = file_path.name.lower()
            is_comment_file = name.startswith("rc_") or "comment" in name
            is_submission_file = name.startswith("rs_") or "submission" in name
            if is_comment_file and not include_comments:
                logger.info(f"Skipping comments file {file_path.name}")
                continue
            if is_submission_file and not include_submissions:
                logger.info(f"Skipping submissions file {file_path.name}")
                continue

            logger.info(f"Reading {file_path.name}")
            dctx = zstd.ZstdDecompressor()
            with file_path.open("rb") as compressed:
                with dctx.stream_reader(compressed) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                    for line_no, line in enumerate(text_stream, start=1):
                        if not line.strip():
                            continue

                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        subreddit = str(obj.get("subreddit") or "").strip()
                        if not subreddit or subreddit.lower() not in allowed_subs:
                            continue

                        if "body" in obj:
                            kind = "comment"
                            if not include_comments:
                                continue
                            raw_id = str(obj.get("id") or "")
                            text = _safe_text(str(obj.get("body") or ""))
                            post_title = ""
                        else:
                            kind = "submission"
                            if not include_submissions:
                                continue
                            raw_id = str(obj.get("id") or "")
                            title = _safe_text(str(obj.get("title") or ""))
                            selftext = _safe_text(str(obj.get("selftext") or ""))
                            text = (title + " " + selftext).strip() if selftext else title
                            post_title = title

                        if not raw_id or not text:
                            continue

                        if len(text.split()) < min_words:
                            continue

                        if keyword_only and keyword_re.search(text) is None:
                            continue

                        record_id = f"{kind}_{raw_id}"
                        if dedupe:
                            if record_id in seen_ids:
                                continue
                            seen_ids.add(record_id)

                        score = int(obj.get("score") or 0)
                        created = _epoch_to_iso(obj.get("created_utc"))
                        writer.writerow(
                            {
                                "id": record_id,
                                "text": text,
                                "subreddit": subreddit,
                                "post_title": post_title,
                                "upvotes": score,
                                "created_utc": created,
                                "source": "pushshift",
                            }
                        )

                        written += 1
                        by_kind[kind] += 1
                        if log_every > 0 and written % log_every == 0:
                            logger.info(
                                f"Wrote {written:,} rows "
                                f"(comments={by_kind['comment']:,}, submissions={by_kind['submission']:,})"
                            )

                        if max_rows is not None and written >= max_rows:
                            logger.info("Reached --max-rows limit")
                            logger.info(
                                f"Final rows: {written:,} "
                                f"(comments={by_kind['comment']:,}, submissions={by_kind['submission']:,})"
                            )
                            return output_csv

                        if line_no % 1_000_000 == 0:
                            logger.info(f"Processed {line_no:,} lines in {file_path.name}")

    logger.info(
        f"Final rows: {written:,} "
        f"(comments={by_kind['comment']:,}, submissions={by_kind['submission']:,})"
    )
    return output_csv


def _scrape_reddit_sociavault(
    subreddits: list[str],
    posts_per_subreddit: int = 100,
    comments_per_post: int = 100,
) -> list[dict]:
    """Scrape Reddit via SociaVault API (post + comments endpoints)."""
    api_key = (settings.sociavault_api_key or "").strip()
    if not api_key or not settings.use_sociavault_reddit:
        return []

    base_urls = [
        "https://api.sociavault.com/v1/scrape/reddit",
        "https://api.sociavault.com/v1/reddit",
        "https://api.sociavault.com/v1/search/reddit",
    ]

    def _as_dict_list(value: object) -> list[dict]:
        """Convert list/dict payload fragments into a list of dict records."""

        def _is_index_map(dct: dict) -> bool:
            return bool(dct) and all(str(k).isdigit() for k in dct.keys())

        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
        if isinstance(value, dict):
            for collection_key in ("posts", "comments", "results", "items"):
                if collection_key in value:
                    extracted = _as_dict_list(value.get(collection_key))
                    if extracted:
                        return extracted
            nested = value.get("items") or value.get("data") or value.get("results")
            if isinstance(nested, list):
                return [x for x in nested if isinstance(x, dict)]
            if isinstance(nested, dict):
                if _is_index_map(nested) and all(isinstance(v, dict) for v in nested.values()):
                    return list(nested.values())
                for collection_key in ("posts", "comments", "results", "items"):
                    if collection_key in nested:
                        extracted = _as_dict_list(nested.get(collection_key))
                        if extracted:
                            return extracted
            if _is_index_map(value) and all(isinstance(v, dict) for v in value.values()):
                return list(value.values())
        return []

    def _extract_list(payload: object, keys: list[str]) -> list[dict]:
        """Extract list payloads from common response envelopes."""
        extracted = _as_dict_list(payload)
        if extracted:
            return extracted
        if not isinstance(payload, dict):
            return []

        for key in keys:
            value = payload.get(key)
            extracted = _as_dict_list(value)
            if extracted:
                return extracted
        return []

    def _request_json(
        session: requests.Session,
        endpoint_candidates: list[str],
        params: dict,
        timeout: int = 25,
    ) -> object:
        """Try GET and POST across endpoint candidates and return first JSON payload."""
        errors: list[str] = []
        for endpoint in endpoint_candidates:
            for method in ("GET", "POST"):
                try:
                    if method == "GET":
                        resp = session.get(endpoint, params=params, timeout=timeout)
                    else:
                        resp = session.post(endpoint, json=params, timeout=timeout)
                    if resp.status_code >= 400:
                        snippet = resp.text[:180].replace("\n", " ")
                        errors.append(f"{method} {endpoint} -> {resp.status_code}: {snippet}")
                        continue
                    return resp.json()
                except Exception as exc:
                    errors.append(f"{method} {endpoint} -> {exc}")
        raise RuntimeError(" | ".join(errors[:6]))

    session = requests.Session()
    session.headers.update({"x-api-key": api_key, "X-API-Key": api_key, **HEADERS})
    rows: list[dict] = []
    seen_ids: set[str] = set()

    def _append_post_row(post: dict, sub_name: str) -> None:
        """Fallback when comments are unavailable: keep post text so data collection still succeeds."""
        pid = str(post.get("id") or "")
        if not pid:
            return
        rid = f"post_{pid}"
        if rid in seen_ids:
            return

        text = _safe_text(str(post.get("selftext") or post.get("title") or ""))
        if len(text.split()) < MIN_WORD_COUNT:
            return

        seen_ids.add(rid)
        rows.append(
            {
                "id": rid,
                "text": text,
                "subreddit": sub_name,
                "post_title": _safe_text(str(post.get("title", ""))),
                "upvotes": int(post.get("score", 0) or 0),
                "created_utc": str(post.get("created_utc", ""))
                or datetime.now(tz=timezone.utc).isoformat(),
                "source": "reddit",
            }
        )

    for sub_name in tqdm(subreddits, desc="SociaVault Reddit"):
        sub_candidates = [sub_name, sub_name.lower()]
        posts: list[dict] = []
        last_exc: Exception | None = None
        # Try multiple query shapes because the endpoint may be strict.
        param_candidates = [
            {"subreddit": sub_candidates[0], "limit": max(1, posts_per_subreddit)},
            {"subreddit": sub_candidates[1], "limit": max(1, posts_per_subreddit)},
            {"name": sub_candidates[0], "limit": max(1, posts_per_subreddit), "sort": "new"},
            {"name": sub_candidates[1], "limit": max(1, posts_per_subreddit), "sort": "new"},
            {"name": sub_candidates[0], "limit": max(1, posts_per_subreddit)},
            {"name": sub_candidates[1], "limit": max(1, posts_per_subreddit)},
        ]
        try:
            endpoint_candidates = [
                f"{base_urls[0]}/subreddit",
                f"{base_urls[0]}/posts",
                f"{base_urls[0]}/search/posts",
                f"{base_urls[1]}/subreddit",
                f"{base_urls[1]}/posts",
                f"{base_urls[2]}/posts",
            ]

            for params in param_candidates:
                try:
                    payload = _request_json(session, endpoint_candidates, params=params)
                    posts = _extract_list(
                        payload,
                        ["posts", "data", "results", "items", "comments"],
                    )
                    if posts:
                        break
                except Exception as exc:
                    last_exc = exc
            if not posts and last_exc is not None:
                raise last_exc
        except Exception as exc:
            logger.warning(
                f"SociaVault subreddit request failed for r/{sub_name}: {exc}"
            )
            continue

        for post in posts:
            if isinstance(post, dict):
                post_url = post.get("permalink") or post.get("url") or ""
                post_title = _safe_text(str(post.get("title", "")))
            elif isinstance(post, str):
                post_url = post
                post_title = ""
            else:
                continue
            if not post_url:
                continue
            if post_url.startswith("/"):
                post_url = f"https://www.reddit.com{post_url}"

            comments = []
            c_last_exc: Exception | None = None
            comment_param_candidates = [
                {"url": post_url, "limit": max(1, comments_per_post)},
                {"postUrl": post_url, "limit": max(1, comments_per_post)},
            ]
            try:
                comment_endpoint_candidates = [
                    f"{base_urls[0]}/post/comments",
                    f"{base_urls[0]}/comments",
                    f"{base_urls[1]}/post/comments",
                    f"{base_urls[1]}/comments",
                ]
                for c_params in comment_param_candidates:
                    try:
                        c_payload = _request_json(
                            session,
                            comment_endpoint_candidates,
                            params=c_params,
                        )
                        comments = _extract_list(
                            c_payload,
                            ["comments", "data", "results", "items"],
                        )
                        if comments:
                            break
                    except Exception as exc:
                        c_last_exc = exc
                if not comments and c_last_exc is not None:
                    raise c_last_exc
            except Exception as exc:
                logger.warning(f"SociaVault comments request failed for {post_url}: {exc}")
                _append_post_row(post, sub_name)
                continue

            if not comments:
                _append_post_row(post, sub_name)
                continue

            for comment in comments:
                cid = str(comment.get("id") or comment.get("comment_id") or "")
                if not cid or cid in seen_ids:
                    continue
                seen_ids.add(cid)

                body = _safe_text(str(comment.get("body", "")))
                if len(body.split()) < MIN_WORD_COUNT:
                    continue

                upvotes = int(comment.get("score", 0) or 0)
                created = str(comment.get("created_utc", "")) or datetime.now(
                    tz=timezone.utc
                ).isoformat()

                rows.append(
                    {
                        "id": cid,
                        "text": body,
                        "subreddit": sub_name,
                        "post_title": post_title,
                        "upvotes": upvotes,
                        "created_utc": created,
                        "source": "reddit",
                    }
                )

    return rows


def _scrape_reddit_html(
    subreddits: list[str],
    pages_per_subreddit: int = 5,
    sort_mode: str = "new",
    rate_limit_seconds: float = 0.8,
) -> list[dict]:
    """Scrape comment text from old.reddit.com using BeautifulSoup."""
    import re

    rows: list[dict] = []
    seen_ids: set[str] = set()
    session = requests.Session()
    session.headers.update(HEADERS)
    rotator = get_rotator()

    for sub_name in subreddits:
        after: str | None = None
        for _ in tqdm(range(pages_per_subreddit), desc=f"HTML r/{sub_name}", leave=False):
            url = f"https://old.reddit.com/r/{sub_name}/{sort_mode}/"
            params = {"limit": 100}
            if after:
                params["after"] = after

            try:
                resp = session_get(session, rotator, url, params=params, timeout=20)
                if resp.status_code in (429, 403):
                    logger.warning(
                        f"Blocked by Reddit HTML endpoint (status={resp.status_code}) for r/{sub_name}"
                    )
                    break
                resp.raise_for_status()
            except Exception as exc:
                logger.warning(f"Reddit HTML request failed for r/{sub_name}: {exc}")
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            comments = soup.select("div.comment")
            for comment in comments:
                cid = comment.get("data-fullname") or comment.get("id") or ""
                if not cid or cid in seen_ids:
                    continue
                seen_ids.add(cid)

                body_el = comment.select_one(".md")
                if body_el is None:
                    continue
                body = _safe_text(body_el.get_text(" ", strip=True))
                if len(body.split()) < MIN_WORD_COUNT:
                    continue

                score = 0
                score_el = comment.select_one(".score.unvoted")
                if score_el:
                    nums = re.findall(r"-?\d+", score_el.get_text())
                    score = int(nums[0]) if nums else 0

                post_title = ""
                title_el = soup.select_one("a.title")
                if title_el:
                    post_title = _safe_text(title_el.get_text())

                rows.append(
                    {
                        "id": cid,
                        "text": body,
                        "subreddit": sub_name,
                        "post_title": post_title,
                        "upvotes": score,
                        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
                        "source": "reddit",
                    }
                )

            next_btn = soup.select_one("span.next-button > a")
            after = None
            if next_btn:
                href = next_btn.get("href", "")
                match = re.search(r"[?&]after=([^&]+)", href)
                after = match.group(1) if match else None
            if not after:
                break

            time.sleep(rate_limit_seconds)

    return rows


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

    # Try SociaVault API first (if configured).
    rows = _scrape_reddit_sociavault(
        subreddits=subreddits,
        posts_per_subreddit=max(25, min(posts_per_sort, 200)),
        comments_per_post=100,
    )
    if rows:
        df = pd.DataFrame(rows)
        logger.info(
            f"Scraped {len(df):,} comments via SociaVault across {len(subreddits)} subreddits"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved → {output_path}")
        return df

    # Then try BeautifulSoup HTML scraping.
    rows = _scrape_reddit_html(
        subreddits=subreddits,
        pages_per_subreddit=max(1, min(10, posts_per_sort // 100 + 1)),
        sort_mode="new",
        rate_limit_seconds=rate_limit_seconds,
    )

    # Fall back to PRAW if HTML path returned nothing and credentials are set.
    if rows:
        df = pd.DataFrame(rows)
        logger.info(
            f"Scraped {len(df):,} comments via BeautifulSoup across {len(subreddits)} subreddits"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved → {output_path}")
        return df

    if not (settings.reddit_client_id and settings.reddit_client_secret):
        logger.warning(
            "BeautifulSoup scrape returned no rows and Reddit credentials are missing. "
            "Add REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET for PRAW fallback."
        )
        df = pd.DataFrame(rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return df

    reddit = _get_reddit_client()
    rows = []
    seen_ids: set[str] = set()

    try:
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
    except Exception as exc:
        logger.warning(f"PRAW fallback failed: {exc}")

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
    parser.add_argument(
        "--stream-zst",
        nargs="+",
        default=None,
        help="One or more .zst dump files (RC_*/RS_*) to stream",
    )
    parser.add_argument(
        "--stream-glob",
        type=str,
        default=None,
        help="Glob for .zst files, e.g. 'data/raw/reddit/RS_2024-*.zst'",
    )
    parser.add_argument(
        "--subreddits",
        nargs="+",
        default=None,
        help="Subreddit allow-list (default: ValoTox subreddits)",
    )
    parser.add_argument("--keyword-only", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--comments-only", action="store_true")
    parser.add_argument("--submissions-only", action="store_true")
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="After streaming, run keyword-stratified sample",
    )
    args = parser.parse_args()

    # Stream mode: for large Pushshift/Academic Torrents .zst dumps.
    if args.stream_zst or args.stream_glob:
        include_comments = not args.submissions_only
        include_submissions = not args.comments_only
        out_csv = stream_pushshift_dumps(
            input_paths=args.stream_zst,
            input_glob=args.stream_glob,
            subreddits=args.subreddits,
            output_path=args.output,
            keyword_only=args.keyword_only,
            include_comments=include_comments,
            include_submissions=include_submissions,
            max_rows=args.max_rows,
        )
        if args.stratify:
            streamed_df = pd.read_csv(out_csv)
            if streamed_df.empty or "text" not in streamed_df.columns:
                logger.warning(
                    "Stream output is empty or malformed; skipping stratified sample generation."
                )
            else:
                stratified_sample(streamed_df, toxic_n=args.toxic_n, clean_n=args.clean_n)
    else:
        raw_df = scrape_subreddits(posts_per_sort=args.posts, output_path=args.output)
        if raw_df.empty or "text" not in raw_df.columns:
            logger.warning("No Reddit rows collected; skipping stratified sample generation.")
        else:
            stratified_sample(raw_df, toxic_n=args.toxic_n, clean_n=args.clean_n)
