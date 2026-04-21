"""
Data pipeline for Reddit-only ValoTox data: clean, deduplicate, and
prepare annotation splits.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from loguru import logger

from valotox.config import PROCESSED_DIR, RAW_DIR
from valotox.lexicon import LABELS, get_regex_pattern

# ── Text cleaning ────────────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")
_MULTI_SPACE = re.compile(r"\s+")
_EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
    "]+",
    flags=re.UNICODE,
)


def clean_text(text: str, remove_emojis: bool = False) -> str:
    """Normalise a single text string for NLP processing."""
    if not isinstance(text, str):
        return ""
    text = _URL_RE.sub("", text)
    text = _MENTION_RE.sub("", text)
    if remove_emojis:
        text = _EMOJI_RE.sub("", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


def text_hash(text: str) -> str:
    """SHA-256 fingerprint for deduplication."""
    return hashlib.sha256(text.lower().encode("utf-8")).hexdigest()


# ── Merge & deduplicate ─────────────────────────────────────────────────────


def merge_sources(
    raw_dir: Path | str | None = None,
    output_path: Path | str | None = None,
    min_words: int = 4,
    max_words: int = 512,
) -> pd.DataFrame:
    """Load Reddit raw CSV, clean, deduplicate, and produce a single dataset.

    Parameters
    ----------
    raw_dir : Path | str | None
        Directory containing ``reddit_raw.csv``. Default: ``data/raw/``.
    output_path : Path | str | None
        Where to write the merged CSV. Default: ``data/processed/valotox_merged.csv``.
    min_words, max_words : int
        Word-count filter bounds.

    Returns
    -------
    pd.DataFrame
    """
    raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
    output_path = Path(output_path) if output_path else PROCESSED_DIR / "valotox_merged.csv"

    reddit_csv = raw_dir / "reddit_raw.csv"
    if not reddit_csv.exists():
        logger.error(f"Missing reddit source file: {reddit_csv}")
        return pd.DataFrame()

    logger.info(f"Loading {reddit_csv.name} …")
    df = pd.read_csv(reddit_csv, dtype=str).fillna("")
    if "text" not in df.columns:
        logger.error(f"{reddit_csv.name} has no 'text' column")
        return pd.DataFrame()

    combined = df.copy()
    logger.info(f"Combined raw rows: {len(combined):,}")

    # ── Clean ────────────────────────────────────────────────────────────
    combined["text_clean"] = combined["text"].apply(clean_text)
    combined["word_count"] = combined["text_clean"].str.split().str.len()
    combined = combined[(combined["word_count"] >= min_words) & (combined["word_count"] <= max_words)].copy()
    logger.info(f"After word-count filter: {len(combined):,}")

    # ── Deduplicate ──────────────────────────────────────────────────────
    combined["text_hash"] = combined["text_clean"].apply(text_hash)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["text_hash"]).reset_index(drop=True)
    logger.info(f"Deduplicated: {before:,} → {len(combined):,}")

    # ── Keyword flag ─────────────────────────────────────────────────────
    pattern = get_regex_pattern()
    combined["has_toxic_keyword"] = combined["text_clean"].str.contains(pattern, case=False, na=False, regex=True)
    n_kw = combined["has_toxic_keyword"].sum()
    logger.info(f"Keyword-matching rows: {n_kw:,} / {len(combined):,}")

    # ── Add empty label columns ──────────────────────────────────────────
    for label in LABELS:
        combined[label] = 0  # will be filled during annotation

    # ── Save ─────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    logger.info(f"Saved merged dataset → {output_path}")
    return combined


def merge_reddit_and_conda(
    conda_csv: Path | str,
    reddit_csv: Path | str | None = None,
    output_path: Path | str | None = None,
    min_words_reddit: int = 4,
    min_words_conda: int = 1,
    dedupe_across_sources: bool = True,
) -> pd.DataFrame:
    """Combine processed Reddit rows with CONDA in-game chat CSV.

    Reddit defaults to ``data/processed/valotox_merged.csv`` if present; otherwise
    ``data/raw/all_sources_raw_normalized.jsonl``.

    CONDA CSV must include an ``utterance`` column (and typically ``Id``, ``matchId``).

    Output columns include ``text``, ``text_clean``, ``source_origin`` (``reddit`` | ``conda``),
    ``weak_toxic`` (keyword flag), and empty multi-label columns for annotation workflows.
    """
    conda_csv = Path(conda_csv)
    if not conda_csv.is_file():
        raise FileNotFoundError(conda_csv)

    output_path = Path(output_path) if output_path else PROCESSED_DIR / "reddit_conda_merged.csv"

    # ── Reddit ─────────────────────────────────────────────────────────────
    reddit_csv = Path(reddit_csv) if reddit_csv else PROCESSED_DIR / "valotox_merged.csv"
    jsonl_fallback = RAW_DIR / "all_sources_raw_normalized.jsonl"

    if reddit_csv.is_file():
        logger.info(f"Loading Reddit from {reddit_csv}")
        rdf = pd.read_csv(reddit_csv, dtype=str, low_memory=False).fillna("")
    elif jsonl_fallback.is_file():
        logger.info(f"Loading Reddit from {jsonl_fallback}")
        rows: list[dict] = []
        with jsonl_fallback.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        rdf = pd.DataFrame(rows)
    else:
        raise FileNotFoundError(f"No Reddit source found. Expected {reddit_csv} or {jsonl_fallback}")

    if "text" not in rdf.columns:
        raise ValueError("Reddit data must contain a 'text' column")

    rdf = rdf.copy()
    rdf["source_origin"] = "reddit"
    if "uid" not in rdf.columns:
        rdf["uid"] = "reddit_row:" + rdf.index.astype(str)

    rdf["text_clean"] = rdf["text"].astype(str).apply(clean_text)
    rdf["word_count"] = rdf["text_clean"].str.split().str.len()
    rdf = rdf[(rdf["word_count"] >= min_words_reddit)].copy()
    logger.info(f"Reddit rows after ≥{min_words_reddit} words: {len(rdf):,}")

    # ── CONDA ──────────────────────────────────────────────────────────────
    logger.info(f"Loading CONDA from {conda_csv}")
    cdf = pd.read_csv(conda_csv, dtype=str, low_memory=False).fillna("")
    if "utterance" not in cdf.columns:
        raise ValueError("CONDA CSV must contain an 'utterance' column")

    cdf = cdf.copy()
    cdf["source_origin"] = "conda"
    cdf["text"] = cdf["utterance"].astype(str)
    id_col = "Id" if "Id" in cdf.columns else cdf.columns[0]
    cdf["uid"] = cdf[id_col].apply(lambda x: f"conda:{x}" if str(x).strip() else "")
    mask_empty_uid = cdf["uid"] == ""
    if mask_empty_uid.any():
        cdf.loc[mask_empty_uid, "uid"] = "conda:idx_" + cdf.loc[mask_empty_uid].index.astype(str)

    cdf["subreddit"] = ""
    cdf["source_type"] = "in_game_chat"
    cdf["text_clean"] = cdf["text"].apply(clean_text)
    cdf["word_count"] = cdf["text_clean"].str.split().str.len()
    cdf = cdf[(cdf["word_count"] >= min_words_conda)].copy()
    logger.info(f"CONDA rows after ≥{min_words_conda} words: {len(cdf):,}")

    for col in ("conda_match_id", "conda_conversation_id"):
        if col not in rdf.columns:
            rdf[col] = ""
    if "matchId" in cdf.columns:
        cdf["conda_match_id"] = cdf["matchId"].astype(str)
    else:
        cdf["conda_match_id"] = ""
    if "conversationId" in cdf.columns:
        cdf["conda_conversation_id"] = cdf["conversationId"].astype(str)
    else:
        cdf["conda_conversation_id"] = ""

    out_cols = [
        "uid",
        "text",
        "text_clean",
        "source_origin",
        "word_count",
        "subreddit",
        "source_type",
        "conda_match_id",
        "conda_conversation_id",
        "created_utc",
    ]
    if "subreddit" not in rdf.columns:
        rdf["subreddit"] = ""
    if "source_type" not in rdf.columns:
        rdf["source_type"] = ""
    if "created_utc" not in rdf.columns:
        rdf["created_utc"] = ""
    if "created_utc" not in cdf.columns:
        cdf["created_utc"] = ""

    r_small = rdf[out_cols].copy()
    c_small = cdf[out_cols].copy()

    combined = pd.concat([r_small, c_small], ignore_index=True)
    pattern = get_regex_pattern()
    combined["weak_toxic"] = combined["text_clean"].str.contains(pattern, case=False, na=False, regex=True).astype(int)
    for label in LABELS:
        if label not in combined.columns:
            combined[label] = 0

    if dedupe_across_sources:
        combined["text_hash"] = combined["text_clean"].apply(text_hash)
        before = len(combined)
        combined = combined.drop_duplicates(subset=["text_hash"], keep="first").reset_index(drop=True)
        combined = combined.drop(columns=["text_hash"])
        logger.info(f"Deduped by text: {before:,} → {len(combined):,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    logger.info(f"Saved Reddit+CONDA merged dataset → {output_path} ({len(combined):,} rows)")
    return combined


# ── Stratified sampling for annotation ───────────────────────────────────────


def create_annotation_splits(
    df: pd.DataFrame | None = None,
    input_path: Path | str | None = None,
    iaa_size: int = 600,
    total_annotation_size: int = 10_000,
    seed: int = 42,
    output_dir: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    """Create IAA overlap split + main annotation split.

    Parameters
    ----------
    df : pd.DataFrame | None
        Merged dataset. If ``None``, reads from ``input_path``.
    input_path : Path | str | None
        Merged CSV path (used when ``df`` is ``None``).
    iaa_size : int
        Number of samples in the IAA (inter-annotator agreement) batch.
    total_annotation_size : int
        Total samples to annotate (including IAA batch).
    seed : int
        Random state.
    output_dir : Path | str | None
        Directory for output CSVs. Default: ``data/processed/``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: ``"iaa"``, ``"batch_1"``, ``"batch_2"``, ``"batch_3"``.
    """
    output_dir = Path(output_dir) if output_dir else PROCESSED_DIR

    if df is None:
        input_path = Path(input_path) if input_path else PROCESSED_DIR / "valotox_merged.csv"
        df = pd.read_csv(input_path)

    # Ensure balanced keyword representation
    pattern = get_regex_pattern()
    mask = df["text_clean"].str.contains(pattern, case=False, na=False, regex=True)
    toxic_pool = df[mask]
    clean_pool = df[~mask]

    # IAA batch: 50/50 toxic-keyword / clean
    iaa_toxic = toxic_pool.sample(n=min(iaa_size // 2, len(toxic_pool)), random_state=seed)
    iaa_clean = clean_pool.sample(n=min(iaa_size // 2, len(clean_pool)), random_state=seed)
    iaa_df = pd.concat([iaa_toxic, iaa_clean]).sample(frac=1, random_state=seed).reset_index(drop=True)

    # Remaining for annotation
    remaining = df[~df.index.isin(iaa_df.index)]
    remaining_annotation_size = total_annotation_size - len(iaa_df)

    if remaining_annotation_size > len(remaining):
        remaining_annotation_size = len(remaining)

    annotation_pool = remaining.sample(n=remaining_annotation_size, random_state=seed).reset_index(drop=True)

    # Split into 3 batches for distribution across annotators
    batch_size = len(annotation_pool) // 3
    batch_1 = annotation_pool.iloc[:batch_size]
    batch_2 = annotation_pool.iloc[batch_size : 2 * batch_size]
    batch_3 = annotation_pool.iloc[2 * batch_size :]

    splits = {
        "iaa": iaa_df,
        "batch_1": batch_1,
        "batch_2": batch_2,
        "batch_3": batch_3,
    }

    for name, split_df in splits.items():
        fp = output_dir / f"annotation_{name}.csv"
        split_df.to_csv(fp, index=False)
        logger.info(f"  {name}: {len(split_df):,} rows → {fp}")

    return splits


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge and prepare ValoTox data")
    parser.add_argument("--merge", action="store_true", help="Run merge pipeline")
    parser.add_argument("--split", action="store_true", help="Create annotation splits")
    parser.add_argument("--iaa-size", type=int, default=600)
    parser.add_argument("--total-size", type=int, default=10000)
    args = parser.parse_args()

    if args.merge:
        merged = merge_sources()

    if args.split:
        create_annotation_splits(
            iaa_size=args.iaa_size,
            total_annotation_size=args.total_size,
        )
