"""
Data pipeline for Reddit-only ValoTox data: clean, deduplicate, and
prepare annotation splits.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
from loguru import logger

from valotox.config import PROCESSED_DIR, RAW_DIR
from valotox.lexicon import ALL_TOXIC_KEYWORDS, get_regex_pattern


# ── Text cleaning ────────────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")
_MULTI_SPACE = re.compile(r"\s+")
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
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
    combined = combined[
        (combined["word_count"] >= min_words) & (combined["word_count"] <= max_words)
    ].copy()
    logger.info(f"After word-count filter: {len(combined):,}")

    # ── Deduplicate ──────────────────────────────────────────────────────
    combined["text_hash"] = combined["text_clean"].apply(text_hash)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["text_hash"]).reset_index(drop=True)
    logger.info(f"Deduplicated: {before:,} → {len(combined):,}")

    # ── Keyword flag ─────────────────────────────────────────────────────
    pattern = get_regex_pattern()
    combined["has_toxic_keyword"] = combined["text_clean"].str.contains(
        pattern, case=False, na=False, regex=True
    )
    n_kw = combined["has_toxic_keyword"].sum()
    logger.info(f"Keyword-matching rows: {n_kw:,} / {len(combined):,}")

    # ── Add empty label columns ──────────────────────────────────────────
    from valotox.lexicon import LABELS
    for label in LABELS:
        combined[label] = 0  # will be filled during annotation

    # ── Save ─────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    logger.info(f"Saved merged dataset → {output_path}")
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
