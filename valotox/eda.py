"""
Exploratory Data Analysis (EDA) & dataset statistics for ValoTox.

Produces all key analyses described in Phase 4:
- Label distribution per subreddit
- Average comment length by toxicity type
- Most discriminative tokens (TF-IDF)
- IAA scores visualisation
- Temporal toxicity trends
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer

from valotox.config import ANNOTATED_DIR, PROCESSED_DIR
from valotox.lexicon import LABELS

# ── Output directory ─────────────────────────────────────────────────────────
FIGURES_DIR = PROCESSED_DIR.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ── 1. Label distribution per subreddit ──────────────────────────────────────

def plot_label_distribution(
    df: pd.DataFrame,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Bar chart of label counts, optionally faceted by subreddit."""
    save_path = Path(save_path) if save_path else FIGURES_DIR / "label_distribution.png"

    label_cols = [l for l in LABELS if l in df.columns]
    counts = df[label_cols].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", ax=ax, color=sns.color_palette("Set2", len(counts)))
    ax.set_title("Label Distribution in ValoTox Dataset")
    ax.set_ylabel("Count")
    ax.set_xlabel("Label")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    logger.info(f"Saved label distribution → {save_path}")
    return fig


def plot_label_by_subreddit(
    df: pd.DataFrame,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Stacked bar chart: label prevalence per subreddit."""
    save_path = Path(save_path) if save_path else FIGURES_DIR / "label_by_subreddit.png"

    if "subreddit" not in df.columns:
        logger.warning("No 'subreddit' column — skipping subreddit breakdown")
        return plt.figure()

    label_cols = [l for l in LABELS if l in df.columns]
    grouped = df.groupby("subreddit")[label_cols].mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    grouped.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
    ax.set_title("Label Prevalence by Subreddit (proportion)")
    ax.set_ylabel("Proportion")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    logger.info(f"Saved subreddit breakdown → {save_path}")
    return fig


# ── 2. Comment length by toxicity type ──────────────────────────────────────

def plot_length_by_label(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Box plot of word count per active label."""
    save_path = Path(save_path) if save_path else FIGURES_DIR / "length_by_label.png"

    df = df.copy()
    df["word_count"] = df[text_col].astype(str).str.split().str.len()

    label_cols = [l for l in LABELS if l in df.columns]
    melted: list[pd.DataFrame] = []
    for label in label_cols:
        subset = df[df[label] == 1][["word_count"]].copy()
        subset["label"] = label
        melted.append(subset)

    if not melted:
        return plt.figure()

    plot_df = pd.concat(melted)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=plot_df, x="label", y="word_count", ax=ax, palette="Set2")
    ax.set_title("Comment Length (words) by Toxicity Label")
    ax.set_ylabel("Word Count")
    ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    logger.info(f"Saved length-by-label → {save_path}")
    return fig


# ── 3. Most discriminative tokens (TF-IDF) ──────────────────────────────────

def tfidf_top_tokens(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    top_k: int = 20,
    save_path: Path | str | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """Compute top-K TF-IDF tokens per label.

    Returns a dict mapping label → [(token, score), ...].
    """
    save_path = Path(save_path) if save_path else FIGURES_DIR / "tfidf_top_tokens.png"

    label_cols = [l for l in LABELS if l in df.columns and l != "not_toxic"]
    result: dict[str, list[tuple[str, float]]] = {}

    vectorizer = TfidfVectorizer(
        max_features=10_000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
    )

    all_texts = df[text_col].fillna("").astype(str).tolist()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    feature_names = vectorizer.get_feature_names_out()

    for label in label_cols:
        mask = df[label] == 1
        if mask.sum() == 0:
            continue
        mean_tfidf = tfidf_matrix[mask.values].mean(axis=0).A1
        top_indices = mean_tfidf.argsort()[::-1][:top_k]
        top_tokens = [(feature_names[i], round(mean_tfidf[i], 4)) for i in top_indices]
        result[label] = top_tokens

    # Plot heatmap of top tokens per label
    if result:
        all_tokens = set()
        for tokens in result.values():
            all_tokens.update(t[0] for t in tokens[:10])

        token_list = sorted(all_tokens)
        heatmap_data = pd.DataFrame(0.0, index=list(result.keys()), columns=token_list)
        for label, tokens in result.items():
            for tok, score in tokens:
                if tok in token_list:
                    heatmap_data.loc[label, tok] = score

        fig, ax = plt.subplots(figsize=(16, 6))
        sns.heatmap(heatmap_data, cmap="YlOrRd", ax=ax, linewidths=0.5)
        ax.set_title("Top TF-IDF Tokens per Toxicity Label")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(save_path, dpi=200)
        logger.info(f"Saved TF-IDF heatmap → {save_path}")

    return result


# ── 4. IAA visualisation ─────────────────────────────────────────────────────

def plot_iaa_scores(
    cohens_path: Path | str | None = None,
    fleiss_path: Path | str | None = None,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Bar chart of IAA kappa scores per label."""
    save_path = Path(save_path) if save_path else FIGURES_DIR / "iaa_scores.png"
    fleiss_path = Path(fleiss_path) if fleiss_path else ANNOTATED_DIR / "iaa_fleiss_kappa.csv"

    if not fleiss_path.exists():
        logger.warning(f"Fleiss kappa file not found: {fleiss_path}")
        return plt.figure()

    fleiss = pd.read_csv(fleiss_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["green" if k >= 0.65 else "orange" if k >= 0.50 else "red" for k in fleiss["fleiss_kappa"]]
    ax.barh(fleiss["label"], fleiss["fleiss_kappa"], color=colors)
    ax.axvline(x=0.65, color="black", linestyle="--", alpha=0.5, label="κ = 0.65 threshold")
    ax.axvline(x=0.50, color="gray", linestyle=":", alpha=0.5, label="κ = 0.50 minimum")
    ax.set_xlabel("Fleiss' Kappa")
    ax.set_title("Inter-Annotator Agreement per Label")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    logger.info(f"Saved IAA chart → {save_path}")
    return fig


# ── 5. Temporal analysis ─────────────────────────────────────────────────────

def plot_temporal_trends(
    df: pd.DataFrame,
    date_col: str = "created_utc",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Line chart of toxicity over time (weekly rolling average)."""
    save_path = Path(save_path) if save_path else FIGURES_DIR / "temporal_trends.png"

    if date_col not in df.columns:
        logger.warning(f"No '{date_col}' column — skipping temporal analysis")
        return plt.figure()

    df = df.copy()
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"])

    if df.empty:
        return plt.figure()

    label_cols = [l for l in LABELS if l in df.columns and l != "not_toxic"]
    df = df.set_index("date").sort_index()

    fig, ax = plt.subplots(figsize=(14, 6))
    for label in label_cols:
        weekly = df[label].resample("W").mean()
        ax.plot(weekly.index, weekly.values, label=label, alpha=0.8)

    ax.set_title("Toxicity Trends Over Time (Weekly Average)")
    ax.set_ylabel("Proportion of Comments with Label")
    ax.set_xlabel("Date")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    logger.info(f"Saved temporal trends → {save_path}")
    return fig


# ── 6. Dataset summary statistics ────────────────────────────────────────────

def dataset_summary(df: pd.DataFrame, text_col: str = "text_clean") -> pd.DataFrame:
    """Return a summary table of key dataset statistics."""
    stats: dict[str, object] = {
        "total_samples": len(df),
        "unique_texts": df[text_col].nunique() if text_col in df.columns else None,
    }

    if "source" in df.columns:
        stats["sources"] = df["source"].value_counts().to_dict()
    if "subreddit" in df.columns:
        stats["subreddits"] = df["subreddit"].value_counts().to_dict()

    label_cols = [l for l in LABELS if l in df.columns]
    for label in label_cols:
        stats[f"{label}_count"] = int(df[label].sum())
        stats[f"{label}_pct"] = round(df[label].mean() * 100, 2)

    if text_col in df.columns:
        wc = df[text_col].astype(str).str.split().str.len()
        stats["avg_word_count"] = round(wc.mean(), 1)
        stats["median_word_count"] = round(wc.median(), 1)

    summary_df = pd.DataFrame([stats])
    summary_path = FIGURES_DIR / "dataset_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Dataset summary saved → {summary_path}")
    return summary_df


# ── Full EDA pipeline ────────────────────────────────────────────────────────

def run_full_eda(
    input_path: Path | str | None = None,
) -> None:
    """Run all EDA analyses and save figures + stats."""
    input_path = Path(input_path) if input_path else ANNOTATED_DIR / "valotox_annotated.csv"

    if not input_path.exists():
        logger.error(f"Annotated dataset not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df):,} annotated samples for EDA")

    dataset_summary(df)
    plot_label_distribution(df)
    plot_label_by_subreddit(df)
    plot_length_by_label(df)
    tfidf_top_tokens(df)
    plot_iaa_scores()
    plot_temporal_trends(df)

    logger.info(f"All EDA figures saved to {FIGURES_DIR}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ValoTox EDA")
    parser.add_argument("--input", type=str, default=None)
    args = parser.parse_args()

    run_full_eda(args.input)
