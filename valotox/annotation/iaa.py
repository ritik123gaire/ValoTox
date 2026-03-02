"""
Inter-Annotator Agreement (IAA) computation for ValoTox.

Computes Cohen's Kappa (pair-wise) and Fleiss' Kappa (multi-annotator)
per label, with special attention to passive_toxic where lower agreement
is expected and documented as a research finding.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import cohen_kappa_score

from valotox.config import ANNOTATED_DIR
from valotox.lexicon import LABELS


def _load_iaa_annotations(
    iaa_dir: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load per-annotator IAA CSVs.

    Expects files named ``iaa_annotator_<name>.csv`` each with columns:
    ``text``, ``toxic``, ``harassment``, ``gender_attack``, ``slur``,
    ``passive_toxic``, ``not_toxic``.

    Returns a dict {annotator_name: DataFrame} sorted by text for alignment.
    """
    iaa_dir = Path(iaa_dir) if iaa_dir else ANNOTATED_DIR
    files = sorted(iaa_dir.glob("iaa_annotator_*.csv"))

    if len(files) < 2:
        logger.error(f"Need ≥ 2 IAA files in {iaa_dir}, found {len(files)}")
        return {}

    annotators: dict[str, pd.DataFrame] = {}
    for fp in files:
        name = fp.stem.replace("iaa_annotator_", "")
        df = pd.read_csv(fp).sort_values("text").reset_index(drop=True)
        annotators[name] = df
        logger.info(f"  Loaded {name}: {len(df)} rows")

    return annotators


def cohens_kappa_pairwise(
    annotators: dict[str, pd.DataFrame] | None = None,
    iaa_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Compute Cohen's Kappa for every annotator pair × every label.

    Returns
    -------
    pd.DataFrame
        Columns: ``annotator_1``, ``annotator_2``, ``label``, ``kappa``.
    """
    if annotators is None:
        annotators = _load_iaa_annotations(iaa_dir)

    names = sorted(annotators.keys())
    records: list[dict] = []

    for a1, a2 in itertools.combinations(names, 2):
        df1, df2 = annotators[a1], annotators[a2]
        assert len(df1) == len(df2), f"Length mismatch: {a1}={len(df1)}, {a2}={len(df2)}"

        for label in LABELS:
            kappa = cohen_kappa_score(df1[label].values, df2[label].values)
            records.append(
                {
                    "annotator_1": a1,
                    "annotator_2": a2,
                    "label": label,
                    "kappa": round(kappa, 4),
                }
            )
            logger.info(f"  κ({a1}, {a2}) [{label}] = {kappa:.4f}")

    result = pd.DataFrame(records)
    return result


def fleiss_kappa(
    annotators: dict[str, pd.DataFrame] | None = None,
    iaa_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Compute Fleiss' Kappa per label across all annotators.

    Returns
    -------
    pd.DataFrame
        Columns: ``label``, ``fleiss_kappa``.
    """
    if annotators is None:
        annotators = _load_iaa_annotations(iaa_dir)

    names = sorted(annotators.keys())
    n_annotators = len(names)
    n_items = len(next(iter(annotators.values())))

    records: list[dict] = []

    for label in LABELS:
        # Build n_items × 2 matrix: [count_0, count_1]
        counts = np.zeros((n_items, 2), dtype=float)
        for name in names:
            vals = annotators[name][label].values.astype(int)
            for i, v in enumerate(vals):
                counts[i, v] += 1

        # Fleiss' kappa computation
        N = n_items
        n = n_annotators
        k = 2  # binary

        p_j = counts.sum(axis=0) / (N * n)  # proportion in each category
        P_i = (np.sum(counts ** 2, axis=1) - n) / (n * (n - 1))  # per-item agreement
        P_bar = P_i.mean()
        P_e = np.sum(p_j ** 2)

        if P_e == 1.0:
            kappa = 1.0
        else:
            kappa = (P_bar - P_e) / (1 - P_e)

        records.append({"label": label, "fleiss_kappa": round(kappa, 4)})
        logger.info(f"  Fleiss κ [{label}] = {kappa:.4f}")

    return pd.DataFrame(records)


def iaa_report(
    iaa_dir: Path | str | None = None,
    output_path: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    """Full IAA report: pairwise Cohen's κ + Fleiss' κ, saved to CSV.

    Returns dict with ``"cohens"`` and ``"fleiss"`` DataFrames.
    """
    output_dir = Path(output_path) if output_path else ANNOTATED_DIR
    annotators = _load_iaa_annotations(iaa_dir)

    if not annotators:
        return {}

    logger.info("── Cohen's Kappa (pairwise) ──")
    cohens = cohens_kappa_pairwise(annotators)

    logger.info("── Fleiss' Kappa (all annotators) ──")
    fleiss = fleiss_kappa(annotators)

    cohens.to_csv(output_dir / "iaa_cohens_kappa.csv", index=False)
    fleiss.to_csv(output_dir / "iaa_fleiss_kappa.csv", index=False)

    # Flag any labels below threshold
    low_kappa = fleiss[fleiss["fleiss_kappa"] < 0.5]
    if not low_kappa.empty:
        logger.warning(
            f"Labels with Fleiss κ < 0.50: {low_kappa['label'].tolist()} "
            "— this is expected for 'passive_toxic', document in the paper."
        )

    return {"cohens": cohens, "fleiss": fleiss}


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    iaa_report()
