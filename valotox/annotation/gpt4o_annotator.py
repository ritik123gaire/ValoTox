"""
GPT-4o synthetic annotator for ValoTox.

Uses OpenAI's GPT-4o as a 4th "annotator" on the IAA batch to:
1. Compare human vs. LLM agreement (standalone research finding)
2. Provide a tiebreaker when human annotators disagree
3. Pre-label large batches for human review (active-learning style)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from valotox.config import ANNOTATED_DIR, PROCESSED_DIR, settings
from valotox.lexicon import LABELS

# ── System prompt for GPT-4o ─────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert content moderator specialising in online gaming
toxicity, specifically for the game VALORANT by Riot Games.

You will be given a single text comment from a Valorant community.  Your job is to
classify it using the following multi-label schema.  A comment can have MULTIPLE labels.

Labels:
- toxic: General insults, aggression, or hostility (e.g., "you're so braindead uninstall")
- harassment: Targeted at a specific player with intent to harm (e.g., "reported you get rekt")
- gender_attack: Misogyny or gender-based abuse (e.g., "go back to the kitchen")
- slur: Contains hate-speech slurs or derogatory language based on identity
- passive_toxic: Subtle toxicity, sarcasm, backhanded compliments, or demeaning
    language that hides behind humour or game terminology (e.g., "ggez wp diff", "nice try lol")
- not_toxic: The comment is clean, friendly, or purely game-related

IMPORTANT context for Valorant-specific terms:
- "diff" = saying the opposing player was better → often used mockingly → passive_toxic
- "ez" / "ggez" = "easy" → disrespectful sportsmanship → passive_toxic
- "bot" = calling someone an AI bot (bad player) → toxic
- "throwing" / "griefing" = accusation of intentional losing → can be toxic or factual
- "kys" = "kill yourself" → always toxic + harassment
- Terms like "NHK", "TMKC", "bsdk" = regional slurs → slur
- Sarcastic "nice try" or "unlucky" after a loss → passive_toxic

Respond ONLY with a JSON object with these exact keys, values must be 0 or 1:
{"toxic": 0, "harassment": 0, "gender_attack": 0, "slur": 0, "passive_toxic": 0, "not_toxic": 0}

Rules:
- not_toxic=1 means ALL other labels should be 0
- If any toxicity label is 1, not_toxic MUST be 0
- A comment can be both toxic AND passive_toxic (e.g., sarcastic insult)
"""


def _call_gpt4o(text: str) -> dict[str, int]:
    """Send a single comment to GPT-4o and parse multi-label response."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        max_tokens=100,
    )

    raw = response.choices[0].message.content.strip()
    try:
        labels = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse GPT-4o response: {raw}")
        labels = {label: 0 for label in LABELS}

    # Validate keys
    validated = {}
    for label in LABELS:
        val = labels.get(label, 0)
        validated[label] = 1 if val == 1 else 0

    # Enforce mutual exclusivity: not_toxic vs. others
    has_toxic = any(validated[l] == 1 for l in LABELS if l != "not_toxic")
    if has_toxic:
        validated["not_toxic"] = 0
    if validated["not_toxic"] == 1:
        for l in LABELS:
            if l != "not_toxic":
                validated[l] = 0

    return validated


def annotate_batch(
    input_path: Path | str,
    output_path: Path | str | None = None,
    text_column: str = "text_clean",
    batch_size: int | None = None,
) -> pd.DataFrame:
    """Run GPT-4o annotation on a CSV batch.

    Parameters
    ----------
    input_path : Path | str
        CSV with a text column.
    output_path : Path | str | None
        Where to save annotated CSV. Defaults to ``data/annotated/gpt4o_annotations.csv``.
    text_column : str
        Column containing text to annotate.
    batch_size : int | None
        Process only the first ``batch_size`` rows (for cost control).

    Returns
    -------
    pd.DataFrame
    """
    input_path = Path(input_path)
    output_path = (
        Path(output_path) if output_path else ANNOTATED_DIR / "gpt4o_annotations.csv"
    )

    df = pd.read_csv(input_path)
    if batch_size:
        df = df.head(batch_size)

    logger.info(f"Annotating {len(df)} texts with GPT-4o …")

    results: list[dict] = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="GPT-4o annotation"):
        text = str(row.get(text_column, row.get("text", "")))
        if not text.strip():
            labels = {label: 0 for label in LABELS}
        else:
            try:
                labels = _call_gpt4o(text)
            except Exception as exc:
                logger.warning(f"Row {idx} failed: {exc}")
                labels = {label: 0 for label in LABELS}

        result = {"text": text, "annotator": "gpt-4o"}
        result.update(labels)
        results.append(result)

    annotated = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(output_path, index=False)
    logger.info(f"GPT-4o annotations saved → {output_path}")
    return annotated


def compare_human_vs_llm(
    human_annotations: pd.DataFrame | dict[str, pd.DataFrame],
    llm_annotations: pd.DataFrame,
) -> pd.DataFrame:
    """Compare human annotators vs GPT-4o on the IAA batch.

    Computes Cohen's κ between each human annotator and GPT-4o per label.

    Parameters
    ----------
    human_annotations : pd.DataFrame | dict[str, pd.DataFrame]
        Either a combined DF with ``annotator`` column, or a dict of per-annotator DFs.
    llm_annotations : pd.DataFrame
        GPT-4o annotations with same text ordering.

    Returns
    -------
    pd.DataFrame
        Columns: ``annotator``, ``label``, ``kappa``.
    """
    from sklearn.metrics import cohen_kappa_score

    if isinstance(human_annotations, dict):
        annotators = human_annotations
    else:
        annotators = {
            name: group.sort_values("text").reset_index(drop=True)
            for name, group in human_annotations.groupby("annotator")
        }

    llm_sorted = llm_annotations.sort_values("text").reset_index(drop=True)

    records: list[dict] = []
    for ann_name, ann_df in sorted(annotators.items()):
        ann_sorted = ann_df.sort_values("text").reset_index(drop=True)
        for label in LABELS:
            kappa = cohen_kappa_score(ann_sorted[label].values, llm_sorted[label].values)
            records.append(
                {"annotator": ann_name, "vs": "gpt-4o", "label": label, "kappa": round(kappa, 4)}
            )
            logger.info(f"  κ({ann_name} vs gpt-4o) [{label}] = {kappa:.4f}")

    result = pd.DataFrame(records)
    result.to_csv(ANNOTATED_DIR / "human_vs_llm_agreement.csv", index=False)
    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPT-4o synthetic annotator")
    parser.add_argument("--input", type=str, required=True, help="CSV to annotate")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    annotate_batch(args.input, args.output, batch_size=args.batch_size)
