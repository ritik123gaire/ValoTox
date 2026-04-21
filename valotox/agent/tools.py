"""
LangGraph agent tools for ValoTox.

These are plugged into the LangGraph agent as callable tools:
1. toxicity_classifier — runs the fine-tuned RoBERTa model
2. valorant_context_checker — detects if passive-toxic gaming slang is
   sarcastic vs. genuine using an LLM with game context
3. severity_scorer — ranks toxicity: passive → moderate → severe → slur
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from valotox.config import MODEL_DIR, settings
from valotox.models.dataset import CLASSIFICATION_LABELS

# ── Pydantic schemas ─────────────────────────────────────────────────────────


class ToxicityResult(BaseModel):
    """Output of the toxicity classifier tool."""

    text: str
    labels: dict[str, float] = Field(description="Label → confidence score")
    active_labels: list[str] = Field(description="Labels above threshold")
    severity: str = Field(description="passive | moderate | severe | slur")
    is_toxic: bool


class ContextCheckResult(BaseModel):
    """Output of the Valorant context checker."""

    text: str
    is_sarcastic: bool
    explanation: str
    adjusted_labels: dict[str, float]


# ── Tool 1: Toxicity Classifier ─────────────────────────────────────────────

_classifier = None


def _pick_model_dir() -> Any:
    candidates = [path for path in MODEL_DIR.glob("*/best") if path.is_dir()]
    if not candidates:
        return None

    def _sort_key(path):
        model_file = path / "model.safetensors"
        if not model_file.exists():
            model_file = path / "pytorch_model.bin"
        if not model_file.exists():
            model_file = path / "config.json"
        return (model_file.stat().st_mtime if model_file.exists() else path.stat().st_mtime, path.as_posix())

    candidates.sort(key=_sort_key, reverse=True)
    return candidates[0]


def _get_classifier():
    """Lazy-load the classifier model."""
    global _classifier
    if _classifier is None:
        from valotox.models.transformer import ToxicityClassifier

        model_path = _pick_model_dir()
        if model_path is None:
            logger.warning(f"No trained model found in {MODEL_DIR}; using heuristic fallback classifier.")
            _classifier = ToxicityClassifier()
            return _classifier
        _classifier = ToxicityClassifier(model_path)
    return _classifier


def classify_toxicity(text: str, threshold: float = 0.5) -> ToxicityResult:
    """Classify a single text for toxicity using the fine-tuned model.

    This is the primary agent tool — wraps the transformer model.
    """
    classifier = _get_classifier()
    predictions = classifier.predict(text)
    pred = predictions[0]

    configured_thresholds = {}
    cfg = getattr(classifier, "_cfg", None)
    if isinstance(cfg, dict):
        raw_thresholds = cfg.get("thresholds")
        if isinstance(raw_thresholds, dict):
            configured_thresholds = {
                str(k): float(v)
                for k, v in raw_thresholds.items()
                if k in CLASSIFICATION_LABELS and isinstance(v, (int, float))
            }

    # Respect calibrated per-label thresholds when caller uses the default slider value.
    if configured_thresholds and abs(float(threshold) - 0.5) < 1e-9:
        label_thresholds = {
            label: configured_thresholds.get(label, 0.5) for label in CLASSIFICATION_LABELS
        }
    else:
        label_thresholds = {label: float(threshold) for label in CLASSIFICATION_LABELS}

    toxic_labels = [lab for lab in CLASSIFICATION_LABELS if lab != "not_toxic"]
    toxic_active = [lab for lab in toxic_labels if pred.get(lab, 0.0) >= label_thresholds.get(lab, 0.5)]
    if toxic_active:
        active = toxic_active
        is_toxic = True
    else:
        is_toxic = False
        active = ["not_toxic"] if pred.get("not_toxic", 0.0) >= label_thresholds.get("not_toxic", 0.5) else []

    severity = _compute_severity(pred, label_thresholds=label_thresholds)

    return ToxicityResult(
        text=text,
        labels=pred,
        active_labels=active,
        severity=severity,
        is_toxic=is_toxic,
    )


# ── Tool 2: Valorant Context Checker ────────────────────────────────────────

CONTEXT_PROMPT = """You are analysing a comment from a Valorant gaming community.

Valorant-specific context:
- "diff" (e.g., "jett diff", "support diff") = claiming opponent was better → usually sarcastic/mocking
- "ez" / "ggez" = "easy game" → disrespectful when directed at opponents
- "skill issue" = blaming the victim → passive-aggressive
- "nice try" / "unlucky" = can be genuine encouragement OR sarcastic mockery
- "go next" = give up → can be pragmatic OR dismissive
- "bot" = calling someone AI-level bad
- "throwing" = accusation; can be factual observation OR toxic accusation

Given this comment, determine:
1. Is the comment being sarcastic/mocking or genuine?
2. Is the gaming context making otherwise-neutral words toxic?

Comment: "{text}"

Respond in JSON:
{{"is_sarcastic": true/false, "explanation": "brief reasoning", "passive_toxic_confidence": 0.0-1.0}}
"""


def check_valorant_context(text: str) -> ContextCheckResult:
    """Use an LLM to determine if passive-toxic slang is sarcastic vs. genuine.

    This addresses the key challenge of passive toxicity in gaming contexts.
    """
    import json

    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a Valorant community moderation expert. Respond only in valid JSON.",
            },
            {"role": "user", "content": CONTEXT_PROMPT.format(text=text)},
        ],
        max_tokens=200,
    )

    try:
        result = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        result = {"is_sarcastic": False, "explanation": "Parse error", "passive_toxic_confidence": 0.0}

    # Get base classifier predictions
    classifier = _get_classifier()
    base_pred = classifier.predict(text)[0]

    # Adjust passive_toxic score based on context
    adjusted = dict(base_pred)
    pt_conf = result.get("passive_toxic_confidence", 0.0)
    if result.get("is_sarcastic", False):
        adjusted["passive_toxic"] = max(adjusted.get("passive_toxic", 0), pt_conf)

    return ContextCheckResult(
        text=text,
        is_sarcastic=result.get("is_sarcastic", False),
        explanation=result.get("explanation", ""),
        adjusted_labels=adjusted,
    )


# ── Tool 3: Severity Scorer ─────────────────────────────────────────────────

SEVERITY_LEVELS = ["none", "passive", "moderate", "severe", "slur"]

SEVERITY_WEIGHTS = {
    "toxic": 2,
    "harassment": 3,
    "gender_attack": 4,
    "slur": 5,
    "passive_toxic": 1,
}


def _compute_severity(
    label_scores: dict[str, float],
    threshold: float = 0.5,
    label_thresholds: dict[str, float] | None = None,
) -> str:
    """Compute overall severity from label confidence scores.

    Ranking: none → passive → moderate → severe → slur
    """
    # Ignore ``not_toxic`` — it must not drive severity to "passive".
    active = {
        lbl: s
        for lbl, s in label_scores.items()
        if s >= (label_thresholds.get(lbl, threshold) if label_thresholds else threshold)
        and lbl != "not_toxic"
        and lbl in SEVERITY_WEIGHTS
    }

    if not active:
        return "none"

    max_weight = max(SEVERITY_WEIGHTS[lbl] for lbl in active)

    if max_weight >= 5:
        return "slur"
    elif max_weight >= 3:
        return "severe"
    elif max_weight >= 2:
        return "moderate"
    else:
        return "passive"


def score_severity(text: str) -> dict[str, Any]:
    """Full severity scoring pipeline.

    Combines model prediction with severity ranking.
    """
    result = classify_toxicity(text)
    return {
        "text": text,
        "severity": result.severity,
        "severity_level": SEVERITY_LEVELS.index(result.severity),
        "labels": result.labels,
        "active_labels": result.active_labels,
        "recommendation": _get_recommendation(result.severity),
    }


def _get_recommendation(severity: str) -> str:
    """Map severity to a moderation recommendation."""
    return {
        "none": "No action needed.",
        "passive": "Flag for review. Consider educational warning.",
        "moderate": "Issue warning. Temporary mute recommended.",
        "severe": "Immediate mute. Escalate to moderation team.",
        "slur": "Immediate ban. Report to trust & safety.",
    }.get(severity, "Review manually.")
