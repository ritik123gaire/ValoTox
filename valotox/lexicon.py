"""
Valorant-specific toxicity lexicon for keyword-stratified sampling.

Organised by category so researchers can oversample specific label buckets.
Each category maps to one or more target annotation labels.
"""

from __future__ import annotations

# ── Label constants ──────────────────────────────────────────────────────────
LABELS: list[str] = [
    "toxic",
    "harassment",
    "gender_attack",
    "slur",
    "passive_toxic",
    "not_toxic",
]

# ── Keyword buckets ──────────────────────────────────────────────────────────

GENERAL_INSULTS: list[str] = [
    "braindead",
    "brain dead",
    "uninstall",
    "bot",
    "trash",
    "useless",
    "noob",
    "dogwater",
    "dog water",
    "garbage",
    "shitter",
    "hardstuck",
    "hard stuck",
    "washed",
    "plat chat",
    "iron brain",
    "boosted",
    "carried",
    "deadweight",
    "dead weight",
    "liability",
    "anchor",
    "baiter",
    "lurk andy",
    "0 iq",
    "zero iq",
    "smooth brain",
]

PASSIVE_TOXIC: list[str] = [
    "ez",
    "ggez",
    "gg ez",
    "diff",
    "ratio",
    "cope",
    "skill issue",
    "nice try",
    "unlucky",
    "just uninstall",
    "free elo",
    "free game",
    "not even close",
    "too easy",
    "go next",
    "ff 15",
    "ff15",
    "just ff",
    "first time?",
    "try hard",
    "tryhard",
    "stay mad",
    "cry about it",
    "mad cuz bad",
    "imagine losing",
    "you tried",
    "close game btw",
    "wp ig",
    "thanks for the rr",
    "thanks for free rr",
    "lol nt",
    "good effort tho",
]

THREATS_ESCALATION: list[str] = [
    "kys",
    "kill yourself",
    "reported",
    "you're done",
    "ur done",
    "get banned",
    "enjoy ban",
    "enjoy the ban",
    "gonna get you banned",
    "end yourself",
    "neck yourself",
    "rope",
    "unalive",
    "go die",
    "die irl",
    "ddos",
    "swat",
    "dox",
    "doxxed",
    "ip grabbed",
    "i know where you live",
]

GENDER_BASED: list[str] = [
    "go back to kitchen",
    "go back to the kitchen",
    "girls can't play",
    "girls cant play",
    "e-girl",
    "egirl",
    "e girl",
    "egirl duo",
    "boosted by bf",
    "boosted by boyfriend",
    "kitchen",
    "make me a sandwich",
    "sandwich",
    "dishwasher",
    "stay in normals",
    "girl gamer",
    "girl moment",
    "she's a girl that's why",
    "female moment",
    "female brain",
    "wh*re",
    "sl*t",
]

REGIONAL_SLANG: list[str] = [
    # Mumbai / South-East Asia server slang
    "NHK",
    "TMKC",
    "kal aana",
    "bhak",
    "mc",
    "bc",
    "bsdk",
    "gandu",
    "chutiya",
    "randi",
    "madarchod",
    "behenchod",
    # EU / Turkish
    "amk",
    "orospu",
    # SEA
    "bobo",
    "gago",
    "putang ina",
    "anjing",
    "kontol",
    "bangsat",
    "tolol",
]

GAMEPLAY_TOXIC: list[str] = [
    "throwing",
    "thrower",
    "threw",
    "griefing",
    "griefer",
    "smurf",
    "smurfing",
    "afk loser",
    "afk",
    "go afk",
    "rage quit",
    "ragequit",
    "trolling",
    "troll",
    "soft int",
    "inting",
    "int",
    "running it down",
    "flash me one more time",
    "stop baiting",
    "no comms",
    "diff bot",
    "team diff",
    "jett diff",
    "sage diff",
    "support diff",
    "aim diff",
]

# ── Aggregated lists ─────────────────────────────────────────────────────────

CATEGORY_MAP: dict[str, list[str]] = {
    "general_insults": GENERAL_INSULTS,
    "passive_toxic": PASSIVE_TOXIC,
    "threats_escalation": THREATS_ESCALATION,
    "gender_based": GENDER_BASED,
    "regional_slang": REGIONAL_SLANG,
    "gameplay_toxic": GAMEPLAY_TOXIC,
}

ALL_TOXIC_KEYWORDS: list[str] = sorted(
    set(
        GENERAL_INSULTS
        + PASSIVE_TOXIC
        + THREATS_ESCALATION
        + GENDER_BASED
        + REGIONAL_SLANG
        + GAMEPLAY_TOXIC
    )
)

# ── Label-to-category mapping (for stratified sampling guidance) ─────────
LABEL_CATEGORY_MAP: dict[str, list[str]] = {
    "toxic": ["general_insults", "gameplay_toxic"],
    "harassment": ["threats_escalation", "gameplay_toxic"],
    "gender_attack": ["gender_based"],
    "slur": ["regional_slang"],
    "passive_toxic": ["passive_toxic"],
    "not_toxic": [],
}


def get_regex_pattern(categories: list[str] | None = None) -> str:
    """Return a compiled regex-ready pattern for the given keyword categories.

    Parameters
    ----------
    categories : list[str] | None
        Subset of CATEGORY_MAP keys. ``None`` → all keywords.

    Returns
    -------
    str
        A ``|``-joined regex pattern (case-insensitive matching recommended).
    """
    import re

    if categories is None:
        keywords = ALL_TOXIC_KEYWORDS
    else:
        keywords = []
        for cat in categories:
            keywords.extend(CATEGORY_MAP.get(cat, []))
    # Escape special regex chars and join
    escaped = [re.escape(kw) for kw in sorted(set(keywords))]
    return "|".join(escaped)
