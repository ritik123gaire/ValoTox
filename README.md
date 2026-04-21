# ValoTox: A Hate Speech & Toxicity Dataset from Valorant Communities with a Fairness-Audited Detection System

> Riot Games issued **500,000+ warnings and bans** for toxic behavior in 2024 alone — the problem is real, documented, and urgent.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg)]()
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)]()

---

## What Makes This Project Unique

| Feature | Detail |
|---------|--------|
| **Domain-specific dataset** | First NLP dataset targeting Valorant community toxicity |
| **6-label multi-label schema** | `toxic`, `harassment`, `gender_attack`, `slur`, `passive_toxic`, `not_toxic` |
| **Passive toxicity detection** | Captures subtle gaming slang ("ggez", "diff", "skill issue") that generic models miss |
| **Reddit-focused collection** | Reddit API pipeline for stable, reproducible collection |
| **GPT-4o synthetic annotator** | Human vs. LLM agreement comparison as a standalone finding |
| **Fairness-audited** | Cross-domain evaluation (Jigsaw → Valorant) exposes generalization gaps |
| **LangGraph agent** | Context-aware moderation with Valorant-specific tool calling |

---

## Project Structure

```
VALO_Toxicity_Detection_System/
├── valotox/
│   ├── __init__.py
│   ├── config.py                  # Centralised settings (.env)
│   ├── lexicon.py                 # Valorant toxicity keyword lexicon
│   ├── eda.py                     # Exploratory Data Analysis & visualisations
│   ├── scraping/
│   │   ├── reddit_scraper.py      # PRAW-based Reddit scraper
│   │   └── data_pipeline.py       # Merge, clean, deduplicate, split
│   ├── annotation/
│   │   ├── label_studio.py        # Label Studio integration
│   │   ├── iaa.py                 # Inter-Annotator Agreement (Cohen/Fleiss κ)
│   │   └── gpt4o_annotator.py     # GPT-4o synthetic annotator
│   ├── models/
│   │   ├── dataset.py             # HuggingFace dataset loader & tokenizer
│   │   ├── baseline.py            # TF-IDF + Logistic Regression
│   │   ├── transformer.py         # ToxicityClassifier (lexicon heuristic; checkpoint metadata)
│   │   └── benchmark.py           # Full benchmarking pipeline
│   ├── agent/
│   │   ├── tools.py               # Classifier, context checker, severity scorer
│   │   └── graph.py               # LangGraph agent definition
│   └── api/
│       └── app.py                 # FastAPI application
├── data/
│   ├── raw/                       # Scraped data (git-ignored)
│   ├── processed/                 # Cleaned & merged data
│   ├── annotated/                 # Human + LLM annotations
│   └── figures/                   # EDA plots (from `valotox.cli eda`)
├── models/                        # Trained artifacts & checkpoints (see `valotox.cli train`)
├── notebooks/                     # Jupyter notebooks
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example
└── README.md
```

---

## How the system runs

Everything is driven from a **single CLI** (`valotox/cli.py`) and optional **HTTP API** (`valotox/api/app.py`). Configuration lives in **`.env`** (see `.env.example`) and is loaded by `valotox/config.py` (`pydantic-settings`).

### Unified CLI

```text
python -m valotox.cli <command> [options]
```

| Command | What it does |
|--------|----------------|
| `scrape` | Collect Reddit posts (PRAW; needs Reddit vars in `.env`) |
| `merge` | Merge/clean raw sources; add `--split` for annotation splits; `--reddit-conda` merges processed Reddit with a CONDA in-game CSV |
| `annotate` | Label Studio project import/export; optional `--gpt4o` batch |
| `iaa` | Inter-annotator agreement report |
| `eda` | Exploratory analysis (figures under `data/figures/` by default) |
| `train` | Training / benchmark (`--models`, `--epochs`, …) |
| `predict` | Run the classifier on a CSV column (`--input`, `--text-column`, `--threshold`) |
| `interactive` | Terminal REPL, or one-shot `-t "text"` / `--threshold` |
| `serve` | Start the FastAPI app with **uvicorn** (`--host`, `--port`, `--reload`) |

Run `python -m valotox.cli` with no subcommand to print built-in help.

### Classification runtime

1. **Call site** — `POST /classify`, `POST /batch`, LangGraph tools, or `python -m valotox.cli interactive` all go through `classify_toxicity()` in `valotox/agent/tools.py` (API preloads the classifier on startup).
2. **Model handle** — `_get_classifier()` instantiates `ToxicityClassifier` from `valotox/models/transformer.py`. It prefers a checkpoint directory under `models/` (e.g. `models/roberta-valotox/best`). If none exists, it logs a warning and uses the same **lexicon-based** implementation.
3. **Scores** — The bundled `ToxicityClassifier` produces multi-label confidences from Valorant-focused regex categories, passive-toxic phrases, and English slur word-boundary checks (`valotox/lexicon.py`). An optional `model_config.json` inside a checkpoint folder can adjust settings such as the default threshold.
4. **Decision** — Toxic labels (`toxic`, `harassment`, `gender_attack`, `slur`, `passive_toxic`) above the threshold drive `is_toxic` and `active_labels`; `not_toxic` is only surfaced as active when no toxic label clears the bar. Severity is derived from the toxic labels, not from `not_toxic`.

So the repo runs **out of the box** for demos: no GPU weights are required for the CLI/API to return scores (they are heuristic until you plug in a trained checkpoint workflow).

### Try it

```bash
python -m valotox.cli interactive
python -m valotox.cli interactive -t "good luck everyone" --threshold 0.5
python -m valotox.cli serve --reload --port 8000
```

---

## Quick Start

### 1. Setup

```bash
# Clone
git clone https://github.com/your-username/VALO_Toxicity_Detection_System.git
cd VALO_Toxicity_Detection_System

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install
pip install -e ".[dev]"

# Configure credentials
copy .env.example .env
# Edit .env with your API keys
```

### 2. Scrape Data (Phase 2)

```bash
# Reddit (requires PRAW credentials in .env)
python -m valotox.cli scrape --posts 500

# Merge Reddit sources (add --split for annotation splits)
python -m valotox.cli merge --split
```

### 2b. Stream from Academic Torrents / Pushshift (.zst)

Use this when you have raw `RC_*.zst` / `RS_*.zst` files from the Reddit dumps and want
memory-safe, line-by-line ingestion.

```bash
# Example: stream one monthly submissions file
python -m valotox.scraping.reddit_scraper \
  --stream-zst data/raw/reddit/RS_2025-06.zst \
  --subreddits VALORANT ValorantMemes ValorantCompetitive AgentAcademy \
  --output data/raw/reddit_raw.csv

# Example: stream multiple files via glob and keep only keyword-matching rows
python -m valotox.scraping.reddit_scraper \
  --stream-glob "data/raw/reddit/RS_2024-*.zst" \
  --keyword-only \
  --max-rows 200000 \
  --output data/raw/reddit_raw.csv

# Optional: comments only / submissions only
# --comments-only OR --submissions-only
```

Then continue with the normal pipeline:

```bash
python -m valotox.cli merge --split
```

### 3. Annotate (Phase 3)

```bash
# Start Label Studio
docker compose up label-studio

# Create project & import data (equivalent: python -m valotox.annotation.label_studio …)
python -m valotox.cli annotate --create-project
python -m valotox.cli annotate --import-csv data/processed/annotation_iaa.csv --project-id 1

# After annotation, export and compute IAA
python -m valotox.cli annotate --export --project-id 1
python -m valotox.cli iaa

# Run GPT-4o synthetic annotator on IAA batch (needs OPENAI_API_KEY)
python -m valotox.cli annotate --gpt4o data/processed/annotation_iaa.csv --gpt4o-batch-size 600
```

### 4. EDA (Phase 4)

```bash
python -m valotox.cli eda --input data/annotated/valotox_annotated.csv
# → Figures and summaries under data/figures/
```

### 5. Train Models (Phase 5)

```bash
# Full benchmark (all models)
python -m valotox.cli train --epochs 5 --batch-size 16

# Specific models only
python -m valotox.cli train --models roberta hatebert --epochs 5

# Cross-domain: Jigsaw → ValoTox
python -m valotox.cli train --jigsaw
```

### 6. Deploy API (Phase 6)

```bash
# Local (same app as: uvicorn valotox.api.app:app --reload --port 8000)
python -m valotox.cli serve --reload --port 8000

# Docker
docker compose up --build

# Test
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "ggez diff uninstall bot"}'
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/classify` | Direct model classification (fast, no LLM) |
| `POST` | `/analyse` | Full LangGraph agent with context analysis |
| `POST` | `/batch` | Batch classification (up to 100 texts) |
| `GET` | `/health` | Health check |
| `GET` | `/labels` | Label schema documentation |

---

## Label Schema

| Label | Description | Example |
|-------|-------------|---------|
| `toxic` | General insults, aggression | "you're so braindead uninstall" |
| `harassment` | Targeted at a specific player | "reported you, get rekt" |
| `gender_attack` | Misogyny / gender-based abuse | "go back to the kitchen" |
| `slur` | Hate speech, slurs | *(explicit examples omitted)* |
| `passive_toxic` | Subtle toxicity, sarcasm | "ggez wp diff", "nice try lol" |
| `not_toxic` | Clean, friendly, game-related | "good game everyone!" |

> **`passive_toxic`** is the key research contribution — it's the hardest to detect and largely ignored in existing datasets.

---

## Model Benchmark

| Model | Role |
|-------|------|
| TF-IDF + LogReg | Classical baseline |
| DistilBERT | Lightweight transformer |
| RoBERTa-base | Strong general transformer |
| HateBERT | Pre-trained on Reddit hate speech |
| **RoBERTa-ValoTox** | **Fine-tuned on ValoTox (main contribution)** |
| Jigsaw Toxic-BERT | Cross-domain generalization test |

### Research Questions Addressed

1. **Does a Jigsaw-trained model generalise to Valorant?** → Hypothesis: No — Valorant slang breaks it
2. **Does passive_toxic require different modeling?** → Expected lower F1, documented as finding
3. **Does HateBERT outperform generic RoBERTa on gaming content?** → Empirically tested

---

## LangGraph Agent Architecture

```
User Input
    │
    ▼
┌─────────┐     ┌──────────────────────┐
│  Agent   │────▶│  classify_text()     │  ← Fine-tuned RoBERTa
│  (GPT-4o │     └──────────────────────┘
│   mini)  │
│          │     ┌──────────────────────┐
│          │────▶│  check_context()     │  ← Valorant slang analyser
│          │     └──────────────────────┘
│          │
│          │     ┌──────────────────────┐
│          │────▶│  score_severity()    │  ← passive→moderate→severe→slur
└─────────┘     └──────────────────────┘
    │
    ▼
Moderation Decision
```

---

## Data Sources

| Source | What You Get | Tool |
|--------|-------------|------|
| r/VALORANT, r/ValorantMemes, r/ValorantCompetitive, r/AgentAcademy | Post-game rants, toxic screenshots, complaints | PRAW |

**Target:** 15,000–20,000 raw samples → 8,000–12,000 after filtering

---

## License

MIT License — see [LICENSE](LICENSE) for details.
