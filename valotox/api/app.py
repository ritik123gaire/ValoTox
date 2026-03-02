"""
FastAPI application for ValoTox toxicity detection.

Endpoints:
- POST /classify      — Direct model classification (fast, no LLM)
- POST /analyse       — Full agent pipeline with context analysis
- POST /batch         — Batch classification
- GET  /health        — Health check
- GET  /labels        — Return label schema
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from valotox.lexicon import LABELS
from valotox.models.dataset import CLASSIFICATION_LABELS


# ── Request / Response schemas ───────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Text to classify")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Classification threshold")


class ClassifyResponse(BaseModel):
    text: str
    labels: dict[str, float]
    active_labels: list[str]
    severity: str
    is_toxic: bool


class AnalyseRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class AnalyseResponse(BaseModel):
    text: str
    agent_response: str
    labels: dict[str, float] | None = None
    severity: str | None = None


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class BatchResponse(BaseModel):
    results: list[ClassifyResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


# ── App lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load model on startup."""
    logger.info("Loading ValoTox model …")
    try:
        from valotox.agent.tools import _get_classifier

        _get_classifier()
        app.state.model_loaded = True
        logger.info("Model loaded successfully")
    except Exception as exc:
        logger.warning(f"Model not loaded: {exc} — classify endpoints will fail")
        app.state.model_loaded = False
    yield
    logger.info("Shutting down ValoTox API")


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="ValoTox API",
    description="Hate Speech & Toxicity Detection for Valorant Communities",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check."""
    return HealthResponse(
        status="ok",
        model_loaded=getattr(app.state, "model_loaded", False),
        version="1.0.0",
    )


@app.get("/labels")
async def get_labels():
    """Return the ValoTox label schema."""
    return {
        "labels": LABELS,
        "classification_labels": CLASSIFICATION_LABELS,
        "description": {
            "toxic": "General insults, aggression",
            "harassment": "Targeted at a specific player",
            "gender_attack": "Misogyny / gender-based abuse",
            "slur": "Hate speech, slurs",
            "passive_toxic": "Subtle toxicity, sarcasm",
            "not_toxic": "Clean, friendly, game-related",
        },
    }


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    """Direct model classification — fast, no LLM calls."""
    if not getattr(app.state, "model_loaded", False):
        raise HTTPException(status_code=503, detail="Model not loaded")

    from valotox.agent.tools import classify_toxicity

    result = classify_toxicity(req.text, threshold=req.threshold)
    return ClassifyResponse(
        text=result.text,
        labels=result.labels,
        active_labels=result.active_labels,
        severity=result.severity,
        is_toxic=result.is_toxic,
    )


@app.post("/analyse", response_model=AnalyseResponse)
async def analyse(req: AnalyseRequest):
    """Full agent analysis — uses LLM + tools for context-aware moderation."""
    try:
        from valotox.agent.graph import run_agent

        result = run_agent(req.text)
        return AnalyseResponse(
            text=req.text,
            agent_response=result["agent_response"],
        )
    except Exception as exc:
        logger.error(f"Agent error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/batch", response_model=BatchResponse)
async def batch_classify(req: BatchRequest):
    """Batch classification — model only, no agent."""
    if not getattr(app.state, "model_loaded", False):
        raise HTTPException(status_code=503, detail="Model not loaded")

    from valotox.agent.tools import classify_toxicity

    results = []
    for text in req.texts:
        result = classify_toxicity(text, threshold=req.threshold)
        results.append(
            ClassifyResponse(
                text=result.text,
                labels=result.labels,
                active_labels=result.active_labels,
                severity=result.severity,
                is_toxic=result.is_toxic,
            )
        )

    return BatchResponse(results=results)
