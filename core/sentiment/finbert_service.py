from __future__ import annotations

import time
from typing import Any


class FinBertService:
    """FinBERT sentiment scoring service.

    Loads ProsusAI/finbert from the local HuggingFace cache and scores
    news headlines. Falls back to 0.0 silently on any load or inference error.
    Results are cached per-symbol for 5 minutes to avoid re-running on every bar.
    """

    MODEL_NAME = "ProsusAI/finbert"
    CACHE_TTL_SEC = 300  # 5 minutes

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded: bool = False
        self._load_error: bool = False
        self._cache: dict[str, tuple[float, float]] = {}  # symbol -> (timestamp, score)

    @property
    def available(self) -> bool:
        return self._loaded and not self._load_error

    def _ensure_loaded(self) -> bool:
        """Attempt to load the model once; return True if available."""
        if self._loaded:
            return True
        if self._load_error:
            return False
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch  # noqa: F401

            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
            self._model.eval()
            self._loaded = True
            return True
        except Exception:
            self._load_error = True
            return False

    def score_texts(self, texts: list[str]) -> float:
        """Score a list of texts and return aggregate sentiment in [-1.0, +1.0].

        Uses the mean of (positive_prob - negative_prob) across all texts.
        Returns 0.0 on empty input or model failure.
        """
        if not texts:
            return 0.0
        if not self._ensure_loaded():
            return 0.0
        try:
            import torch

            scores: list[float] = []
            for text in texts:
                inputs = self._tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    padding=True,
                )
                with torch.no_grad():
                    logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze().tolist()
                # FinBERT label order: positive=0, negative=1, neutral=2
                if len(probs) == 3:
                    score = float(probs[0]) - float(probs[1])
                else:
                    score = 0.0
                scores.append(score)
            return float(sum(scores) / len(scores)) if scores else 0.0
        except Exception:
            return 0.0

    def score_symbol(self, symbol: str, headlines: list[str], *, fallback_score: float = 0.0) -> float:
        """Score headlines relevant to symbol, with 5-minute caching.

        Strips the /USD suffix to get the base token name (e.g. ETH/USD -> ETH)
        and filters headlines that mention it (case-insensitive).
        Returns 0.0 if no relevant headlines or model unavailable.
        """
        now = time.time()
        cached = self._cache.get(symbol)
        if cached is not None and (now - cached[0]) < self.CACHE_TTL_SEC:
            return cached[1]

        base = symbol.split("/")[0].upper()
        relevant = [h for h in headlines if base in h.upper()]
        score = self.score_texts(relevant)
        if not relevant or (score == 0.0 and not self.available):
            score = float(fallback_score)
        self._cache[symbol] = (now, score)
        return score
