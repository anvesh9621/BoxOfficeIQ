"""
SentimentAnalyzer: Inference wrapper for the DistilBERT sentiment model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """DistilBERT-based sentiment analysis for movie reviews.

    Parameters
    ----------
    model_path : str | Path
        Directory containing the saved DistilBertForSequenceClassification model.
    tokenizer_path : str | Path
        Directory containing the saved DistilBertTokenizer.
    max_length : int
        Maximum token length for inputs (must match training setting).
    batch_size : int
        Default batch size used by analyze_batch / aggregate_movie_sentiment.
    """

    _LABEL_MAP = {0: "negative", 1: "positive"}

    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: str | Path,
        max_length: int = 256,
        batch_size: int = 16,
    ) -> None:
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.max_length = max_length
        self.batch_size = batch_size

        self._model = None
        self._tokenizer = None
        self._device = None

        self._load_model()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the DistilBERT model and tokenizer from disk."""
        try:
            import torch
            from transformers import (
                DistilBertForSequenceClassification,
                DistilBertTokenizer,
            )
        except ImportError as exc:
            logger.error(
                "transformers / torch not installed — SentimentAnalyzer will not work. "
                f"Install with: pip install torch transformers  ({exc})"
            )
            return

        if not self.model_path.exists():
            logger.warning(
                f"Model path does not exist: {self.model_path}. "
                "Run ml/training/train_sentiment.py first."
            )
            return

        if not self.tokenizer_path.exists():
            logger.warning(
                f"Tokenizer path does not exist: {self.tokenizer_path}."
            )
            return

        try:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._tokenizer = DistilBertTokenizer.from_pretrained(
                str(self.tokenizer_path)
            )
            self._model = DistilBertForSequenceClassification.from_pretrained(
                str(self.model_path)
            )
            self._model.to(self._device)
            self._model.eval()
            logger.info(
                f"SentimentAnalyzer loaded on {self._device} "
                f"(model={self.model_path.name})"
            )
        except Exception as exc:
            logger.error(f"Failed to load sentiment model: {exc}")
            self._model = None
            self._tokenizer = None

    def _predict_batch(self, texts: List[str]) -> List[dict]:
        """Run inference on a list of texts. Returns list of result dicts."""
        import torch

        encodings = self._tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encodings = {k: v.to(self._device) for k, v in encodings.items()}

        with torch.no_grad():
            logits = self._model(**encodings).logits  # (B, 2)

        probs = torch.softmax(logits, dim=1).cpu().numpy()  # (B, 2)
        pred_ids = probs.argmax(axis=1)  # (B,)

        results = []
        for i, (pred_id, prob) in enumerate(zip(pred_ids, probs)):
            label = self._LABEL_MAP[int(pred_id)]
            # score = P(positive)
            score = float(prob[1])
            confidence = float(prob.max())
            results.append(
                {"label": label, "score": round(score, 4), "confidence": round(confidence, 4)}
            )
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def has_model_loaded(self) -> bool:
        """True if the model and tokenizer are loaded and ready for inference."""
        return self._model is not None and self._tokenizer is not None

    def analyze(self, text: str) -> dict:
        """Analyze the sentiment of a single review.

        Parameters
        ----------
        text : str
            The review text to analyze.

        Returns
        -------
        dict
            ``{"label": "positive" | "negative", "score": float, "confidence": float}``
            *score* is the probability of the positive class (0–1).
            *confidence* is the max class probability (0–1).

        Raises
        ------
        RuntimeError
            If the model has not been loaded.
        """
        if not self.has_model_loaded:
            raise RuntimeError(
                "Sentiment model is not loaded. "
                "Call SentimentAnalyzer with valid model/tokenizer paths."
            )
        results = self._predict_batch([text])
        return results[0]

    def analyze_batch(self, texts: List[str]) -> List[dict]:
        """Analyze the sentiment of multiple reviews.

        Parameters
        ----------
        texts : list[str]
            Review texts to analyze.

        Returns
        -------
        list[dict]
            One dict per input, each with keys ``label``, ``score``, ``confidence``.
        """
        if not self.has_model_loaded:
            raise RuntimeError("Sentiment model is not loaded.")
        if not texts:
            return []

        all_results: List[dict] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            all_results.extend(self._predict_batch(chunk))
        return all_results

    def aggregate_movie_sentiment(self, reviews: List[str]) -> dict:
        """Compute aggregate sentiment statistics for a movie's reviews.

        Parameters
        ----------
        reviews : list[str]
            Collection of review texts for a single movie.

        Returns
        -------
        dict
            ::

                {
                    "positive_pct": float,       # 0–100
                    "negative_pct": float,       # 0–100
                    "avg_score": float,          # mean P(positive), 0–1
                    "review_count": int,
                    "sample_positives": list[str],   # up to 3 positive reviews
                    "sample_negatives": list[str],   # up to 3 negative reviews
                }
        """
        if not reviews:
            return {
                "positive_pct": 0.0,
                "negative_pct": 0.0,
                "avg_score": 0.0,
                "review_count": 0,
                "sample_positives": [],
                "sample_negatives": [],
            }

        results = self.analyze_batch(reviews)

        positive_reviews: List[str] = []
        negative_reviews: List[str] = []
        total_score = 0.0

        for text, res in zip(reviews, results):
            total_score += res["score"]
            if res["label"] == "positive":
                positive_reviews.append(text)
            else:
                negative_reviews.append(text)

        n = len(reviews)
        positive_pct = round(len(positive_reviews) / n * 100, 2)
        negative_pct = round(len(negative_reviews) / n * 100, 2)
        avg_score = round(total_score / n, 4)

        # Sort sample reviews by confidence (highest first) — pick up to 3
        pos_with_conf = [
            (text, res["confidence"])
            for text, res in zip(reviews, results)
            if res["label"] == "positive"
        ]
        neg_with_conf = [
            (text, res["confidence"])
            for text, res in zip(reviews, results)
            if res["label"] == "negative"
        ]
        pos_with_conf.sort(key=lambda x: x[1], reverse=True)
        neg_with_conf.sort(key=lambda x: x[1], reverse=True)

        sample_positives = [t for t, _ in pos_with_conf[:3]]
        sample_negatives = [t for t, _ in neg_with_conf[:3]]

        return {
            "positive_pct": positive_pct,
            "negative_pct": negative_pct,
            "avg_score": avg_score,
            "review_count": n,
            "sample_positives": sample_positives,
            "sample_negatives": sample_negatives,
        }
