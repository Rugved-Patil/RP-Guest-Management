"""
utils/ml.py
------------
AI/Data Science helpers. Sentiment analysis on guest reviews lives
here first; no-show prediction, guest segmentation, and turnout
forecasting will land in this same file as those features get built.

--- What "sentiment analysis" means here ---
We want to turn a review's free-text ("Amazing event, loved it!") into
a number the Analytics page can chart. Rather than writing rules
ourselves, we use a HuggingFace "pipeline" -- a few lines of code that
load a model someone else already trained on millions of examples, so
it works immediately with no training of our own.

The model used here is "distilbert-base-uncased-finetuned-sst-2-english":
a small, fast transformer model fine-tuned specifically to classify
English text as positive or negative.

--- Why the caching matters ---
Loading the model is slow -- a few seconds to read its weights off
disk, or, the very first time it's ever used on your machine, several
minutes to download them (~260MB) from the internet. We do NOT want to
pay that cost every time someone submits a review. _get_pipeline()
below loads the model once and reuses that same loaded copy for every
call after that, no matter which page triggered it.
"""

from functools import lru_cache

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


@lru_cache(maxsize=1)
def _get_pipeline():
    """
    Load and cache the sentiment pipeline.

    @lru_cache(maxsize=1) is a standard-library decorator that
    remembers the return value of the first call and hands back that
    same value on every later call instead of re-running the function
    -- exactly the "load once, reuse everywhere" behavior we want.

    The `from transformers import pipeline` line lives inside this
    function on purpose, not at the top of the file. transformers is a
    fairly heavy library to import; keeping it out of the top-level
    imports means pages that never touch sentiment analysis (Create
    Event, Check-In, etc.) don't pay that cost just because they
    imported utils/db.py, which imports this file.
    """
    from transformers import pipeline
    return pipeline("sentiment-analysis", model=MODEL_NAME)


def _to_signed_score(result):
    """
    The model returns e.g. {"label": "POSITIVE", "score": 0.98} --
    a category plus a 0-1 confidence. To get a single signed number on
    the same -1..1 scale this project already uses for sentiment_score,
    we flip the sign for NEGATIVE results:
      POSITIVE, confidence 0.98  ->  +0.98
      NEGATIVE, confidence 0.98  ->  -0.98
    """
    signed = result["score"] if result["label"] == "POSITIVE" else -result["score"]
    return round(signed, 4)


def score_sentiment(text):
    """
    Score a single review's text.
    Returns a float from -1 (very negative) to +1 (very positive), or
    None if there's no text to score.
    """
    if not text or not text.strip():
        return None
    result = _get_pipeline()(text.strip(), truncation=True, max_length=512)[0]
    return _to_signed_score(result)


def score_sentiment_batch(texts):
    """
    Score a list of review texts in one call.

    Used by the sample-data seeder, which creates dozens of reviews at
    once -- running them through the model together as a single batch
    is much faster than calling score_sentiment() once per review in a
    loop, since the model does its work in one pass instead of many.

    Returns a list of scores in the same order as `texts`. Blank
    entries get None back, same as score_sentiment().
    """
    cleaned = [t.strip() if t else "" for t in texts]
    non_empty = [t for t in cleaned if t]
    if not non_empty:
        return [None] * len(cleaned)

    results = _get_pipeline()(non_empty, truncation=True, max_length=512)
    scored = iter(_to_signed_score(r) for r in results)
    return [next(scored) if t else None for t in cleaned]


def sentiment_label(score):
    """
    Turn a -1..1 score into a human-readable label for display:
    "Positive" or "Negative". A None score (not yet scored) comes back
    as "Unscored".

    There's no "Neutral" here on purpose -- this model only has two
    classes (see the module docstring above), so anything it scores
    always leans one way or the other. A middle "Neutral" band would
    just be mislabeling confident model output as uncertain.
    """
    if score is None:
        return "Unscored"
    return "Positive" if score >= 0 else "Negative"