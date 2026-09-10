"""
utils/ml.py
------------
AI/Data Science helpers: sentiment analysis on guest reviews, and
no-show prediction. Guest segmentation and turnout forecasting will
land in this same file as those features get built.

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

--- What "no-show prediction" means here ---
Unlike sentiment (which reads text), this trains a small model of our
own -- a logistic regression -- on your actual past registrations: for
every guest who registered for a past event, did they show up or not?
Once it's found a pattern in that history, it applies the same pattern
to guests who've registered for events that haven't happened yet, and
gives each one a 0-100% no-show risk.
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


# ---------------------------------------------------------------------
# No-show prediction
# ---------------------------------------------------------------------
# A registration only ever has one of three attendance_status values:
#   "attended" / "no_show"  -- the event already happened; the real
#                              outcome is known, so these rows are what
#                              the model trains on.
#   "registered"            -- the event hasn't happened yet; the
#                              outcome is unknown, so these are the
#                              rows we generate a prediction for.

# Below this many resolved (attended/no_show) rows, there just isn't
# enough history for a logistic regression to find a reliable pattern
# -- it'd effectively be memorizing noise. 10 is a low bar on purpose,
# since a small portfolio-project dataset won't have thousands of rows;
# it's enough to demo the mechanism, not to make the model rigorous.
MIN_TRAINING_ROWS = 10


def _guest_history_totals(history_rows):
    """
    For every guest appearing in history_rows, count how many resolved
    (attended/no_show) registrations they have and how many of those
    were no-shows. Returns a dict: guest_id -> (no_show_count, total_count).

    Computed once as a lookup table (rather than re-scanning all of
    history_rows for every single row elsewhere) so building features
    for N registrations stays fast even as N grows -- one pass here
    instead of, effectively, N passes later.
    """
    totals = {}
    for row in history_rows:
        guest_id = row["guest_id"]
        no_show_count, total_count = totals.get(guest_id, (0, 0))
        total_count += 1
        if row["attendance_status"] == "no_show":
            no_show_count += 1
        totals[guest_id] = (no_show_count, total_count)
    return totals


def _build_features(target_rows, history_rows, leave_one_out, fallback_rate):
    """
    Build the feature table the model trains or predicts on: each
    row's guest's historical no-show rate and event count, plus a
    one-hot column per event tag (e.g. tag_Sports = 1 if this event is
    tagged "Sports", else 0).

    leave_one_out=True is used for the *training* rows: it subtracts
    each row's own outcome from its own guest's totals before
    computing that guest's rate. Without this, a guest with only one
    past event would always show a 0% or 100% historical no-show rate
    that exactly matches what we're asking the model to predict for
    that very same row -- the model would look artificially perfect by
    partly "seeing the answer" for every row with little guest history.
    Rows actually being predicted for (leave_one_out=False) don't need
    this: their outcome isn't in history_rows at all yet, since it
    hasn't happened.

    fallback_rate is used for any guest with zero *other* resolved
    events (a first-timer, or someone whose only history is the row
    being left out) -- there's no personal history to compute a rate
    from, so they're assumed to behave like an average guest until
    proven otherwise.

    Returns a pandas DataFrame, one row per item in target_rows, in
    the same order.
    """
    import pandas as pd

    totals = _guest_history_totals(history_rows)

    no_show_rates = []
    event_counts = []
    tags = []
    for row in target_rows:
        no_show_count, total_count = totals.get(row["guest_id"], (0, 0))

        if leave_one_out:
            total_count -= 1
            if row["attendance_status"] == "no_show":
                no_show_count -= 1

        if total_count > 0:
            rate = no_show_count / total_count
        else:
            rate = fallback_rate

        no_show_rates.append(rate)
        event_counts.append(total_count)
        tags.append(row["tag"])

    features = pd.DataFrame({
        "guest_past_no_show_rate": no_show_rates,
        "guest_past_event_count": event_counts,
    })
    tag_dummies = pd.get_dummies(pd.Series(tags, name="tag"), prefix="tag")
    return pd.concat([features, tag_dummies], axis=1)


def train_and_predict_noshow(rows):
    """
    Train a logistic regression on past (resolved) registrations and
    predict no-show probability for upcoming ("registered") ones.

    `rows` should come from db.get_registrations_for_noshow_model() --
    passed in rather than queried here so this file never has to
    import utils/db.py; db.py already imports *this* file (for
    sentiment scoring), and a file can't import something that in turn
    imports it back -- that's a "circular import" and Python will
    refuse to run it. Keeping the dependency one-directional (db.py ->
    ml.py, never the reverse) avoids that entirely.

    Returns a dict describing what happened:
      status: "not_enough_data" | "no_predictions_needed" | "ok"
      predictions: [(registration_id, no_show_probability), ...]
      coefficients: {feature_name: coefficient} or None
      train_rows: how many resolved registrations were trained on
      train_accuracy: accuracy on that same training data, or None

    train_accuracy is measured on the training data itself (not a
    separate held-out set) -- for a small sample-data-sized project
    this is a simple, honest-enough gut check ("did the model learn
    *something*"), not a rigorous evaluation. A real deployment would
    hold out a test set instead.
    """
    if not rows:
        return {
            "status": "not_enough_data", "predictions": [],
            "coefficients": None, "train_rows": 0, "train_accuracy": None,
        }

    resolved = [r for r in rows if r["attendance_status"] in ("attended", "no_show")]
    upcoming = [r for r in rows if r["attendance_status"] == "registered"]

    resolved_no_show_count = sum(1 for r in resolved if r["attendance_status"] == "no_show")
    has_both_classes = 0 < resolved_no_show_count < len(resolved)

    if len(resolved) < MIN_TRAINING_ROWS or not has_both_classes:
        return {
            "status": "not_enough_data", "predictions": [],
            "coefficients": None, "train_rows": len(resolved), "train_accuracy": None,
        }

    if not upcoming:
        return {
            "status": "no_predictions_needed", "predictions": [],
            "coefficients": None, "train_rows": len(resolved), "train_accuracy": None,
        }

    from sklearn.linear_model import LogisticRegression

    fallback_rate = resolved_no_show_count / len(resolved)

    # Training features use leave-one-out history (each row excludes
    # its own outcome); prediction features use each guest's full known
    # history, since none of it includes the row being predicted.
    X_train = _build_features(resolved, resolved, leave_one_out=True, fallback_rate=fallback_rate)
    X_predict = _build_features(upcoming, resolved, leave_one_out=False, fallback_rate=fallback_rate)

    # A tag might appear in one set of rows but not the other (e.g. no
    # upcoming Dance events right now) -- align both feature tables to
    # the same columns so the model sees a consistent shape either way.
    all_columns = sorted(set(X_train.columns) | set(X_predict.columns))
    X_train = X_train.reindex(columns=all_columns, fill_value=0)
    X_predict = X_predict.reindex(columns=all_columns, fill_value=0)

    y_train = [1 if r["attendance_status"] == "no_show" else 0 for r in resolved]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)

    # predict_proba returns [P(attended), P(no_show)] per row -- we
    # only want the second column.
    no_show_probabilities = model.predict_proba(X_predict)[:, 1]
    predictions = [
        (row["registration_id"], round(float(prob), 4))
        for row, prob in zip(upcoming, no_show_probabilities)
    ]

    coefficients = dict(zip(X_train.columns, (round(c, 4) for c in model.coef_[0])))

    return {
        "status": "ok",
        "predictions": predictions,
        "coefficients": coefficients,
        "train_rows": len(resolved),
        "train_accuracy": round(float(train_accuracy), 4),
    }