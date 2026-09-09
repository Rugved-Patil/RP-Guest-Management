"""
pages/admin_analytics.py
--------------------------
Admin-facing page: sentiment analysis on guest reviews.

Every review's sentiment_score -- computed by the HuggingFace model in
utils/ml.py, either the moment a guest submits it on review.py, or in
one batch when sample data is generated -- gets summarized here: an
overall breakdown, a per-event comparison (the "sentiment trend across
events" feature from the scope doc), and a browsable, filterable table.

Guest segmentation, no-show prediction, and turnout forecasting will
get added to this same page later as those pieces get built.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
from utils.db import init_db, get_all_reviews_detailed
from utils.ml import sentiment_label

init_db()

st.title("Analytics")

reviews = get_all_reviews_detailed()

if not reviews:
    st.info(
        "No reviews yet -- once guests start leaving reviews (or you add "
        "sample data on the Data Tools page), sentiment charts will show "
        "up here."
    )
    st.stop()

# --- Turn the raw rows into something easy to chart/filter -------------
# pandas' DataFrame is basically a spreadsheet inside Python: rows plus
# named columns, with grouping/averaging (df.groupby(...).mean()) built
# in -- what'd otherwise be several lines of manual looping.
rows = []
for r in reviews:
    rows.append({
        "guest_name": r["guest_name"],
        "event_id": r["event_id"],
        "event_name": r["event_name"],
        "event_date": r["event_date"],
        "rating": r["rating"],
        "sentiment_score": r["sentiment_score"],
        "sentiment_label": sentiment_label(r["sentiment_score"]),
        "review_text": r["review_text"],
    })
df = pd.DataFrame(rows)

# sentiment_score is None for any review that hasn't been scored (see
# the caption below for when that happens) -- pandas stores those as
# NaN, and .notna() is how you filter those out before averaging.
scored = df[df["sentiment_score"].notna()]
unscored_count = len(df) - len(scored)

# --- KPI row -------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total reviews", len(df))

if len(scored):
    kpi2.metric("Average sentiment", f"{scored['sentiment_score'].mean():+.2f}")
    pct_positive = (scored["sentiment_label"] == "Positive").mean() * 100
    kpi3.metric("% Positive", f"{pct_positive:.0f}%")
else:
    kpi2.metric("Average sentiment", "--")
    kpi3.metric("% Positive", "--")

kpi4.metric("Unscored", unscored_count)

if unscored_count:
    st.caption(
        f"{unscored_count} review(s) don't have a sentiment score yet -- "
        "they were submitted before sentiment scoring was wired up, or "
        "came from an older sample-data batch. Resetting and re-adding "
        "sample data on the Data Tools page regenerates everything with "
        "real scores."
    )

st.divider()

# --- Sentiment distribution ----------------------------------------------
st.subheader("Sentiment breakdown")

LABEL_ORDER = ["Positive", "Negative", "Unscored"]
LABEL_COLORS = {
    "Positive": "#2ecc71",
    "Negative": "#e74c3c",
    "Unscored": "#dfe6e9",
}

counts = df["sentiment_label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
counts = counts[counts > 0]  # don't show a 0-size slice for a label nothing falls into

fig_dist = px.pie(
    names=counts.index,
    values=counts.values,
    color=counts.index,
    color_discrete_map=LABEL_COLORS,
    hole=0.4,
)
fig_dist.update_traces(textinfo="label+percent")
st.plotly_chart(fig_dist, use_container_width=True)

st.divider()

# --- Average sentiment per event ------------------------------------------
st.subheader("Sentiment by event")

if len(scored):
    per_event = (
        scored.groupby(["event_id", "event_name", "event_date"])["sentiment_score"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("event_date")  # ISO dates sort correctly as plain strings
    )
    per_event["event_label"] = per_event.apply(
        lambda row: f"{row['event_name']} ({date.fromisoformat(row['event_date']).strftime('%d-%m-%Y')})",
        axis=1,
    )

    fig_events = px.bar(
        per_event,
        x="event_label",
        y="mean",
        color="mean",
        color_continuous_scale=["#e74c3c", "#95a5a6", "#2ecc71"],  # red -> grey -> green
        range_color=[-1, 1],
        labels={"mean": "Avg. sentiment", "event_label": "Event", "count": "Reviews"},
        hover_data={"count": True, "mean": ":.2f"},
    )
    fig_events.update_layout(xaxis_tickangle=-30, coloraxis_showscale=False)
    st.plotly_chart(fig_events, use_container_width=True)
else:
    st.info("No scored reviews yet to compare across events.")

st.divider()

# --- Review table (filterable) --------------------------------------------
st.subheader("All reviews")

event_options = ["All events"] + sorted(df["event_name"].unique().tolist())
sentiment_options = ["All"] + LABEL_ORDER

filter_col, sentiment_col = st.columns(2)
with filter_col:
    event_filter = st.selectbox("Filter by event", event_options)
with sentiment_col:
    sentiment_filter = st.selectbox("Filter by sentiment", sentiment_options)

filtered = df
if event_filter != "All events":
    filtered = filtered[filtered["event_name"] == event_filter]
if sentiment_filter != "All":
    filtered = filtered[filtered["sentiment_label"] == sentiment_filter]

display_rows = []
for _, r in filtered.iterrows():
    # pandas stores a column that mixes numbers and Nones as floats,
    # turning missing ratings into NaN rather than None -- pd.notna()
    # is the correct way to check for "missing" here (a plain
    # `is not None` check would miss it, since NaN is not None).
    rating_display = int(r["rating"]) if pd.notna(r["rating"]) else "--"
    display_rows.append({
        "Event": r["event_name"],
        "Event Date": date.fromisoformat(r["event_date"]).strftime("%d-%m-%Y"),
        "Guest": r["guest_name"],
        "Rating": rating_display,
        "Sentiment": r["sentiment_label"],
        "Score": f"{r['sentiment_score']:+.2f}" if pd.notna(r["sentiment_score"]) else "--",
        "Review": r["review_text"],
    })

st.caption(f"Showing {len(display_rows)} of {len(df)} reviews.")
st.dataframe(display_rows, use_container_width=True)