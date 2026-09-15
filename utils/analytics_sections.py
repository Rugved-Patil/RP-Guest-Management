"""
utils/analytics_sections.py
-----------------------------
Rendering functions shared between pages/admin_analytics.py and
pages/organizer_analytics.py.

Admin's page and the Organizer's page show almost the same charts --
sentiment, no-show risk, attendance-over-time, turnout forecast --
just scoped differently: Admin sees everything, an Organizer sees
only their own events. Rather than have two ~150-line copies of the
same chart code slowly drift apart, each function here takes an
optional `owned_event_ids` set. Leave it as None and everything shows
(that's what Admin's page does); pass a set of event_ids and the
function filters down to just those events (that's what an
Organizer's page does).

One thing this does NOT change: the no-show and turnout models still
always TRAIN on every registration/event system-wide, never just the
current viewer's own events, even when this file is called from the
Organizer's page. Two real reasons for that, not just convenience:

1. A guest's history (used as a no-show feature) naturally spans
   every organizer's events, not just one -- narrowing training data
   to a single organizer's events would break the leave-one-out
   history feature in utils/ml.py, which already assumes it's looking
   at a guest's whole history.
2. A new Organizer might have only 1-2 events of their own -- nowhere
   near MIN_TRAINING_EVENTS/enough resolved registrations to train
   anything meaningful. Training on the shared, system-wide history
   and only filtering what gets *displayed* means every Organizer
   gets predictions backed by a real amount of data, from day one.

Practically, this means clicking "Run" from the Organizer's page
still recomputes and saves predictions for EVERY upcoming
registration/event system-wide (same as Admin's "Run" button does) --
it's a shared model over shared history, not a private one. Only the
table/chart underneath is scoped to what that viewer is allowed to
see.

Guest segmentation is deliberately NOT here. It tags a *guest*
(VIP/Regular/New) based on their attendance across their whole
history, not any one event -- there's no clean way to scope that to
"one organizer's events" without the label itself becoming
misleading, so it stays Admin-only, in admin_analytics.py.
"""

import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import date

from utils.db import (
    get_all_reviews_detailed,
    get_all_registrations_detailed,
    get_registrations_for_noshow_model,
    bulk_set_predicted_no_show,
    get_events_for_forecast,
    bulk_set_predicted_turnout,
)
from utils.ml import (
    sentiment_label,
    train_and_predict_noshow,
    train_and_predict_turnout,
    MIN_TRAINING_EVENTS,
    _is_resolved_event,
)


def _risk_level(probability):
    """Turn a 0-1 no-show probability into a plain-English bucket for
    the table below. Thresholds are just reasonable defaults, not
    derived from anything -- tune them if 60%/35% doesn't feel right
    against real predictions."""
    if probability is None:
        return "Not yet scored"
    if probability >= 0.6:
        return "High"
    if probability >= 0.35:
        return "Medium"
    return "Low"


def render_sentiment_section(owned_event_ids=None):
    st.header("Guest sentiment")

    reviews = get_all_reviews_detailed()
    if owned_event_ids is not None:
        reviews = [r for r in reviews if r["event_id"] in owned_event_ids]

    if not reviews:
        st.info(
            "No reviews yet -- once guests start leaving reviews (or you "
            "add sample data on the Data Tools page), sentiment charts "
            "will show up here."
        )
        return

    # --- Turn the raw rows into something easy to chart/filter ---------
    # pandas' DataFrame is basically a spreadsheet inside Python: rows
    # plus named columns, with grouping/averaging (df.groupby(...).mean())
    # built in -- what'd otherwise be several lines of manual looping.
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
    # NaN, and .notna() filters those out before averaging.
    scored = df[df["sentiment_score"].notna()]
    unscored_count = len(df) - len(scored)

    # --- KPI row ---------------------------------------------------------
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

    # --- Sentiment distribution ------------------------------------------
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

    # --- Average sentiment per event --------------------------------------
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

    # --- Review table (filterable) ------------------------------------------
    st.subheader("All reviews")

    event_options = ["All events"] + sorted(df["event_name"].unique().tolist())
    sentiment_options = ["All"] + LABEL_ORDER

    filter_col, sentiment_col = st.columns(2)
    with filter_col:
        event_filter = st.selectbox("Filter by event", event_options, key="sentiment_event_filter")
    with sentiment_col:
        sentiment_filter = st.selectbox("Filter by sentiment", sentiment_options, key="sentiment_sentiment_filter")

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


def render_noshow_section(owned_event_ids=None):
    st.header("No-show risk prediction")
    st.write(
        "Trains a logistic regression on past events system-wide -- who "
        "registered, and whether they actually showed up -- then scores "
        "every guest who's registered for an event that hasn't happened "
        "yet with a no-show risk from 0-100%. Uses two signals: that "
        "guest's own history of attending/skipping past events, and the "
        "event's tag."
        + (
            " Training always uses data across the whole system (a "
            "guest's history isn't limited to your events alone, and it "
            "gives the model enough to work with even if you've only "
            "run a couple of events) -- the table below just shows "
            "scores for registrations to *your* upcoming events."
            if owned_event_ids is not None else ""
        )
    )

    if st.button("Run No-Show Prediction", key="noshow_run_button"):
        with st.spinner("Training model and scoring upcoming registrations..."):
            model_rows = get_registrations_for_noshow_model()
            result = train_and_predict_noshow(model_rows)

        if result["status"] == "not_enough_data":
            st.warning(
                f"Only {result['train_rows']} past registration(s) with a "
                "known outcome so far (need at least a handful of resolved "
                "check-ins -- attended or no-show -- covering both outcomes "
                "before there's enough history to train on). Add more "
                "sample data on the Data Tools page, or wait for more real "
                "check-ins, then try again."
            )
        elif result["status"] == "no_predictions_needed":
            st.info(
                "Nothing to predict right now -- every current registration "
                "is already resolved (attended or no-show). Predictions "
                "show up here once guests register for an upcoming event."
            )
        else:
            bulk_set_predicted_no_show(result["predictions"])
            st.success(
                f"Scored {len(result['predictions'])} registration(s), "
                f"trained on {result['train_rows']} past ones "
                f"({result['train_accuracy'] * 100:.0f}% accuracy on that "
                "training data)."
            )

            with st.expander("What the model learned"):
                st.caption(
                    "Each bar is one feature's effect on the prediction -- "
                    "positive pushes toward 'no-show', negative pushes "
                    "toward 'attended'. Longer bars (either direction) "
                    "mattered more to the model; this is measured only on "
                    "the training data itself, so treat it as a rough "
                    "gut-check rather than a rigorous evaluation."
                )
                coef_df = pd.DataFrame({
                    "feature": list(result["coefficients"].keys()),
                    "coefficient": list(result["coefficients"].values()),
                }).sort_values("coefficient")
                fig_coef = px.bar(
                    coef_df, x="coefficient", y="feature", orientation="h",
                    color="coefficient",
                    color_continuous_scale=["#2ecc71", "#95a5a6", "#e74c3c"],
                )
                fig_coef.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig_coef, use_container_width=True)

    st.divider()

    # Always show the current state of predictions -- whether they were
    # just computed above, or came from an earlier run -- straight from
    # the database, rather than only right after the button's been
    # clicked in this particular browser tab.
    registrations = get_all_registrations_detailed()
    if owned_event_ids is not None:
        registrations = [r for r in registrations if r["event_id"] in owned_event_ids]
    upcoming = [r for r in registrations if r["attendance_status"] == "registered"]

    st.subheader("Registrations awaiting check-in")

    if not upcoming:
        st.info("No registrations are currently awaiting check-in.")
        return

    scored_upcoming = [r for r in upcoming if r["predicted_no_show"] is not None]
    high_risk_count = sum(1 for r in scored_upcoming if r["predicted_no_show"] >= 0.6)

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Awaiting check-in", len(upcoming))
    kpi2.metric("Scored", len(scored_upcoming))
    kpi3.metric("High risk (60%+)", high_risk_count)

    if len(scored_upcoming) < len(upcoming):
        st.caption(
            f"{len(upcoming) - len(scored_upcoming)} registration(s) haven't "
            "been scored yet -- click \"Run No-Show Prediction\" above."
        )

    table_rows = []
    for r in sorted(
        upcoming,
        key=lambda r: (r["predicted_no_show"] is None, -(r["predicted_no_show"] or 0)),
    ):
        table_rows.append({
            "Guest": r["guest_name"],
            "Event": r["event_name"],
            "Event Date": date.fromisoformat(r["event_date"]).strftime("%d-%m-%Y"),
            "Tag": r["tag"] or "--",
            "No-Show Risk": (
                f"{r['predicted_no_show'] * 100:.0f}%"
                if r["predicted_no_show"] is not None else "--"
            ),
            "Risk Level": _risk_level(r["predicted_no_show"]),
        })

    st.dataframe(table_rows, use_container_width=True)


def render_attendance_trend_section(owned_event_ids=None):
    st.header("Attendance rate over time")
    st.write(
        "Attended divided by registered, for every past event with a "
        "known outcome, plotted in chronological order. This is the "
        "same historical numbers the turnout forecast below trains on "
        "-- shown on their own, with no model or forecast line mixed "
        "in."
    )

    # Reuses the same query the turnout forecast trains on (already
    # imported above) -- no need for a second db.py function just to
    # look at the same rows a different way.
    event_rows = get_events_for_forecast()
    resolved_rows = [e for e in event_rows if _is_resolved_event(e)]
    if owned_event_ids is not None:
        resolved_rows = [e for e in resolved_rows if e["event_id"] in owned_event_ids]

    if not resolved_rows:
        st.info(
            "No fully-resolved past events yet -- once every "
            "registration for an event is checked in or marked "
            "no-show, that event will show up here."
        )
        return

    rows = []
    for e in resolved_rows:
        rows.append({
            "event_name": e["event_name"],
            "event_date": e["event_date"],
            "tag": e["tag"] or "--",
            "registered": e["registered_count"],
            "attended": e["attended_count"],
            "rate": e["attended_count"] / e["registered_count"],
        })
    df = pd.DataFrame(rows).sort_values("event_date")
    df["label"] = df.apply(
        lambda row: f"{row['event_name']} ({date.fromisoformat(row['event_date']).strftime('%d-%m-%Y')})",
        axis=1,
    )

    kpi1, kpi2 = st.columns(2)
    kpi1.metric("Resolved events", len(df))
    kpi2.metric("Average attendance rate", f"{df['rate'].mean() * 100:.0f}%")

    fig_trend = px.line(
        df, x="label", y="rate", markers=True,
        hover_data={"tag": True, "registered": True, "attended": True},
        labels={"rate": "Attendance rate", "label": "Event", "tag": "Tag"},
    )
    fig_trend.update_layout(xaxis_tickangle=-30, yaxis_tickformat=".0%")
    st.plotly_chart(fig_trend, use_container_width=True)


def render_turnout_section(owned_event_ids=None):
    st.header("Turnout forecast")
    st.write(
        "Fits a trend line (via Prophet) through past events' turnout "
        "rates over time system-wide -- what fraction of registrants "
        "actually showed up -- adjusted per event tag, and projects "
        "that forward onto every upcoming event. Only events where "
        "every registration has a known outcome (attended or no-show) "
        "count as training data."
        + (
            " Same reasoning as the no-show model above: training uses "
            "every organizer's past events so there's enough history to "
            "fit a trend, even if you've only run a couple yourself -- "
            "the table below only shows the forecast for *your* "
            "upcoming events."
            if owned_event_ids is not None else ""
        )
    )

    if st.button("Run Turnout Forecast", key="turnout_run_button"):
        with st.spinner("Fitting trend and forecasting upcoming events..."):
            event_rows = get_events_for_forecast()
            result = train_and_predict_turnout(event_rows)

        if result["status"] == "not_enough_data":
            st.warning(
                f"Only {result['train_events']} fully-resolved past event(s) "
                f"so far (need at least {MIN_TRAINING_EVENTS} -- every "
                "registration for an event has to be checked in or marked "
                "no-show for that event to count). Add more sample data on "
                "the Data Tools page, or check in more past events, then "
                "try again."
            )
        elif result["status"] == "no_predictions_needed":
            st.info(
                "Nothing to forecast right now -- every event is already "
                "fully resolved. Forecasts show up here once a new event "
                "is on the calendar."
            )
        else:
            bulk_set_predicted_turnout(result["predictions"])
            st.success(
                f"Forecasted {len(result['predictions'])} upcoming event(s), "
                f"trained on {result['train_events']} past ones."
            )

    st.divider()

    # Always show the current state of forecasts -- whether just computed
    # above, or from an earlier run -- straight from the database, same
    # reasoning as the other sections above.
    event_rows = get_events_for_forecast()
    if owned_event_ids is not None:
        event_rows = [e for e in event_rows if e["event_id"] in owned_event_ids]

    if not event_rows:
        st.info("No events yet.")
        return

    rows = []
    for e in event_rows:
        resolved = _is_resolved_event(e)
        rows.append({
            "event_name": e["event_name"],
            "event_date": e["event_date"],
            "tag": e["tag"] or "--",
            "registered": e["registered_count"],
            "attended": e["attended_count"] if resolved else None,
            "actual_rate": (e["attended_count"] / e["registered_count"]) if resolved else None,
            "predicted_turnout": e["predicted_turnout"],
            "status": "Resolved" if resolved else "Upcoming",
        })
    df = pd.DataFrame(rows)

    resolved_count = int((df["status"] == "Resolved").sum())
    upcoming_count = int((df["status"] == "Upcoming").sum())
    scored_upcoming = int(df.loc[df["status"] == "Upcoming", "predicted_turnout"].notna().sum())

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Past events (resolved)", resolved_count)
    kpi2.metric("Upcoming events", upcoming_count)
    kpi3.metric("Forecasted", scored_upcoming)

    if upcoming_count and scored_upcoming < upcoming_count:
        st.caption(
            f"{upcoming_count - scored_upcoming} upcoming event(s) haven't "
            "been forecasted yet -- click \"Run Turnout Forecast\" above."
        )

    st.subheader("Actual vs. forecasted turnout")

    chart_rows = []
    for _, r in df.iterrows():
        if r["status"] == "Resolved":
            chart_rows.append({"event": r["event_name"], "date": r["event_date"],
                                "rate": r["actual_rate"], "type": "Actual"})
        elif pd.notna(r["predicted_turnout"]):
            chart_rows.append({"event": r["event_name"], "date": r["event_date"],
                                "rate": r["predicted_turnout"], "type": "Forecast"})

    if chart_rows:
        chart_df = pd.DataFrame(chart_rows).sort_values("date")
        chart_df["label"] = chart_df.apply(
            lambda row: f"{row['event']} ({date.fromisoformat(row['date']).strftime('%d-%m-%Y')})",
            axis=1,
        )
        fig_turnout = px.bar(
            chart_df, x="label", y="rate", color="type",
            color_discrete_map={"Actual": "#3498db", "Forecast": "#e67e22"},
            labels={"rate": "Turnout rate", "label": "Event", "type": "Type"},
        )
        fig_turnout.update_layout(xaxis_tickangle=-30, yaxis_tickformat=".0%")
        st.plotly_chart(fig_turnout, use_container_width=True)
    else:
        st.info("No actual or forecasted turnout to chart yet.")

    st.divider()

    st.subheader("All events")
    display_rows = []
    for _, r in df.sort_values("event_date").iterrows():
        display_rows.append({
            "Event": r["event_name"],
            "Date": date.fromisoformat(r["event_date"]).strftime("%d-%m-%Y"),
            "Tag": r["tag"],
            "Registered": r["registered"],
            "Attended": r["attended"] if pd.notna(r["attended"]) else "--",
            "Actual Turnout": f"{r['actual_rate'] * 100:.0f}%" if pd.notna(r["actual_rate"]) else "--",
            "Forecasted Turnout": (
                f"{r['predicted_turnout'] * 100:.0f}%"
                if pd.notna(r["predicted_turnout"]) else "--"
            ),
            "Status": r["status"],
        })
    st.dataframe(display_rows, use_container_width=True)
