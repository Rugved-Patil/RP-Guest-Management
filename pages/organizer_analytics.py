"""
pages/organizer_analytics.py
------------------------------
Organizer-facing page: sentiment, no-show risk, attendance trend, and
turnout forecast -- scoped to events this Organizer created. Built on
the same render functions admin_analytics.py uses (see
utils/analytics_sections.py), just passed this Organizer's own
event_id set so each section filters down to what they're allowed to
see.

Guest segmentation isn't here -- it's a guest-level tag based on their
attendance across the whole system, not any one Organizer's events,
so it stays Admin-only. See the note at the top of
utils/analytics_sections.py for the full reasoning.
"""

import streamlit as st
from utils.db import init_db, get_events_by_owner
from utils.analytics_sections import (
    render_sentiment_section,
    render_noshow_section,
    render_attendance_trend_section,
    render_turnout_section,
)
from utils.auth import require_role

init_db()
require_role("organizer")

owner_id = st.session_state.user_id
owned_event_ids = {e["event_id"] for e in get_events_by_owner(owner_id)}

st.title("My Analytics")

if not owned_event_ids:
    st.info("You haven't created any events yet -- create one on the Manage My Events page first.")
    st.stop()

st.caption(
    "Everything below is scoped to your own events. (Guest segmentation "
    "isn't shown here -- it tags a guest based on their attendance "
    "across the whole system, not any single organizer's events, so "
    "it's an Admin-only view.)"
)

render_sentiment_section(owned_event_ids)
st.divider()
render_noshow_section(owned_event_ids)
st.divider()
render_attendance_trend_section(owned_event_ids)
st.divider()
render_turnout_section(owned_event_ids)
