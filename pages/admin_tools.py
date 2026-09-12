"""
pages/admin_tools.py
----------------------
Admin-only utility page -- not part of the guest-facing flow.

Two tools:
1. Add sample data: generates a realistic demo dataset (events, guests,
   registrations, reviews) so the dashboard doesn't look empty during
   development and testing.
2. Reset all data: wipes everything back to a clean, empty state.
"""

import streamlit as st
from utils.db import (
    init_db, get_table_counts, reset_all_data, seed_sample_data,
    get_bypass_setting, set_bypass_setting,
)

init_db()

st.title("Data Tools")
st.caption("Admin-only utilities for demoing and testing -- not something a guest would ever see.")

counts = get_table_counts()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Events", counts["events"])
col2.metric("Guests", counts["guests"])
col3.metric("Registrations", counts["registrations"])
col4.metric("Reviews", counts["reviews"])

st.divider()

# ---------------------------------------------------------------------
# Add sample data
# ---------------------------------------------------------------------
st.subheader("Add sample data")
st.write(
    "Generates 10-20 events spread across past and future dates, a pool of "
    "guests registered across them, a realistic mix of attended / no-show "
    "outcomes for past events, and reviews from some of the guests who "
    "attended. Safe to click more than once -- it adds on top of whatever's "
    "already there rather than replacing it."
)
st.caption(
    "Every generated review is scored by the real sentiment model "
    "(same one the Leave a Review page uses) -- the first click after "
    "starting the app will take longer than usual while that model "
    "loads, and its very first run ever on this machine also has to "
    "download it (~260MB, needs internet just that once)."
)

if st.button("Add Sample Data"):
    with st.spinner("Generating sample data and scoring reviews..."):
        summary = seed_sample_data()
    st.success(
        f"Added {summary['events']} events, {summary['guests']} guests, "
        f"{summary['registrations']} registrations, and {summary['reviews']} reviews."
    )
    st.rerun()  # Refresh so the counts above reflect what was just added.

st.divider()

# ---------------------------------------------------------------------
# Reset all data (kill switch)
# ---------------------------------------------------------------------
st.subheader("Reset all data")
st.write("Wipes every event, guest, registration, and review, and resets IDs back to 1.")

# Streamlit reruns this whole file top-to-bottom on every click, so a
# plain Python variable can't "remember" that Reset was already clicked
# once -- it resets to its starting value every rerun. st.session_state
# is a dict that *does* persist across reruns for the same browser
# session, which is exactly what a two-step confirm needs: click once
# to see the warning, click again to actually delete.
if "confirm_reset" not in st.session_state:
    st.session_state.confirm_reset = False

if not st.session_state.confirm_reset:
    if st.button("Reset All Data"):
        st.session_state.confirm_reset = True
        st.rerun()
else:
    st.warning(
        "This permanently deletes ALL events, guests, registrations, and "
        "reviews. This cannot be undone."
    )
    confirm_col, cancel_col = st.columns(2)
    with confirm_col:
        if st.button("Yes, delete everything", type="primary"):
            reset_all_data()
            st.session_state.confirm_reset = False
            st.success("All data cleared.")
            st.rerun()
    with cancel_col:
        if st.button("Cancel"):
            st.session_state.confirm_reset = False
            st.rerun()

st.divider()

# ---------------------------------------------------------------------
# Testing settings
# ---------------------------------------------------------------------
st.subheader("Testing settings")
st.write(
    "Reviews are normally only open to guests checked in as attended, "
    "once the event has passed. Turn this on to skip both checks, so "
    "the review form can be tested without a real past event and a "
    "checked-in guest on hand."
)

# Stored in the database (see get_bypass_setting/set_bypass_setting in
# db.py) rather than st.session_state, so it stays on even when the
# guest review link is tested in a separate browser tab or window --
# session state only lives inside one browser session, but this needs
# to be visible to a completely different "guest" session too.
bypass_enabled = st.checkbox(
    "Bypass review eligibility (testing)",
    value=get_bypass_setting(),
)
set_bypass_setting(bypass_enabled)

if bypass_enabled:
    st.caption("⚠️ Bypass is ON -- any ticket code can leave a review right now.")
else:
    st.caption("Bypass is OFF -- normal attended + event-over rule applies.")