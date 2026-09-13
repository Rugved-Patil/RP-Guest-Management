"""
pages/admin_events.py
----------------------
Admin-facing page: create, view, and delete events.
"""

import streamlit as st
from datetime import date, time
from utils.db import init_db, add_event, get_all_events, delete_event
from utils.auth import require_role

init_db()
require_role("admin")

# Fixed set of event categories. Also used as a feature by the
# no-show and turnout-forecasting models in utils/ml.py, so adding a
# new tag here means it'll show up as a new one-hot column for both --
# not just a UI-level change.
TAG_OPTIONS = ["Movie", "Play", "Sports", "Dance"]

st.title("Create Event")

with st.form("create_event_form", clear_on_submit=True):
    event_name = st.text_input("Event name")

    # format="DD-MM-YYYY" only changes how the date LOOKS in the widget --
    # Streamlit still hands us back a normal Python date object either way,
    # which we convert to ISO text ourselves before saving (see note below).
    event_date = st.date_input("Event date", value=date.today(), format="DD-MM-YYYY")

    col1, col2 = st.columns(2)
    with col1:
        start_time = st.time_input("Start time", value=time(18, 0))
    with col2:
        end_time = st.time_input("End time", value=time(20, 0))

    tag = st.selectbox("Category", TAG_OPTIONS)

    submitted = st.form_submit_button("Create event")

    if submitted:
        if not event_name.strip():
            st.error("Please enter an event name.")
        elif end_time <= start_time:
            st.error("End time must be after start time.")
        else:
            # We store the date as ISO text (YYYY-MM-DD), NOT dd-mm-yyyy.
            # ISO text sorts correctly with a plain SQL ORDER BY (so
            # "Existing events" and the registration dropdown list events
            # in the right order); dd-mm-yyyy text would sort wrong because
            # '25-12-2026' would come before '03-01-2027' alphabetically.
            # We only ever show dd-mm-yyyy in the UI, not in storage.
            add_event(
                event_name.strip(),
                event_date.isoformat(),
                start_time.strftime("%H:%M"),
                end_time.strftime("%H:%M"),
                tag,
            )
            st.success(f"Event '{event_name}' created.")

st.divider()
st.subheader("Existing events")

events = get_all_events()

if events:
    display_rows = []
    for e in events:
        display_rows.append({
            "Event": e["event_name"],
            "Date": date.fromisoformat(e["event_date"]).strftime("%d-%m-%Y"),
            "Time": f"{e['start_time'] or '--'} - {e['end_time'] or '--'}",
            "Tag": e["tag"] or "--",
        })
    st.dataframe(display_rows, use_container_width=True)

    st.divider()
    st.subheader("Delete an event")

    event_labels = {
        f"{e['event_name']} -- {date.fromisoformat(e['event_date']).strftime('%d-%m-%Y')}": e["event_id"]
        for e in events
    }
    selected_label = st.selectbox("Select event to delete", list(event_labels.keys()))

    if st.button("Delete event", type="primary"):
        event_id = event_labels[selected_label]
        deleted = delete_event(event_id)
        if deleted:
            st.success("Event deleted.")
            st.rerun()  # Refresh so the deleted event drops off the list/dropdown.
        else:
            st.error(
                "Can't delete this event -- one or more guests are already "
                "registered for it. Deleting it would orphan their tickets."
            )
else:
    st.info("No events yet -- create one above.")