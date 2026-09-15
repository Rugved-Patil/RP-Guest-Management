"""
pages/organizer_events.py
---------------------------
Organizer-facing page: create events (auto-owned by whoever's logged
in) and manage the ones already created. Mirrors Admin's Create Event
page, with two differences:

1. Every event created here is automatically stamped with the
   logged-in Organizer's user_id as created_by, and the "Existing
   events" list below only shows events owned by that same Organizer
   -- not every event system-wide.
2. Deleting an event here cascades: it also deletes that event's
   registrations and reviews, rather than being blocked by them like
   Admin's delete does. An Organizer is expected to be able to fully
   remove an event they created, registrations and all -- see
   delete_event()'s docstring in utils/db.py for the cascade=True
   details.
"""

import streamlit as st
from datetime import date, time
from utils.db import (
    init_db,
    add_event,
    get_events_by_owner,
    get_registration_count,
    delete_event,
    TAG_OPTIONS,
)
from utils.auth import require_role

init_db()
require_role("organizer")

owner_id = st.session_state.user_id

st.title("Manage My Events")

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
            # We store the date as ISO text (YYYY-MM-DD), NOT dd-mm-yyyy --
            # see admin_events.py for why. created_by=owner_id is what
            # makes this event show up under this Organizer's own views
            # (this page, Check-In, Guest List, Analytics) and nobody
            # else's.
            add_event(
                event_name.strip(),
                event_date.isoformat(),
                start_time.strftime("%H:%M"),
                end_time.strftime("%H:%M"),
                tag,
                created_by=owner_id,
            )
            st.success(f"Event '{event_name}' created.")

st.divider()
st.subheader("My events")

events = get_events_by_owner(owner_id)

if not events:
    st.info("You haven't created any events yet -- use the form above to create one.")
    st.stop()

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
st.caption(
    "Unlike Admin's Create Event page, this permanently deletes every "
    "registration and review tied to the event too -- not just the "
    "event itself."
)

event_labels = {
    f"{e['event_name']} -- {date.fromisoformat(e['event_date']).strftime('%d-%m-%Y')}": e["event_id"]
    for e in events
}
selected_label = st.selectbox("Select event to delete", list(event_labels.keys()))
selected_event_id = event_labels[selected_label]
reg_count = get_registration_count(selected_event_id)

delete_key = f"confirm_delete_event_{selected_event_id}"
if delete_key not in st.session_state:
    st.session_state[delete_key] = False

if not st.session_state[delete_key]:
    if st.button("Delete event", type="primary", key=f"delete_btn_{selected_event_id}"):
        st.session_state[delete_key] = True
        st.rerun()
else:
    if reg_count:
        st.warning(
            f"This permanently deletes '{selected_label.split(' -- ')[0]}' "
            f"along with {reg_count} registration(s) and any reviews "
            "attached to them. This cannot be undone."
        )
    else:
        st.warning(
            f"This permanently deletes '{selected_label.split(' -- ')[0]}'. "
            "This cannot be undone."
        )
    confirm_col, cancel_col = st.columns(2)
    with confirm_col:
        if st.button("Yes, delete", type="primary", key=f"confirm_delete_btn_{selected_event_id}"):
            delete_event(selected_event_id, cascade=True)
            st.session_state[delete_key] = False
            st.success("Event deleted.")
            st.rerun()
    with cancel_col:
        if st.button("Cancel", key=f"cancel_delete_btn_{selected_event_id}"):
            st.session_state[delete_key] = False
            st.rerun()
