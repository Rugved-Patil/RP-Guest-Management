"""
pages/organizer_guests.py
---------------------------
Organizer-facing page: view guests registered for this Organizer's
own events. Same filter/search table as the top of Admin's Manage
Guests page, scoped to events this Organizer created -- but read-only.
No edit or delete here on purpose: guests are shared records across
the whole system (the same guest might be registered for other
organizers' events too), so editing or deleting them isn't something
any one Organizer should be able to do. That, and duplicate-guest
resolution, stay Admin-only, on admin_manage.py.
"""

import streamlit as st
from datetime import date, datetime
from utils.db import init_db, get_all_registrations_detailed, get_events_by_owner
from utils.auth import require_role

init_db()
require_role("organizer")

owner_id = st.session_state.user_id
owned_events = get_events_by_owner(owner_id)
owned_event_ids = {e["event_id"] for e in owned_events}

st.title("Guest List")

if not owned_event_ids:
    st.info("You haven't created any events yet -- create one on the Manage My Events page first.")
    st.stop()

registrations = [
    r for r in get_all_registrations_detailed() if r["event_id"] in owned_event_ids
]

if not registrations:
    st.info("No one has registered for your events yet.")
    st.stop()

# --- Build filter options (only this Organizer's own events) -----------

def _event_label(e):
    return f"{e['event_name']} -- {date.fromisoformat(e['event_date']).strftime('%d-%m-%Y')}"

event_id_by_label = {_event_label(e): e["event_id"] for e in owned_events}
event_options = ["All events"] + list(event_id_by_label.keys())

filter_col, status_col, search_col = st.columns([2, 1, 2])
with filter_col:
    selected_event_label = st.selectbox("Filter by event", event_options)
with status_col:
    status_filter = st.selectbox(
        "Attendance status", ["All", "registered", "attended", "no_show"]
    )
with search_col:
    search_text = st.text_input("Search by name or email").strip().lower()

selected_event_id = event_id_by_label.get(selected_event_label)  # None means "All events"

# --- Apply filters -------------------------------------------------------
# Same "filter a plain Python list" approach as admin_manage.py -- the
# dataset (already narrowed to this Organizer's events) is small enough
# that this is simpler than writing separate SQL per filter combo.

filtered = registrations

if selected_event_id is not None:
    filtered = [r for r in filtered if r["event_id"] == selected_event_id]

if status_filter != "All":
    filtered = [r for r in filtered if r["attendance_status"] == status_filter]

if search_text:
    filtered = [
        r for r in filtered
        if search_text in r["guest_name"].lower() or search_text in r["email"].lower()
    ]

st.caption(f"Showing {len(filtered)} of {len(registrations)} registrations.")

# --- Build the table -----------------------------------------------------

display_rows = []
for r in filtered:
    checked_in_display = "--"
    if r["checked_in_at"]:
        checked_in_display = datetime.fromisoformat(r["checked_in_at"]).strftime("%d-%m-%Y %H:%M")

    display_rows.append({
        "Guest": r["guest_name"],
        "Email": r["email"],
        "Phone": r["phone"] or "--",
        "Event": r["event_name"],
        "Event Date": date.fromisoformat(r["event_date"]).strftime("%d-%m-%Y"),
        "Ticket Code": r["ticket_code"],
        "Attendance": r["attendance_status"],
        "Checked In": checked_in_display,
    })

# st.dataframe supports sorting by clicking a column header out of the
# box -- no extra sorting code needed here for that.
st.dataframe(display_rows, use_container_width=True)
