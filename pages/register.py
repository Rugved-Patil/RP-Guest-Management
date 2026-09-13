"""
pages/register.py
-------------------
Guest-facing page: book a spot at an event.

Flow: guest picks an event, fills in their details, submits.
On success, we look them up (or create them) as a Guest, create a
Registration row, and show them their ticket code.

Only events that haven't happened yet are offered here (see the date
filter below) -- get_all_events() itself returns every event ever
created, since other pages (Manage Guests, Analytics) need to see past
ones too. A guest who's already registered for the event they pick is
stopped before a second registration gets created; see
has_registration() in utils/db.py.
"""

import streamlit as st
from datetime import date
from utils.db import (
    init_db,
    get_all_events,
    get_or_create_guest,
    add_registration,
    has_registration,
)
from utils.auth import require_role

init_db()
require_role("guest")

st.title("Event Registration")

events = get_all_events()
events = [e for e in events if date.fromisoformat(e["event_date"]) >= date.today()]

if not events:
    st.warning("There are no events open for registration yet. Check back soon.")
    st.stop()  # Stops the page here -- no point showing a form with no events.

# Build a simple label like "Movie Premiere Night -- 20-09-2026" for each
# event (same dd-mm-yyyy format used on every other page), and keep a
# lookup back to the real event_id for when we save.
def _event_label(e):
    return f"{e['event_name']} -- {date.fromisoformat(e['event_date']).strftime('%d-%m-%Y')}"

event_options = {_event_label(e): e["event_id"] for e in events}

with st.form("register_form"):
    selected_label = st.selectbox("Which event are you registering for?", list(event_options.keys()))
    name = st.text_input("Full name")
    email = st.text_input("Email")
    phone = st.text_input("Phone number")
    submitted = st.form_submit_button("Register")

    if submitted:
        # Basic required-field checks.
        if not name.strip() or not email.strip():
            st.error("Name and email are required.")
        else:
            event_id = event_options[selected_label]

            # Handles the exact-email-match reuse and the fuzzy-name
            # duplicate flagging in one place -- see get_or_create_guest()
            # in utils/db.py for the actual logic. The guest sees nothing
            # different regardless of which path it takes.
            guest_id = get_or_create_guest(name.strip(), email.strip(), phone.strip())

            if has_registration(guest_id, event_id):
                st.warning(
                    "You're already registered for this event -- check "
                    "your earlier confirmation for your ticket code. "
                    "Registering again won't create a second one."
                )
            else:
                ticket_code = add_registration(guest_id, event_id)

                st.success("You're registered! Save your ticket code below.")
                st.metric("Your ticket code", ticket_code)
                st.caption("You'll need this code for check-in at the event.")