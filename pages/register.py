"""
pages/register.py
-------------------
Guest-facing page: book a spot at an event.

Flow: guest picks an event, fills in their details, submits.
On success, we look them up (or create them) as a Guest, create a
Registration row, and show them their ticket code.
"""

import streamlit as st
from utils.db import init_db, get_all_events, get_guest_by_email, add_guest, add_registration

init_db()

st.title("Event Registration")

events = get_all_events()

if not events:
    st.warning("There are no events open for registration yet. Check back soon.")
    st.stop()  # Stops the page here -- no point showing a form with no events.

# Build a simple label like "Product Launch -- 2026-09-20" for each event,
# and keep a lookup back to the real event_id for when we save.
event_options = {f"{e['event_name']} -- {e['event_date']}": e["event_id"] for e in events}

with st.form("register_form"):
    selected_label = st.selectbox("Which event are you registering for?", list(event_options.keys()))
    name = st.text_input("Full name")
    email = st.text_input("Email")
    phone = st.text_input("Phone number")
    submitted = st.form_submit_button("Register")

    if submitted:
        # Basic required-field checks. Real duplicate detection (fuzzy
        # matching similar names/emails) is a separate step we'll add
        # after this form is working end to end.
        if not name.strip() or not email.strip():
            st.error("Name and email are required.")
        else:
            event_id = event_options[selected_label]

            # If this email is already a known guest, reuse that guest_id
            # instead of creating a duplicate row in the guests table.
            existing = get_guest_by_email(email.strip())
            if existing:
                guest_id = existing["guest_id"]
            else:
                guest_id = add_guest(name.strip(), email.strip(), phone.strip())

            ticket_code = add_registration(guest_id, event_id)

            st.success("You're registered! Save your ticket code below.")
            st.metric("Your ticket code", ticket_code)
            st.caption("You'll need this code for check-in at the event.")