"""
pages/organizer_checkin.py
----------------------------
Organizer-facing page: check a guest in at the door, same as Admin's
Check-In page, but scoped to events this Organizer created.

mark_attendance_by_ticket() itself doesn't know or care who owns an
event -- so before calling it, this page looks the ticket up first
and checks the registration's event is one of this Organizer's own.
That way an Organizer can never accidentally (or deliberately) check
a guest in for someone else's event, even if they somehow got hold of
a ticket code for it.
"""

import streamlit as st
from datetime import datetime
from utils.db import init_db, get_events_by_owner, get_registration_by_ticket, mark_attendance_by_ticket
from utils.auth import require_role

init_db()
require_role("organizer")

owner_id = st.session_state.user_id
owned_event_ids = {e["event_id"] for e in get_events_by_owner(owner_id)}

st.title("Check-In")
st.write("Enter the guest's ticket code and check them in.")

if not owned_event_ids:
    st.info("You haven't created any events yet -- create one on the Manage My Events page first.")
    st.stop()

# clear_on_submit=True: check-in happens over and over in a row for a
# stream of guests at the door, so clearing the box after each one lets
# the organizer go straight into typing the next code without deleting
# the old one by hand -- same as admin_attendance.py.
with st.form("check_in_form", clear_on_submit=True):
    ticket_code = st.text_input("Ticket code").strip().upper()
    submitted = st.form_submit_button("Check in", type="primary")

    if submitted:
        if not ticket_code:
            st.error("Please enter a ticket code.")
        else:
            registration = get_registration_by_ticket(ticket_code)

            if registration is None:
                st.error(
                    f"No registration found for ticket code '{ticket_code}'. "
                    "Double-check and try again."
                )
            elif registration["event_id"] not in owned_event_ids:
                st.error(
                    f"Ticket code '{ticket_code}' is for an event you "
                    "didn't create. You can only check guests in for "
                    "your own events."
                )
            else:
                result = mark_attendance_by_ticket(ticket_code)

                if result["status"] == "already_checked_in":
                    reg = result["registration"]
                    checked_in_display = "an earlier time"
                    if reg["checked_in_at"]:
                        checked_in_display = datetime.fromisoformat(
                            reg["checked_in_at"]
                        ).strftime("%d-%m-%Y %H:%M")
                    st.info(
                        f"**{reg['guest_name']}** is already checked in for "
                        f"**{reg['event_name']}** (at {checked_in_display})."
                    )

                elif result["status"] == "checked_in":
                    reg = result["registration"]
                    st.success(f"**{reg['guest_name']}** checked in for **{reg['event_name']}**. Welcome!")
                    st.balloons()
